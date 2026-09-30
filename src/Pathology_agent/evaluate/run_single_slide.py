#!/usr/bin/env python3
"""Run WSI agents on a single slide or ROI collection (headless, no GUI).

Usage:
    # AML auto mode (ROI collection + diagnosis):
    python evaluate/run_single_slide.py \
        --slide /path/to/patient.mrxs \
        --output-dir ./batch_outputs \
        --agent aml_auto

    # AML ROI collector only:
    python evaluate/run_single_slide.py \
        --slide /path/to/patient.mrxs \
        --output-dir ./batch_outputs \
        --agent aml_roi

    # AML diagnosis only (from existing ROI collection):
    python evaluate/run_single_slide.py \
        --output-dir ./batch_outputs \
        --agent vlm_diagnosis \
        --roi-input-path /path/to/roi_collection.json

    [Optional: --model GLM-4.6V-FP8, --extractor uni2, --experiment-root ./cache, --tile-filter hybrid]
"""

import argparse
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsi_core_pkg.aml_output import extract_final_decision as shared_extract_final_decision
from wsi_core_pkg.aml_output import persist_aml_slide_bundle

MIRAX_EXTS = {".mrxs", ".mrsx"}
FINAL_DECISIONS = (
    "Normal Bone Marrow",
    "Acute Leukemia",
    "Normal marrow",
    "Acute leukemia",
    "Call for more diagnostics",
)
FINAL_DECISION_LOOKUP = {label.lower(): label for label in FINAL_DECISIONS}
try:
    from wsi_core_pkg.tuning_config import tuning_value
    DEFAULT_MPP_UM_FALLBACK = float(tuning_value("tools.slide", "DEFAULT_MPP_UM"))
    DEFAULT_MAX_ACCEPTED_ROIS = int(tuning_value("tools.navigation", "MAX_ACCEPTED_ROIS"))
    DEFAULT_TARGET_ACCEPTED_ROIS = int(tuning_value("tools.navigation", "TARGET_ACCEPTED_ROIS"))
    try:
        CONFIG_CACHE_ROOT_DIR = str(tuning_value("tools.cache", "CACHE_ROOT_DIR") or "").strip()
    except Exception:
        CONFIG_CACHE_ROOT_DIR = ""
except Exception:
    DEFAULT_MPP_UM_FALLBACK = 0.159
    DEFAULT_MAX_ACCEPTED_ROIS = 10
    DEFAULT_TARGET_ACCEPTED_ROIS = 5
    CONFIG_CACHE_ROOT_DIR = ""


def _sanitize_output_component(value: str, default: str = "run") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "-", text)
    text = text.strip(".-_")
    return text or default


def _make_run_id(patient_name: str, model_name: str = "", extractor_name: str = "", tile_size_px: int | str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_parts = [ts, uuid.uuid4().hex[:8]]
    if model_name:
        run_parts.append(_sanitize_output_component(model_name, "model"))
    if extractor_name:
        run_parts.append(_sanitize_output_component(extractor_name, "extractor"))
    if tile_size_px not in (None, ""):
        run_parts.append(f"{_sanitize_output_component(str(tile_size_px), 'tile')}px")
    return "_".join(run_parts)


def _patient_output_name(patient_name: str, model_name: str, *, include_model: bool = False) -> str:
    if not include_model:
        return patient_name
    model_tag = _sanitize_output_component(model_name, "model")
    patient_tag = _sanitize_output_component(patient_name, "slide")
    return f"{model_tag}_{patient_tag}"


def _read_slide_mpp_um(slide_path: str) -> tuple[float | None, str | None]:
    try:
        import openslide
    except Exception:
        return None, None

    slide = None
    try:
        slide = openslide.open_slide(str(slide_path))
        props = getattr(slide, "properties", {}) or {}

        for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
            value = props.get(key)
            if value:
                try:
                    return float(value), key
                except Exception:
                    pass

        slide_comment = props.get("openslide.comment", "")
        match = re.search(r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>", slide_comment)
        if match is not None:
            try:
                return float(match.group(1)), "openslide.comment:PixelSizeMicrons"
            except Exception:
                pass

        xml_text = props.get("tiff.ImageDescription")
        if xml_text:
            try:
                doc = minidom.parseString(xml_text)
                images = doc.documentElement.getElementsByTagName("Image")
                pixels = images[0].getElementsByTagName("Pixels")
                physical_size_x = pixels[0].getAttribute("PhysicalSizeX")
                if physical_size_x:
                    return float(physical_size_x), "tiff.ImageDescription:PhysicalSizeX"
            except Exception:
                pass

        objective_power = props.get("openslide.objective-power")
        if objective_power:
            try:
                return 10.0 / float(objective_power), "openslide.objective-power"
            except Exception:
                pass
    except Exception:
        return None, None
    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass

    return None, None


def _resolve_tile_size_config(
    *,
    slide_path: str,
    tile_size_px: int,
    requested_tile_size_um: float | None,
    requested_default_mpp_um: float | None,
) -> dict[str, float | str | None]:
    slide_mpp_um, mpp_source = _read_slide_mpp_um(slide_path)
    if requested_default_mpp_um is not None and float(requested_default_mpp_um) > 0:
        resolved_mpp_um = float(requested_default_mpp_um)
        effective_mpp_source = "input_default_mpp"
    elif slide_mpp_um and slide_mpp_um > 0:
        resolved_mpp_um = float(slide_mpp_um)
        effective_mpp_source = str(mpp_source or "slide_metadata")
    else:
        resolved_mpp_um = DEFAULT_MPP_UM_FALLBACK
        effective_mpp_source = "default_fallback"

    if requested_tile_size_um is not None:
        tile_size_um = float(requested_tile_size_um)
        tile_size_um_source = "explicit"
    else:
        tile_size_um = float(tile_size_px) * float(resolved_mpp_um)
        if effective_mpp_source == "input_default_mpp":
            tile_size_um_source = "auto_from_input_mpp"
        elif slide_mpp_um and slide_mpp_um > 0:
            tile_size_um_source = "auto_from_slide_mpp"
        else:
            tile_size_um_source = "auto_from_default_mpp"

    return {
        "tile_size_um": tile_size_um,
        "tile_size_um_requested": requested_tile_size_um,
        "tile_size_um_source": tile_size_um_source,
        "default_mpp_um_requested": requested_default_mpp_um,
        "slide_mpp_um": slide_mpp_um,
        "resolved_mpp_um": resolved_mpp_um,
        "mpp_source": effective_mpp_source,
        "default_mpp_um_fallback": DEFAULT_MPP_UM_FALLBACK,
    }


def _read_slide_mpp_um(slide_path: str) -> tuple[float | None, str | None]:
    try:
        import openslide
    except Exception:
        return None, None

    slide = None
    try:
        slide = openslide.open_slide(str(slide_path))
        props = getattr(slide, "properties", {}) or {}

        for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
            value = props.get(key)
            if value:
                try:
                    return float(value), key
                except Exception:
                    pass

        slide_comment = props.get("openslide.comment", "")
        match = re.search(r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>", slide_comment)
        if match is not None:
            try:
                return float(match.group(1)), "openslide.comment:PixelSizeMicrons"
            except Exception:
                pass

        xml_text = props.get("tiff.ImageDescription")
        if xml_text:
            try:
                doc = minidom.parseString(xml_text)
                images = doc.documentElement.getElementsByTagName("Image")
                pixels = images[0].getElementsByTagName("Pixels")
                physical_size_x = pixels[0].getAttribute("PhysicalSizeX")
                if physical_size_x:
                    return float(physical_size_x), "tiff.ImageDescription:PhysicalSizeX"
            except Exception:
                pass

        objective_power = props.get("openslide.objective-power")
        if objective_power:
            try:
                return 10.0 / float(objective_power), "openslide.objective-power"
            except Exception:
                pass
    except Exception:
        return None, None
    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass

    return None, None


def _resolve_tile_size_config(
    *,
    slide_path: str,
    tile_size_px: int,
    requested_tile_size_um: float | None,
    requested_default_mpp_um: float | None,
) -> dict[str, float | str | None]:
    slide_mpp_um, mpp_source = _read_slide_mpp_um(slide_path)
    if requested_default_mpp_um is not None and float(requested_default_mpp_um) > 0:
        resolved_mpp_um = float(requested_default_mpp_um)
        effective_mpp_source = "input_default_mpp"
    elif slide_mpp_um and slide_mpp_um > 0:
        resolved_mpp_um = float(slide_mpp_um)
        effective_mpp_source = str(mpp_source or "slide_metadata")
    else:
        resolved_mpp_um = DEFAULT_MPP_UM_FALLBACK
        effective_mpp_source = "default_fallback"

    if requested_tile_size_um is not None:
        tile_size_um = float(requested_tile_size_um)
        tile_size_um_source = "explicit"
    else:
        tile_size_um = float(tile_size_px) * float(resolved_mpp_um)
        if effective_mpp_source == "input_default_mpp":
            tile_size_um_source = "auto_from_input_mpp"
        elif slide_mpp_um and slide_mpp_um > 0:
            tile_size_um_source = "auto_from_slide_mpp"
        else:
            tile_size_um_source = "auto_from_default_mpp"

    return {
        "tile_size_um": tile_size_um,
        "tile_size_um_requested": requested_tile_size_um,
        "tile_size_um_source": tile_size_um_source,
        "default_mpp_um_requested": requested_default_mpp_um,
        "slide_mpp_um": slide_mpp_um,
        "resolved_mpp_um": resolved_mpp_um,
        "mpp_source": effective_mpp_source,
        "default_mpp_um_fallback": DEFAULT_MPP_UM_FALLBACK,
    }


def _strip_tile_cache(root: Path) -> None:
    for cache_dir in (root / "tile_cache", root / "_tile_cache"):
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)


def _configure_fresh_embedding_cache(run_id: str) -> dict[str, str]:
    cache_root = REPO_ROOT / "outputs" / "_fresh_embedding_cache" / run_id
    reference_cache_dir = cache_root / "_cache" / "reference_hnsw"
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
    cache_info = {
        "cache_root": str(cache_root),
        "reference_cache_dir": str(reference_cache_dir),
    }
    return cache_info


def _cleanup_fresh_embedding_cache(cache_info: dict[str, str] | None) -> None:
    if not cache_info:
        return
    cache_root = Path(cache_info["cache_root"])
    if cache_root.exists() and cache_root.is_dir():
        shutil.rmtree(cache_root, ignore_errors=True)


def _resolve_cache_root(*, output_dir: Path, experiment_root: Path | None) -> Path:
    if experiment_root is not None:
        return experiment_root.resolve() / "_cache"
    env_value = os.getenv("CACHE_ROOT_DIR", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    configured = CONFIG_CACHE_ROOT_DIR.strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return output_dir.resolve() / "_cache"


def _configure_persistent_reference_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
) -> Path:
    cache_root = _resolve_cache_root(output_dir=output_dir, experiment_root=experiment_root)
    reference_cache_dir = cache_root / "reference_hnsw" / _sanitize_stem(extractor_name)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
    os.environ["CACHE_ROOT_DIR"] = str(cache_root)
    return reference_cache_dir


def _configure_experiment_tile_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
    tile_filter: str,
    enable_tile_cache: bool,
) -> Path | None:
    if not enable_tile_cache:
        os.environ["ROI_DISABLE_TILE_CACHE"] = "1"
        os.environ.pop("ROI_TILE_CACHE_DIR", None)
        return None

    os.environ.pop("ROI_DISABLE_TILE_CACHE", None)
    cache_root = _resolve_cache_root(output_dir=output_dir, experiment_root=experiment_root)
    tile_cache_dir = (
        cache_root
        / "tile_cache"
        / _sanitize_stem(extractor_name)
        / _sanitize_stem(_normalize_tile_filter_name(tile_filter))
    )
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROI_TILE_CACHE_DIR"] = str(tile_cache_dir)
    os.environ["CACHE_ROOT_DIR"] = str(cache_root)
    return tile_cache_dir


def _normalize_tile_filter_name(value: str | None) -> str:
    raw = str(value or "hybrid").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"coarse_to_fine", "coarse2fine"}:
        return "hybrid"
    if raw not in {"none", "coarse", "quality", "hybrid"}:
        return "hybrid"
    return raw


def _configure_experiment_feature_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
    tile_filter: str,
) -> Path:
    cache_root = _resolve_cache_root(output_dir=output_dir, experiment_root=experiment_root)
    feature_cache_dir = (
        cache_root
        / "feature_cache"
        / _sanitize_stem(extractor_name)
        / _sanitize_stem(_normalize_tile_filter_name(tile_filter))
    )
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROI_FEATURE_CACHE_DIR"] = str(feature_cache_dir)
    os.environ["CACHE_ROOT_DIR"] = str(cache_root)
    return feature_cache_dir


def _sanitize_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "image"


def _validate_slide_package(slide_path: str) -> None:
    slide = Path(slide_path).resolve()
    if not slide.is_file():
        raise FileNotFoundError(f"Slide not found: {slide}")

    if slide.suffix.lower() in MIRAX_EXTS:
        companion_dir = slide.parent / slide.stem
        if not companion_dir.is_dir():
            raise RuntimeError(
                "MIRAX slide detected (.mrxs/.mrsx) but companion data directory was not found "
                f"next to {slide.name}."
            )

    try:
        import openslide
    except Exception as exc:
        raise RuntimeError(f"OpenSlide import failed: {type(exc).__name__}: {exc}") from exc

    try:
        handle = openslide.open_slide(str(slide))
        try:
            _ = handle.level_count
            _ = handle.level_dimensions
        finally:
            handle.close()
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as exc:
        raise RuntimeError(f"Unsupported or missing image file: {slide.name}") from exc
    except openslide.OpenSlideError as exc:
        raise RuntimeError(f"Failed to open slide '{slide.name}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open slide '{slide.name}': {type(exc).__name__}: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AML agent on a single slide")
    parser.add_argument("--slide", default=None, help="Path to .mrxs file")
    parser.add_argument("--output-dir", required=True, help="Directory to collect results")
    parser.add_argument(
        "--experiment-root",
        default=None,
        help="Experiment root used for shared tile/reference cache placement across repeat runs",
    )
    parser.add_argument(
        "--cuda-device",
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for this run, e.g. 1",
    )
    parser.add_argument("--model", default="GLM-4.6V-FP8", help="VLM model name")
    parser.add_argument("--extractor", default="uni2", help="Feature extractor key")
    parser.add_argument(
        "--include-model-in-output-name",
        action="store_true",
        help="Prefix the per-slide output folder with the model name.",
    )
    parser.add_argument("--tile-filter", default="hybrid", help="Tile prefilter method")
    parser.add_argument("--agent", default="aml_auto", help="Agent mode (e.g. aml_auto, aml_auto_stamp, aml_roi, vlm_diagnosis, aml_stamp, wsi)")
    parser.add_argument("--roi-input-path", default=None, help="Path to roi_collection.json for vlm_diagnosis or aml_stamp mode")
    parser.add_argument("--roi-collection-path", default=None, help="Path where to write roi_collection.json for aml_roi mode")
    parser.add_argument(
        "--tile-size-um",
        type=float,
        default=None,
        help=f"Tile size in microns. If omitted, auto-compute from preferred MPP override when set, otherwise slide MPP; fallback MPP is {DEFAULT_MPP_UM_FALLBACK}.",
    )
    parser.add_argument("--tile-size-px", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--roi-size-px", type=int, default=2048, help="ROI crop/output size in pixels")
    parser.add_argument(
        "--max-accepted-rois",
        type=int,
        default=DEFAULT_MAX_ACCEPTED_ROIS,
        help="Hard cap on kept AML ROIs before ROI collection stops immediately.",
    )
    parser.add_argument(
        "--target-accepted-rois",
        type=int,
        default=DEFAULT_TARGET_ACCEPTED_ROIS,
        help="Soft target number of kept AML ROIs.",
    )
    parser.add_argument(
        "--default-mpp-um",
        type=float,
        default=None,
        help="Preferred MPP override. When set, this wins over slide metadata.",
    )
    parser.add_argument(
        "--fresh-embedding-cache",
        action="store_true",
        help="Use a fresh per-run reference cache namespace and delete it after the run.",
    )
    parser.add_argument(
        "--use-tile-cache",
        action="store_true",
        help="Allow on-disk ROI tile caching during evaluation. Disabled by default to save space.",
    )
    parser.add_argument(
        "--stamp-train-root",
        default="/data/private/stamp_train/All_Tiles",
        help="Root containing trained STAMP task/extractor checkpoint directories.",
    )
    parser.add_argument(
        "--stamp-repo-root",
        default="/code/source/AML_Agent_Project/STAMP",
        help="STAMP repository root used for feature encoding and model deployment.",
    )
    parser.add_argument(
        "--stamp-extractors",
        nargs="*",
        default=None,
        help="STAMP extractors to deploy. Omit to use --extractor; use 'all' to run all available trained extractors.",
    )
    parser.add_argument(
        "--stamp-tasks",
        nargs="*",
        default=["WHO_CLASSE_SIMPLE", "NPM1"],
        help="STAMP task directories to deploy.",
    )
    parser.add_argument("--stamp-device", default="auto", help="Device for STAMP feature extraction: auto, cpu, or cuda.")
    parser.add_argument("--stamp-accelerator", default="auto", help="Lightning accelerator for STAMP deployment.")
    parser.add_argument("--stamp-num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-stamp-train-valid-overlap",
        action="store_true",
        help="Disable the hard leakage guard for STAMP checkpoint train/validation patient overlap.",
    )
    args = parser.parse_args()
    args.tile_filter = _normalize_tile_filter_name(args.tile_filter)
    if args.tile_size_um is not None and float(args.tile_size_um) <= 0:
        parser.error("--tile-size-um must be > 0")
    if args.default_mpp_um is not None and float(args.default_mpp_um) <= 0:
        parser.error("--default-mpp-um must be > 0")
    if int(args.max_accepted_rois) <= 0:
        parser.error("--max-accepted-rois must be > 0")
    if int(args.target_accepted_rois) <= 0:
        parser.error("--target-accepted-rois must be > 0")
    if int(args.target_accepted_rois) > int(args.max_accepted_rois):
        args.target_accepted_rois = int(args.max_accepted_rois)

    # Normalize agent type
    agent_type_lower = (args.agent or "aml_auto").lower()
    if agent_type_lower in {"aml", "aml_auto"}:
        agent_type_lower = "aml_auto"
    elif agent_type_lower == "aml_diagnosis":
        agent_type_lower = "vlm_diagnosis"
    elif agent_type_lower == "aml_auto_stamp":
        agent_type_lower = "aml_auto_stamp"
    elif agent_type_lower == "aml_stamp":
        agent_type_lower = "aml_stamp"
    args.agent = agent_type_lower

    # For vlm_diagnosis mode, roi_input_path is required; slide is not.
    # Use the parent directory of roi_collection.json as the case/patient id.
    if args.agent == "vlm_diagnosis":
        if not args.roi_input_path:
            print("[ERROR] --roi-input-path is required for vlm_diagnosis mode", file=sys.stderr)
            return 1
        roi_input_path = Path(args.roi_input_path).resolve()
        if not roi_input_path.is_file():
            print(f"[ERROR] ROI input not found: {roi_input_path}", file=sys.stderr)
            return 1
        slide_path = None
        # Keep case ID stable for image-path inputs like <case>/images/roi_1.jpg
        if roi_input_path.parent.name == "images" and roi_input_path.parent.parent.name:
            patient_name = roi_input_path.parent.parent.name
        else:
            patient_name = roi_input_path.parent.name
    else:
        # Standard modes require a slide.
        if not args.slide:
            print("[ERROR] --slide is required unless --agent vlm_diagnosis", file=sys.stderr)
            return 1
        slide_path = os.path.abspath(args.slide)
        if not os.path.exists(slide_path):
            print(f"[ERROR] Slide not found: {slide_path}", file=sys.stderr)
            return 1
        if args.agent == "aml_stamp" and args.roi_input_path:
            roi_input_path = Path(args.roi_input_path).resolve()
            patient_name = roi_input_path.parent.name
        else:
            patient_name = Path(slide_path).stem

    patient_output_name = _patient_output_name(
        patient_name,
        args.model,
        include_model=bool(args.include_model_in_output_name),
    )
    run_id = _make_run_id(
        patient_name,
        model_name=args.model,
        extractor_name=args.extractor,
        tile_size_px=args.tile_size_px,
    )
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else None
    cache_root = _resolve_cache_root(output_dir=out_dir, experiment_root=experiment_root)

    if args.cuda_device not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    tile_cache_dir = _configure_experiment_tile_cache(
        output_dir=out_dir,
        experiment_root=experiment_root,
        extractor_name=args.extractor,
        tile_filter=args.tile_filter,
        enable_tile_cache=args.use_tile_cache,
    )
    feature_cache_dir = _configure_experiment_feature_cache(
        output_dir=out_dir,
        experiment_root=experiment_root,
        extractor_name=args.extractor,
        tile_filter=args.tile_filter,
    )

    fresh_cache_info = (
        _configure_fresh_embedding_cache(run_id)
        if args.fresh_embedding_cache
        else None
    )
    reference_cache_dir = None
    if fresh_cache_info is None:
        reference_cache_dir = _configure_persistent_reference_cache(
            output_dir=out_dir,
            experiment_root=experiment_root,
            extractor_name=args.extractor,
        )
    run_artifacts = REPO_ROOT / "outputs" / run_id

    from wsi_core_pkg.runtime import run_wsi_agent_for_web

    t0 = time.time()
    try:
        # For vlm_diagnosis mode, skip slide validation and tile size resolution
        if args.agent != "vlm_diagnosis":
            _validate_slide_package(slide_path)
            tile_size_config = _resolve_tile_size_config(
                slide_path=slide_path,
                tile_size_px=args.tile_size_px,
                requested_tile_size_um=args.tile_size_um,
                requested_default_mpp_um=args.default_mpp_um,
            )
            args.tile_size_um = float(tile_size_config["tile_size_um"])
            args.tile_size_um_requested = tile_size_config["tile_size_um_requested"]
            args.tile_size_um_source = str(tile_size_config["tile_size_um_source"])
            args.default_mpp_um_requested = tile_size_config["default_mpp_um_requested"]
            args.slide_mpp_um = tile_size_config["slide_mpp_um"]
            args.resolved_mpp_um = float(tile_size_config["resolved_mpp_um"])
            args.mpp_source = str(tile_size_config["mpp_source"])
            args.default_mpp_um_fallback = float(tile_size_config["default_mpp_um_fallback"])

            print(f"[SLIDE] {slide_path}")
        else:
            # For vlm_diagnosis, set tile size config to defaults
            args.tile_size_um = args.tile_size_um or (args.tile_size_px * DEFAULT_MPP_UM_FALLBACK)
            args.tile_size_um_requested = None
            args.tile_size_um_source = "not_applicable"
            args.default_mpp_um_requested = None
            args.slide_mpp_um = None
            args.resolved_mpp_um = DEFAULT_MPP_UM_FALLBACK
            args.mpp_source = "not_applicable"
            args.default_mpp_um_fallback = DEFAULT_MPP_UM_FALLBACK
            print("[MODE] vlm_diagnosis (no slide needed)")
            print(f"[ROI_INPUT] {args.roi_input_path}")

        print(f"[RUN]   {run_id}")
        print(f"[AGENT] {args.agent}")
        print(f"[MODEL] {args.model}  [EXTRACTOR] {args.extractor}  [FILTER] {args.tile_filter}")
        print(f"[ROI]   {args.roi_size_px}px")
        if args.agent != "vlm_diagnosis":
            print(
                "[TILE]  %spx / %.3fum  [source=%s, mpp=%.6f from %s]"
                % (
                    args.tile_size_px,
                    args.tile_size_um,
                    args.tile_size_um_source,
                    args.resolved_mpp_um,
                    args.mpp_source,
                )
            )
        print(f"[CACHEROOT] {cache_root}")
        if tile_cache_dir is not None:
            print(f"[TILECACHE] {tile_cache_dir}")
        else:
            print("[TILECACHE] disabled")
        if reference_cache_dir is not None:
            print(f"[REFCACHE] {reference_cache_dir}")
        print(f"[FEATCACHE] {feature_cache_dir}")
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            print(f"[CUDA]  CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")

        # For AML modes, pre-create patient_out and pass as case_output_dir
        patient_out = out_dir / patient_output_name
        case_output_dir = None
        if args.agent in {"aml_roi", "aml_auto", "aml_auto_stamp", "vlm_diagnosis", "aml_stamp"}:
            patient_out.mkdir(parents=True, exist_ok=True)
            case_output_dir = str(patient_out)

        result = run_wsi_agent_for_web(
            slide_path=slide_path,
            prompt=None,
            agent_type=args.agent,
            run_id=run_id,
            model_name=args.model,
            extractor_name=args.extractor,
            tile_size_um=args.tile_size_um,
            tile_size_px=args.tile_size_px,
            batch_size=args.batch_size,
            tile_prefilter_method=args.tile_filter,
            roi_output_size_px=args.roi_size_px,
            max_accepted_rois=args.max_accepted_rois,
            target_accepted_rois=args.target_accepted_rois,
            default_mpp_um=args.default_mpp_um,
            roi_input_path=args.roi_input_path,
            roi_collection_path=args.roi_collection_path,
            case_output_dir=case_output_dir,
            stamp_enabled=args.agent == "aml_auto_stamp",
            stamp_train_root=args.stamp_train_root,
            stamp_repo_root=args.stamp_repo_root,
            stamp_extractors=args.stamp_extractors,
            stamp_tasks=args.stamp_tasks,
            stamp_device=args.stamp_device,
            stamp_accelerator=args.stamp_accelerator,
            stamp_num_workers=args.stamp_num_workers,
            stamp_enforce_no_leakage=not bool(args.allow_stamp_train_valid_overlap),
        )
        elapsed = time.time() - t0

        # Only create patient_out for vlm_diagnosis mode (aml_roi/auto already created it)
        if args.agent == "vlm_diagnosis":
            patient_out = out_dir / patient_output_name
            patient_out.mkdir(parents=True, exist_ok=True)

        final_output = result.get("final_output") or ""

        # Only check for final decision in modes that produce one (not aml_roi)
        final_decision = None
        if args.agent not in {"aml_roi", "aml_stamp"}:
            final_decision = shared_extract_final_decision(final_output)
            if not final_decision and args.agent not in {"aml_auto", "aml_auto_stamp", "vlm_diagnosis"}:
                raise RuntimeError("Completed run did not include a parseable final decision")
            # For aml_auto, diagnosis happens after ROI collection, so it should have final_decision
            if args.agent in {"aml_auto", "aml_auto_stamp"} and not final_decision:
                raise RuntimeError(f"{args.agent} mode: diagnosis phase did not produce a parseable final decision")

        persist_aml_slide_bundle(
            case_output_dir=patient_out,
            result=result,
            run_id=run_id,
            agent_type=args.agent,
            model_name=args.model,
            extractor_name=args.extractor,
            tile_filter=args.tile_filter,
            tile_size_px=args.tile_size_px,
            tile_size_um=args.tile_size_um,
            tile_size_um_requested=getattr(args, "tile_size_um_requested", args.tile_size_um),
            tile_size_um_source=getattr(args, "tile_size_um_source", "explicit"),
            default_mpp_um_requested=getattr(args, "default_mpp_um_requested", None),
            slide_mpp_um=getattr(args, "slide_mpp_um", None),
            resolved_mpp_um=getattr(args, "resolved_mpp_um", None),
            default_mpp_um_fallback=getattr(args, "default_mpp_um_fallback", DEFAULT_MPP_UM_FALLBACK),
            mpp_source=getattr(args, "mpp_source", None),
            roi_size_px=args.roi_size_px,
            batch_size=args.batch_size,
            elapsed_sec=elapsed,
            slide_path=slide_path,
            slide_name=result.get("slide_name"),
            patient_name=patient_name,
            roi_input_path=args.roi_input_path,
            status="ok",
            fresh_embedding_cache=bool(args.fresh_embedding_cache),
            tile_cache_enabled=bool(args.use_tile_cache),
            tile_cache_dir=str(tile_cache_dir) if tile_cache_dir else "",
            feature_cache_dir=str(feature_cache_dir),
        )

        print(f"[OK]    {patient_name}  ({elapsed:.0f}s)  -> {patient_out}")
        return 0

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[FAIL]  {patient_name}  ({elapsed:.0f}s)  {exc}", file=sys.stderr)

        # Ensure patient_out exists for error reporting
        if 'patient_out' not in locals():
            patient_out = out_dir / patient_output_name
        patient_out.mkdir(parents=True, exist_ok=True)
        persist_aml_slide_bundle(
            case_output_dir=patient_out,
            result=None,
            run_id=run_id,
            agent_type=args.agent,
            model_name=args.model,
            extractor_name=args.extractor,
            tile_filter=args.tile_filter,
            tile_size_px=args.tile_size_px,
            tile_size_um=args.tile_size_um,
            tile_size_um_requested=getattr(args, "tile_size_um_requested", args.tile_size_um),
            tile_size_um_source=getattr(args, "tile_size_um_source", "explicit"),
            default_mpp_um_requested=getattr(args, "default_mpp_um_requested", None),
            slide_mpp_um=getattr(args, "slide_mpp_um", None),
            resolved_mpp_um=getattr(args, "resolved_mpp_um", None),
            default_mpp_um_fallback=getattr(args, "default_mpp_um_fallback", DEFAULT_MPP_UM_FALLBACK),
            mpp_source=getattr(args, "mpp_source", None),
            roi_size_px=args.roi_size_px,
            batch_size=args.batch_size,
            elapsed_sec=elapsed,
            slide_path=slide_path,
            slide_name=patient_name,
            patient_name=patient_name,
            roi_input_path=args.roi_input_path,
            status="error",
            error=str(exc),
            fresh_embedding_cache=bool(args.fresh_embedding_cache),
            tile_cache_enabled=bool(args.use_tile_cache),
            tile_cache_dir=str(tile_cache_dir) if tile_cache_dir else "",
            feature_cache_dir=str(feature_cache_dir),
        )
        return 1
    finally:
        if run_artifacts.exists() and run_artifacts.is_dir():
            shutil.rmtree(run_artifacts, ignore_errors=True)
        _cleanup_fresh_embedding_cache(fresh_cache_info)


if __name__ == "__main__":
    sys.exit(main())
