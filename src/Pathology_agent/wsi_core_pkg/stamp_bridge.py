import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import openslide
from PIL import Image

from . import state
from .slide_utils import _resolve_mpp


STAMP_TILE_SIZE_UM = 256.0
STAMP_TILE_SIZE_PX = 224
STAMP_TILES_PER_ROI = 4
DEFAULT_STAMP_TRAIN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "Experimnes"
    / "STAMP_Train"
    / "All_Tiles"
)
DEFAULT_STAMP_REPO_ROOT = Path(__file__).resolve().parents[2] / "STAMP"


@dataclass(frozen=True)
class StampRunConfig:
    enabled: bool = False
    train_root: Path = DEFAULT_STAMP_TRAIN_ROOT
    repo_root: Path = DEFAULT_STAMP_REPO_ROOT
    extractors: Optional[Sequence[str]] = None
    tasks: Sequence[str] = ("WHO_CLASSE_SIMPLE", "NPM1")
    tile_size_um: float = STAMP_TILE_SIZE_UM
    tile_size_px: int = STAMP_TILE_SIZE_PX
    tiles_per_roi: int = STAMP_TILES_PER_ROI
    default_mpp_um: Optional[float] = 0.159
    device: str = "auto"
    accelerator: str = "auto"
    num_workers: int = 0
    enforce_no_leakage: bool = True


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe.strip("._-") or "case"


def _stamp_src_path(repo_root: Path) -> Path:
    return repo_root / "src"


def _ensure_stamp_importable(repo_root: Path) -> None:
    src = _stamp_src_path(repo_root).resolve()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _canonical_extractor_name(raw: str) -> str:
    aliases = {
        "uni": "uni",
        "uni2": "uni2",
        "uni2-49b04e14": "uni2",
        "virchow2": "virchow2",
        "virchow2-49b04e14": "virchow2",
        "dinobloom": "dino-bloom",
        "dinobloom_s": "dino-bloom",
        "dinobloom-small": "dino-bloom",
        "dinobloom_s-49b04e14": "dino-bloom",
        "dino_bloom": "dino-bloom",
        "dino-bloom": "dino-bloom",
        "dino-bloom-49b04e14": "dino-bloom",
        "h_optimus_1": "h-optimus-1",
        "h-optimus-1": "h-optimus-1",
        "h-optimus-1-49b04e14": "h-optimus-1",
    }
    key = str(raw or "").strip().lower()
    return aliases.get(key, key)


def _model_dir_for_extractor(task_dir: Path, extractor: str) -> Optional[Path]:
    canonical = _canonical_extractor_name(extractor)
    exact = task_dir / canonical
    if (exact / "model.ckpt").is_file():
        return exact
    candidates = sorted(
        p for p in task_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{canonical}-") and (p / "model.ckpt").is_file()
    )
    return candidates[0] if candidates else None


def _discover_extractors(train_root: Path, tasks: Sequence[str]) -> List[str]:
    names: set[str] = set()
    for task in tasks:
        task_dir = train_root / task
        if not task_dir.is_dir():
            continue
        for child in task_dir.iterdir():
            if child.is_dir() and (child / "model.ckpt").is_file():
                name = child.name
                if name.endswith("-49b04e14"):
                    name = name[: -len("-49b04e14")]
                names.add(_canonical_extractor_name(name))
    return sorted(names)


def _normalise_extractors(
    requested: Optional[Sequence[str]],
    *,
    train_root: Path,
    tasks: Sequence[str],
    fallback: str,
) -> List[str]:
    if requested:
        values = [str(v).strip() for v in requested if str(v).strip()]
        if len(values) == 1 and values[0].lower() == "all":
            return _discover_extractors(train_root, tasks)
        return [_canonical_extractor_name(v) for v in values]
    return [_canonical_extractor_name(fallback)]


def _read_region_rgb(
    slide: openslide.AbstractSlide,
    *,
    x: int,
    y: int,
    size: int,
) -> Image.Image:
    region = slide.read_region((int(x), int(y)), 0, (int(size), int(size)))
    if region.mode == "RGBA":
        bg = Image.new("RGBA", region.size, (255, 255, 255, 255))
        region = Image.alpha_composite(bg, region)
    return region.convert("RGB")


def _roi_center(roi: Dict[str, Any]) -> Optional[tuple[int, int]]:
    bbox = roi.get("bbox_level0") or roi.get("view_bbox_level0")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x0, y0, w, h = [int(round(float(v))) for v in bbox]
    return x0 + w // 2, y0 + h // 2


def extract_stamp_tiles_from_roi_collection(
    *,
    slide_path: str,
    roi_collection: Dict[str, Any],
    output_dir: Path,
    tile_size_um: float = STAMP_TILE_SIZE_UM,
    tile_size_px: int = STAMP_TILE_SIZE_PX,
    tiles_per_roi: int = STAMP_TILES_PER_ROI,
    default_mpp_um: Optional[float] = 0.159,
) -> Dict[str, Any]:
    """Extract a fixed 2x2 STAMP tile grid around each agent-selected ROI center."""
    if int(tiles_per_roi) != 4:
        raise ValueError("Only 4 tiles per ROI are currently supported.")

    tile_dir = output_dir / "stamp_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.open_slide(slide_path)
    previous_override = getattr(state, "DEFAULT_MPP_UM_OVERRIDE", None)
    if default_mpp_um is not None:
        state.DEFAULT_MPP_UM_OVERRIDE = float(default_mpp_um)
    try:
        mpp = float(_resolve_mpp(slide))
    finally:
        state.DEFAULT_MPP_UM_OVERRIDE = previous_override

    slide_w, slide_h = slide.level_dimensions[0]
    native_tile_px = max(1, int(round(float(tile_size_um) / mpp)))
    accepted = [r for r in roi_collection.get("accepted_rois", []) if isinstance(r, dict)]
    offsets = ((-1, -1), (0, -1), (-1, 0), (0, 0))
    tile_records: List[Dict[str, Any]] = []

    for roi_idx, roi in enumerate(accepted, start=1):
        center = _roi_center(roi)
        if center is None:
            continue
        cx, cy = center
        grid_x0 = cx - native_tile_px
        grid_y0 = cy - native_tile_px
        roi_id = str(roi.get("roi_id", roi_idx))
        for tile_idx, (ox, oy) in enumerate(offsets, start=1):
            x0 = grid_x0 + (ox + 1) * native_tile_px
            y0 = grid_y0 + (oy + 1) * native_tile_px
            x0 = max(0, min(int(x0), max(0, slide_w - native_tile_px)))
            y0 = max(0, min(int(y0), max(0, slide_h - native_tile_px)))
            tile = _read_region_rgb(slide, x=x0, y=y0, size=native_tile_px)
            tile = tile.resize((int(tile_size_px), int(tile_size_px)), Image.BILINEAR)
            name = f"roi_{_sanitize_filename(roi_id)}_tile_{tile_idx}.jpg"
            path = tile_dir / name
            tile.save(path, format="JPEG", quality=95)
            tile_records.append(
                {
                    "roi_id": roi_id,
                    "tile_index_within_roi": tile_idx,
                    "image_path": str(path.relative_to(output_dir)),
                    "absolute_image_path": str(path.resolve()),
                    "x0_level0": x0,
                    "y0_level0": y0,
                    "native_tile_px": native_tile_px,
                    "tile_size_um": float(tile_size_um),
                    "tile_size_px": int(tile_size_px),
                    "mpp_um": mpp,
                    "coord_um": [float(x0 * mpp), float(y0 * mpp)],
                }
            )

    manifest = {
        "schema_version": 1,
        "task": "agent_selected_stamp_tiles",
        "slide_path": slide_path,
        "case_id": roi_collection.get("case_id") or Path(slide_path).stem,
        "roi_collection_path": roi_collection.get("roi_collection_path"),
        "roi_count": len(accepted),
        "tiles_per_roi": int(tiles_per_roi),
        "tile_count": len(tile_records),
        "tile_size_um": float(tile_size_um),
        "tile_size_px": int(tile_size_px),
        "resolved_mpp_um": mpp,
        "native_tile_px": native_tile_px,
        "tiles": tile_records,
        "leakage_note": "Tiles are derived only from agent-selected ROI coordinates and WSI pixels.",
    }
    (output_dir / "stamp_tiles_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _make_extractor(name: str):
    from stamp.preprocessing.config import ExtractorName

    canonical = _canonical_extractor_name(name)
    enum_value = ExtractorName(canonical)
    match enum_value:
        case ExtractorName.UNI:
            from stamp.preprocessing.extractor.uni import uni

            return uni()
        case ExtractorName.UNI2:
            from stamp.preprocessing.extractor.uni2 import uni2

            return uni2()
        case ExtractorName.DINO_BLOOM:
            from stamp.preprocessing.extractor.dinobloom import dino_bloom

            return dino_bloom()
        case ExtractorName.VIRCHOW2:
            from stamp.preprocessing.extractor.virchow2 import virchow2

            return virchow2()
        case ExtractorName.H_OPTIMUS_1:
            from stamp.preprocessing.extractor.h_optimus_1 import h_optimus_1

            return h_optimus_1()
        case _:
            raise ValueError(f"Unsupported STAMP extractor for this bridge: {name}")


def encode_stamp_tiles(
    *,
    tile_manifest: Dict[str, Any],
    output_dir: Path,
    extractor_name: str,
    repo_root: Path,
    device: str,
) -> Path:
    _ensure_stamp_importable(repo_root)
    import h5py
    import stamp
    import torch

    extractor = _make_extractor(extractor_name)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    model = extractor.model.to(resolved_device).eval()

    tile_records = [r for r in tile_manifest.get("tiles", []) if isinstance(r, dict)]
    if not tile_records:
        raise RuntimeError("No STAMP tiles were extracted from the selected ROIs.")

    feats = []
    coords = []
    for record in tile_records:
        image = Image.open(record["absolute_image_path"]).convert("RGB")
        tensor = extractor.transform(image).unsqueeze(0).to(resolved_device)
        with torch.inference_mode():
            feats.append(model(tensor).detach().cpu())
        coords.append(record["coord_um"])

    feature_dir = output_dir / "stamp_features" / _canonical_extractor_name(extractor_name)
    feature_dir.mkdir(parents=True, exist_ok=True)
    case_id = _sanitize_filename(str(tile_manifest.get("case_id") or "case"))
    h5_path = feature_dir / f"{case_id}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["coords"] = np.asarray(coords, dtype=np.float32)
        h5["feats"] = torch.concat(feats, dim=0).half().numpy()
        h5.attrs["stamp_version"] = stamp.__version__
        h5.attrs["extractor"] = str(extractor.identifier)
        h5.attrs["unit"] = "um"
        h5.attrs["tile_size_um"] = float(tile_manifest["tile_size_um"])
        h5.attrs["tile_size_px"] = int(tile_manifest["tile_size_px"])
        h5.attrs["feat_type"] = "tile"
        h5.attrs["source"] = "agent_selected_roi_tiles"
    return h5_path


def _write_slide_table(feature_h5: Path, output_dir: Path, case_id: str) -> Path:
    slide_table = output_dir / f"slide_table_{feature_h5.parent.name}.csv"
    with slide_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PATIENT", "FILENAME"])
        writer.writeheader()
        writer.writerow({"PATIENT": case_id, "FILENAME": feature_h5.name})
    return slide_table


def _read_patient_preds(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_patient_overlap(checkpoint_path: Path, case_id: str, repo_root: Path) -> List[str]:
    _ensure_stamp_importable(repo_root)
    from stamp.modeling.deploy import load_model_from_ckpt

    model = load_model_from_ckpt(checkpoint_path).eval()
    train_valid = set(str(p) for p in getattr(model, "train_patients", []) or [])
    train_valid |= set(str(p) for p in getattr(model, "valid_patients", []) or [])
    candidates = {str(case_id), Path(str(case_id)).stem}
    return sorted(train_valid & candidates)


def deploy_stamp_models_from_agent_rois(
    *,
    slide_path: str,
    roi_collection: Dict[str, Any],
    output_dir: Path,
    agent_extractor_name: str,
    config: StampRunConfig,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest = extract_stamp_tiles_from_roi_collection(
        slide_path=slide_path,
        roi_collection=roi_collection,
        output_dir=output_dir,
        tile_size_um=config.tile_size_um,
        tile_size_px=config.tile_size_px,
        tiles_per_roi=config.tiles_per_roi,
        default_mpp_um=config.default_mpp_um,
    )

    _ensure_stamp_importable(config.repo_root)
    from stamp.modeling.deploy import deploy_categorical_model_

    case_id = _sanitize_filename(str(tile_manifest.get("case_id") or Path(slide_path).stem))
    extractors = _normalise_extractors(
        config.extractors,
        train_root=config.train_root,
        tasks=config.tasks,
        fallback=agent_extractor_name,
    )
    deployments: List[Dict[str, Any]] = []

    for extractor in extractors:
        try:
            feature_h5 = encode_stamp_tiles(
                tile_manifest=tile_manifest,
                output_dir=output_dir,
                extractor_name=extractor,
                repo_root=config.repo_root,
                device=config.device,
            )
        except Exception as exc:
            deployments.append(
                {
                    "extractor": extractor,
                    "status": "failed_feature_extraction",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        slide_table = _write_slide_table(feature_h5, output_dir, case_id)
        for task in config.tasks:
            task_dir = config.train_root / task
            model_dir = _model_dir_for_extractor(task_dir, extractor) if task_dir.is_dir() else None
            if model_dir is None:
                deployments.append(
                    {
                        "task": task,
                        "extractor": extractor,
                        "status": "missing_checkpoint",
                        "checkpoint_path": "",
                    }
                )
                continue

            ckpt = model_dir / "model.ckpt"
            try:
                overlap = _checkpoint_patient_overlap(ckpt, case_id, config.repo_root)
                if overlap and config.enforce_no_leakage:
                    raise RuntimeError(
                        "DATA LEAKAGE DETECTED: deployment patient overlaps with "
                        f"checkpoint train/validation patients: {overlap}"
                    )
                deploy_out = output_dir / "stamp_predictions" / task / extractor
                deploy_categorical_model_(
                    output_dir=deploy_out,
                    checkpoint_paths=[ckpt],
                    clini_table=None,
                    slide_table=slide_table,
                    feature_dir=feature_h5.parent,
                    ground_truth_label=None,
                    time_label=None,
                    status_label=None,
                    patient_label="PATIENT",
                    filename_label="FILENAME",
                    num_workers=int(config.num_workers),
                    accelerator=config.accelerator,
                )
                pred_csv = deploy_out / "patient-preds.csv"
                ci_csv = deploy_out / "patient-preds_95_confidence_interval.csv"
                deployments.append(
                    {
                        "task": task,
                        "extractor": extractor,
                        "status": "ok",
                        "checkpoint_path": str(ckpt.resolve()),
                        "feature_h5": str(feature_h5.resolve()),
                        "slide_table": str(slide_table.resolve()),
                        "prediction_csv": str(pred_csv.resolve()),
                        "confidence_csv": str(ci_csv.resolve()) if ci_csv.is_file() else "",
                        "patient_predictions": _read_patient_preds(pred_csv),
                        "leakage_overlap": overlap,
                    }
                )
            except Exception as exc:
                deployments.append(
                    {
                        "task": task,
                        "extractor": extractor,
                        "status": "failed_deploy",
                        "checkpoint_path": str(ckpt.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    result = {
        "schema_version": 1,
        "status": "ok" if any(d.get("status") == "ok" for d in deployments) else "failed",
        "tile_manifest_path": str((output_dir / "stamp_tiles_manifest.json").resolve()),
        "tile_manifest": tile_manifest,
        "deployments": deployments,
        "leakage_policy": {
            "uses_clinical_labels": False,
            "enforce_no_train_valid_overlap": bool(config.enforce_no_leakage),
            "roi_selection_source": "agent WSI navigation only",
        },
    }
    (output_dir / "stamp_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
