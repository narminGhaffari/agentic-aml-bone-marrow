import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import state
from .tuning_config import tuning_value

FINAL_DECISIONS = (
    "Normal Bone Marrow",
    "Acute Leukemia",
    "Normal marrow",
    "Acute leukemia",
    "Call for more diagnostics",
)
FINAL_DECISION_LOOKUP = {label.lower(): label for label in FINAL_DECISIONS}


def _aml_pipeline_config(key: str, default: Any) -> Any:
    try:
        return tuning_value("aml_pipeline", key)
    except Exception:
        return default


def resolve_aml_output_root(path_text: str | None) -> Path:
    raw = str(path_text or "outputs/").strip() or "outputs/"
    return Path(raw).expanduser().resolve()


def resolve_aml_case_output_dir(output_root: str | Path, case_name: str) -> Path:
    root = resolve_aml_output_root(str(output_root))
    safe_case_name = str(case_name or "slide").strip() or "slide"
    return root / safe_case_name


def _normalize_final_decision(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return FINAL_DECISION_LOOKUP.get(text.lower(), text)


def extract_final_decision(final_output: str) -> str:
    text = str(final_output or "").strip()
    if not text:
        return ""

    label_group = "|".join(re.escape(label) for label in FINAL_DECISIONS)

    patterns = (
        rf"final\s+(?:decision|diagnosis)\s*[:\-]\s*({label_group})",
        rf"(?:diagnosis|decision)\s*[:\-]\s*({label_group})",
        rf"(?:findings\s+)?(?:are\s+)?(?:most\s+)?(?:consistent\s+(?:with|:)\s+)({label_group})",
        rf"conclusion\s*[:\-]\s*({label_group})",
        rf"({label_group})\s+(?:is\s+)?(?:the\s+)?(?:final\s+)?(?:diagnosis|decision|conclusion)",
        rf"(?:I\s+(?:conclude|diagnose|decide|determine)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?(?:a\s+)?)(?:case\s+of\s+)?({label_group})",
        rf"(?:therefore|thus|hence),?\s+(?:this\s+)?(?:is\s+)?(?:a\s+)?(?:case\s+of\s+)?({label_group})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return _normalize_final_decision(match.group(1))

    seen_labels = [
        (text.lower().rfind(label.lower()), label)
        for label in FINAL_DECISIONS
        if text.lower().rfind(label.lower()) >= 0
    ]
    if seen_labels:
        _, label = max(seen_labels, key=lambda item: item[0])
        return label
    return ""


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _extract_vlm_prediction_row(
    *,
    final_output: str,
    patient: str,
    slide: str,
    run_id: str,
    agent_type: str,
    model_name: str,
    extractor_name: str,
    final_decision: str,
    selected_roi_count: Optional[int] = None,
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    try:
        candidate = str(final_output or "").strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.DOTALL)
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            parsed = loaded
    except Exception:
        parsed = {}

    npm1_prediction = parsed.get("npm1_prediction")
    npm1_applicable = ""
    npm1_classification = ""
    npm1_justification = ""
    if isinstance(npm1_prediction, dict):
        npm1_applicable = npm1_prediction.get("applicable", "")
        npm1_classification = npm1_prediction.get("classification", "")
        npm1_justification = npm1_prediction.get("justification", "")

    return {
        "Patient": patient,
        "slide": slide,
        "run_id": run_id,
        "agent_type": agent_type,
        "model_name": model_name,
        "extractor_name": extractor_name,
        "selected_roi_count": selected_roi_count if selected_roi_count is not None else "",
        "diagnosis": parsed.get("diagnosis", final_decision),
        "final_decision": final_decision,
        "thoughts": parsed.get("thoughts", ""),
        "npm1_prediction": npm1_prediction if isinstance(npm1_prediction, dict) else "",
        "npm1_applicable": npm1_applicable,
        "npm1_classification": npm1_classification,
        "npm1_justification": npm1_justification,
        "roi_assessment": parsed.get("roi_assessment", ""),
        "raw_final_output": final_output,
    }


def _write_single_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_roi_collection_json(collection: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(collection)
    clean_rois = []
    for roi in clean.get("accepted_rois", []):
        if not isinstance(roi, dict):
            continue
        clean_rois.append({k: v for k, v in roi.items() if not k.startswith("_")})
    clean["accepted_rois"] = clean_rois
    output_path.write_text(json.dumps(clean, indent=2, default=str))


def build_roi_collection_from_current_state(
    *,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an AML ROI collection manifest from the current in-memory WSI state."""
    images_subdir = str(_aml_pipeline_config("ROI_IMAGES_SUBDIR", "images"))
    roi_marks = list(getattr(state, "_roi_marks", []) or [])

    accepted_rois = []
    for idx, roi in enumerate(roi_marks, start=1):
        roi_id = str(roi.get("roi_id", idx))
        accepted_rois.append(
            {
                "roi_id": roi_id,
                "image_path": f"{images_subdir}/roi_{roi_id}.jpg",
                "absolute_image_path": None,
                "bbox_level0": roi.get("view_bbox_level0"),
                "field_width_um": roi.get("field_width_um"),
                "field_height_um": roi.get("field_height_um"),
                "tissue_fraction": roi.get("tissue_fraction"),
                "effective_magnification": roi.get("effective_magnification"),
                "candidate_rank": roi.get("candidate_rank"),
                "requested_bbox_norm": roi.get("requested_bbox_norm"),
                "accepted_reason": roi.get("label", ""),
                "_debug_path": roi.get("debug_path", ""),
            }
        )

    slide_path = str(getattr(state, "SLIDE_PATH", "") or "")
    run_id = str(getattr(state, "RUN_ID", "") or "")
    case_id = Path(slide_path).stem if slide_path else run_id
    slide_name = Path(slide_path).name if slide_path else ""
    collection: Dict[str, Any] = {
        "schema_version": 1,
        "task": "aml_roi_collection",
        "slide_path": slide_path,
        "slide_name": slide_name,
        "run_id": run_id,
        "case_id": case_id,
        "model_name": str(model_name if model_name is not None else getattr(state, "MODEL_NAME", "") or ""),
        "extractor_name": str(getattr(state, "EXTRACTOR_NAME", "") or ""),
        "tile_filter": str(getattr(state, "TILE_PREFILTER_METHOD", "") or ""),
        "tile_size_px": int(getattr(state, "TILE_SIZE_PX", 0) or 0),
        "roi_size_px": int(getattr(state, "ROI_OUTPUT_SIZE_PX", 0) or 0),
        "roi_images_dir": images_subdir,
        "accepted_roi_count": len(roi_marks),
        "target_accepted_rois": int(getattr(state, "TARGET_ACCEPTED_ROIS", 0) or 0),
        "slide_overview_image": None,
        "roi_candidates_image": None,
        "accepted_rois": accepted_rois,
        "discard_summary": [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return collection


def persist_current_aml_roi_collection(
    *,
    case_output_dir: str | Path | None = None,
    model_name: Optional[str] = None,
) -> Optional[Path]:
    """Checkpoint the currently accepted AML ROIs to disk immediately."""
    case_out_raw = case_output_dir or getattr(state, "CASE_OUTPUT_DIR", None)
    if not case_out_raw:
        return None

    case_out = Path(case_out_raw).expanduser().resolve()
    case_out.mkdir(parents=True, exist_ok=True)

    collection = build_roi_collection_from_current_state(model_name=model_name)

    candidates_src = str(getattr(state, "_last_roi_candidate_overlay_path", "") or "")
    if candidates_src and Path(candidates_src).is_file():
        collection["roi_candidates_image"] = str(Path(candidates_src).resolve())

    overview_src = str(getattr(state, "_last_overview_with_box_path", "") or "")
    if overview_src and Path(overview_src).is_file():
        collection["slide_overview_image"] = str(Path(overview_src).resolve())

    updated = materialize_aml_case_collection(collection, case_out)
    roi_collection_filename = str(_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json"))
    output_path = case_out / roi_collection_filename
    write_roi_collection_json(updated, output_path)
    return output_path


def _resolve_roi_source_path(roi: Dict[str, Any], collection_dir: Optional[Path]) -> Optional[Path]:
    for key in ("_debug_path", "absolute_image_path"):
        raw = roi.get(key)
        if raw:
            candidate = Path(str(raw)).expanduser()
            if candidate.is_file():
                return candidate.resolve()

    rel = roi.get("image_path")
    if rel and collection_dir is not None:
        candidate = (collection_dir / str(rel)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def _reconstruct_roi_image_from_slide(
    *,
    collection: Dict[str, Any],
    roi: Dict[str, Any],
    dst: Path,
) -> bool:
    slide_path = collection.get("slide_path")
    bbox = roi.get("bbox_level0") or roi.get("view_bbox_level0")
    if not slide_path or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    slide_candidate = Path(str(slide_path)).expanduser()
    if not slide_candidate.is_file():
        return False

    try:
        import openslide  # type: ignore
        from PIL import Image  # type: ignore

        x0, y0, width, height = [int(round(float(v))) for v in bbox]
        width = max(1, width)
        height = max(1, height)
        with openslide.open_slide(str(slide_candidate)) as slide:
            region = slide.read_region((max(0, x0), max(0, y0)), 0, (width, height))
        if region.mode == "RGBA":
            background = Image.new("RGBA", region.size, (255, 255, 255, 255))
            region = Image.alpha_composite(background, region)
        region = region.convert("RGB")
        output_px = int(collection.get("roi_size_px") or max(width, height))
        if output_px > 0 and region.size != (output_px, output_px):
            region = region.resize((output_px, output_px), Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        region.save(dst, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def _resolve_extra_image_path(
    collection: Dict[str, Any],
    collection_dir: Optional[Path],
    key: str,
) -> Optional[Path]:
    raw = collection.get(key)
    if not raw:
        return None
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute() and collection_dir is not None:
        candidate = collection_dir / candidate
    if candidate.is_file():
        return candidate.resolve()
    return None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def materialize_aml_case_collection(
    collection: Dict[str, Any],
    case_output_dir: Path,
    *,
    collection_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    images_subdir = str(_aml_pipeline_config("ROI_IMAGES_SUBDIR", "images"))
    images_dir = case_output_dir / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    roi_sources = []
    for roi in collection.get("accepted_rois", []):
        if not isinstance(roi, dict):
            continue
        roi_sources.append(_resolve_roi_source_path(roi, collection_dir))

    can_clear_images = all(
        src is None or not _path_is_within(src, images_dir)
        for src in roi_sources
    )
    if can_clear_images:
        for candidate in images_dir.iterdir():
            if candidate.is_file() and (
                candidate.name.lower().startswith("roi_")
                or candidate.name.lower().startswith("roi-candidates")
                or candidate.name.lower() == "roi_candidates.jpg"
                or candidate.name.lower().startswith("slide_overview")
            ):
                candidate.unlink()

    updated_rois = []
    for idx, roi in enumerate(collection.get("accepted_rois", []), start=1):
        if not isinstance(roi, dict):
            continue
        roi_copy = dict(roi)
        roi_id = str(roi_copy.get("roi_id", idx))
        src = _resolve_roi_source_path(roi_copy, collection_dir)
        suffix = (src.suffix if src is not None and src.suffix else ".jpg")
        dst = images_dir / f"roi_{roi_id}{suffix}"
        if src is None:
            reconstructed = images_dir / f"roi_{roi_id}.jpg"
            if _reconstruct_roi_image_from_slide(collection=collection, roi=roi_copy, dst=reconstructed):
                src = reconstructed
                dst = reconstructed
        if src is not None and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        roi_copy["image_path"] = f"{images_subdir}/{dst.name}"
        roi_copy["absolute_image_path"] = str(dst.resolve())
        updated_rois.append({k: v for k, v in roi_copy.items() if not k.startswith("_")})

    updated = dict(collection)
    updated["accepted_rois"] = updated_rois

    candidates_src = _resolve_extra_image_path(updated, collection_dir, "roi_candidates_image")
    candidates_filename = str(_aml_pipeline_config("ROI_CANDIDATES_FILENAME", "roi_candidates.jpg"))
    if candidates_src is not None:
        candidates_dst = images_dir / candidates_filename
        if candidates_src.resolve() != candidates_dst.resolve():
            shutil.copy2(candidates_src, candidates_dst)
        updated["roi_candidates_image"] = f"{images_subdir}/{candidates_dst.name}"

    overview_src = _resolve_extra_image_path(updated, collection_dir, "slide_overview_image")
    overview_filename = str(_aml_pipeline_config("SLIDE_OVERVIEW_FILENAME", "slide_overview.jpg"))
    if overview_src is not None:
        overview_dst = images_dir / overview_filename
        if overview_src.resolve() != overview_dst.resolve():
            shutil.copy2(overview_src, overview_dst)
        updated["slide_overview_image"] = f"{images_subdir}/{overview_dst.name}"

    roi_collection_filename = str(_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json"))
    write_roi_collection_json(updated, case_output_dir / roi_collection_filename)
    return updated


def materialize_aml_diagnosis_input(
    input_path: str,
    case_output_dir: Path,
) -> Dict[str, Any]:
    from .runtime import load_aml_diagnosis_input

    collection, collection_dir, slide_name, _ = load_aml_diagnosis_input(input_path)
    updated = materialize_aml_case_collection(collection, case_output_dir, collection_dir=collection_dir)
    return {
        "collection": updated,
        "slide_name": slide_name,
        "roi_collection_path": str(case_output_dir / str(_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json"))),
    }


def persist_aml_slide_bundle(
    *,
    case_output_dir: str | Path,
    result: Optional[Dict[str, Any]],
    run_id: str,
    agent_type: str,
    model_name: str,
    extractor_name: str,
    tile_filter: str,
    tile_size_px: int,
    tile_size_um: Optional[float],
    roi_size_px: int,
    batch_size: int,
    elapsed_sec: Optional[float],
    slide_path: Optional[str] = None,
    slide_name: Optional[str] = None,
    patient_name: Optional[str] = None,
    roi_input_path: Optional[str] = None,
    status: str = "ok",
    error: Optional[str] = None,
    tile_size_um_requested: Optional[float] = None,
    tile_size_um_source: Optional[str] = None,
    default_mpp_um_requested: Optional[float] = None,
    slide_mpp_um: Optional[float] = None,
    resolved_mpp_um: Optional[float] = None,
    default_mpp_um_fallback: Optional[float] = None,
    mpp_source: Optional[str] = None,
    fresh_embedding_cache: Optional[bool] = None,
    tile_cache_enabled: Optional[bool] = None,
    tile_cache_dir: Optional[str] = None,
    feature_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    case_out = Path(case_output_dir).expanduser().resolve()
    case_out.mkdir(parents=True, exist_ok=True)
    for filename in ("report.md", "report.txt", "report.json", "state.json", "summary.json", "final_output.txt"):
        candidate = case_out / filename
        if candidate.exists() and candidate.is_file():
            candidate.unlink()

    result = dict(result or {})
    state_data = result.get("state") or {}
    if not isinstance(state_data, dict):
        state_data = {}

    diagnosis_roi_count: Optional[int] = None
    if agent_type in {"aml_diagnosis", "vlm_diagnosis"} and roi_input_path:
        try:
            materialized = materialize_aml_diagnosis_input(roi_input_path, case_out)
            slide_name = slide_name or str(materialized.get("slide_name") or "")
            result["roi_collection_path"] = str(materialized.get("roi_collection_path") or result.get("roi_collection_path") or "")
            collection = materialized.get("collection") or {}
            if isinstance(collection, dict):
                diagnosis_roi_count = len([r for r in collection.get("accepted_rois", []) if isinstance(r, dict)])
        except Exception:
            if status == "ok":
                raise

    saved_roi_count: Optional[int] = None
    roi_collection_path = case_out / str(_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json"))
    if roi_collection_path.is_file():
        try:
            loaded_collection = json.loads(roi_collection_path.read_text())
            if isinstance(loaded_collection, dict):
                saved_roi_count = len([r for r in loaded_collection.get("accepted_rois", []) if isinstance(r, dict)])
        except Exception:
            saved_roi_count = None

    final_output = str(result.get("final_output") or "")
    final_decision = ""
    if agent_type not in {"aml_roi", "aml_stamp"} and status == "ok":
        final_decision = extract_final_decision(final_output)

    report_src = result.get("report_path")
    report_src_path = Path(report_src).expanduser() if isinstance(report_src, str) and report_src else None
    if report_src_path is not None and report_src_path.is_file():
        shutil.copy2(report_src_path, case_out / "report.md")
        txt_src = report_src_path.with_suffix(".txt")
        if txt_src.is_file():
            shutil.copy2(txt_src, case_out / "report.txt")

    if status == "ok" and final_output:
        (case_out / "final_output.txt").write_text(final_output)
        if agent_type in {"aml_diagnosis", "vlm_diagnosis"}:
            (case_out / "vlm_final_output.txt").write_text(final_output)

    _write_json(case_out / "state.json", state_data)

    effective_patient_name = patient_name or slide_name or case_out.name
    effective_slide = slide_path or slide_name or ""
    tile_size_um_for_report = tile_size_um if tile_size_um is not None else tile_size_um_requested
    vlm_prediction_row = None
    if agent_type not in {"aml_roi", "aml_stamp"} and final_output:
        vlm_prediction_row = _extract_vlm_prediction_row(
            final_output=final_output,
            patient=effective_patient_name,
            slide=effective_slide,
            run_id=run_id,
            agent_type=agent_type,
            model_name=model_name,
            extractor_name=extractor_name,
            final_decision=final_decision,
            selected_roi_count=diagnosis_roi_count,
        )
        _write_single_row_csv(case_out / "agent_vlm_output.csv", vlm_prediction_row)

    state_roi_count = len(state_data.get("roi_marks") or [])
    if diagnosis_roi_count is not None:
        report_roi_count = diagnosis_roi_count
    elif saved_roi_count is not None:
        report_roi_count = saved_roi_count
    else:
        report_roi_count = state_roi_count
    report_payload = {
        "patient": effective_patient_name,
        "slide": effective_slide,
        "run_id": run_id,
        "agent_type": agent_type,
        "model_name": model_name,
        "feature_extractor": {"key": extractor_name},
        "tile_filter": tile_filter,
        "tile_size": {
            "px": tile_size_px,
            "um": tile_size_um_for_report,
            "requested_um": tile_size_um_requested if tile_size_um_requested is not None else tile_size_um_for_report,
            "source": tile_size_um_source or "explicit",
            "mpp_um": resolved_mpp_um,
            "mpp_source": mpp_source,
        },
        "tile_size_px": tile_size_px,
        "tile_size_um": tile_size_um_for_report,
        "tile_size_um_requested": tile_size_um_requested if tile_size_um_requested is not None else tile_size_um_for_report,
        "tile_size_um_source": tile_size_um_source or "explicit",
        "default_mpp_um_requested": default_mpp_um_requested,
        "slide_mpp_um": slide_mpp_um,
        "resolved_mpp_um": resolved_mpp_um,
        "default_mpp_um_fallback": default_mpp_um_fallback,
        "mpp_source": mpp_source,
        "roi_size_px": roi_size_px,
        "batch_size": batch_size,
        "elapsed_sec": round(float(elapsed_sec), 1) if elapsed_sec is not None else None,
        "final_decision": final_decision,
        "final_output": final_output,
        "roi_count": report_roi_count,
        "selected_roi_count": report_roi_count,
        "step_count": len(state_data.get("step_log") or []),
        "source_report_path": str(report_src_path) if report_src_path and report_src_path.is_file() else "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if vlm_prediction_row is not None:
        report_payload["agent_vlm_output_csv"] = str(case_out / "agent_vlm_output.csv")
    if "stamp_result" in result:
        report_payload["stamp_result"] = result.get("stamp_result")
        report_payload["stamp_result_path"] = result.get("stamp_result_path", "")
    _write_json(case_out / "report.json", report_payload)
    if agent_type in {"aml_diagnosis", "vlm_diagnosis"}:
        _write_json(case_out / "vlm_report.json", report_payload)

    summary_payload = {
        "patient": effective_patient_name,
        "slide": effective_slide,
        "run_id": run_id,
        "agent": agent_type,
        "model": model_name,
        "extractor": extractor_name,
        "tile_filter": tile_filter,
        "tile_size_px": tile_size_px,
        "tile_size_um": tile_size_um_for_report,
        "tile_size_um_requested": tile_size_um_requested if tile_size_um_requested is not None else tile_size_um_for_report,
        "tile_size_um_source": tile_size_um_source or "explicit",
        "default_mpp_um_requested": default_mpp_um_requested,
        "slide_mpp_um": slide_mpp_um,
        "resolved_mpp_um": resolved_mpp_um,
        "default_mpp_um_fallback": default_mpp_um_fallback,
        "mpp_source": mpp_source,
        "roi_size_px": roi_size_px,
        "batch_size": batch_size,
        "elapsed_sec": round(float(elapsed_sec), 1) if elapsed_sec is not None else None,
        "status": status,
        "final_decision": final_decision,
        "selected_roi_count": report_roi_count,
    }
    if error:
        summary_payload["error"] = error
    if fresh_embedding_cache is not None:
        summary_payload["fresh_embedding_cache"] = bool(fresh_embedding_cache)
    if tile_cache_enabled is not None:
        summary_payload["tile_cache_enabled"] = bool(tile_cache_enabled)
    if tile_cache_dir is not None:
        summary_payload["tile_cache_dir"] = str(tile_cache_dir)
    if feature_cache_dir is not None:
        summary_payload["feature_cache_dir"] = str(feature_cache_dir)
    if "stamp_result" in result:
        stamp_result = result.get("stamp_result") or {}
        summary_payload["stamp_result_path"] = result.get("stamp_result_path", "")
        summary_payload["stamp_status"] = stamp_result.get("status", "")
    _write_json(case_out / "summary.json", summary_payload)
    if agent_type in {"aml_diagnosis", "vlm_diagnosis"}:
        _write_json(case_out / "vlm_summary.json", summary_payload)

    diagnosis_done = case_out / "diagnosis_done"
    if agent_type in {"aml_diagnosis", "vlm_diagnosis"} and status == "ok":
        diagnosis_done.touch()
    elif diagnosis_done.exists():
        diagnosis_done.unlink()

    return {
        "case_output_dir": str(case_out),
        "final_decision": final_decision,
        "summary_path": str(case_out / "summary.json"),
    }
