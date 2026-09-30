#!/usr/bin/env python3
"""Aggregate completed AML agent runs after inference.

This script is intentionally post-hoc: it reads finished VLM/STAMP prediction
files and merges ground truth metadata only after inference is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from zipfile import ZipFile


DEFAULT_SLIDE_METADATA = (
    "/data/private/metadata/AML-HEALTHY-SLIDE-DX_test.csv"
)
DEFAULT_CLINICAL_METADATA = (
    "/data/private/metadata/AML_HEALTHY-CLINI-DX_test.xlsx"
)
DEFAULT_GT_COLUMNS = ("WHO_CLASSE_SIMPLE", "NPM1", "AMLSTAT", "WHO_CLASS_COMPLEX", "WHO")


def _stem(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem


def _norm_key(value: object) -> str:
    return str(value or "").strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in str(cell_ref or "") if ch.isalpha())
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(0, idx - 1)


def _read_xlsx_first_sheet(path: Path) -> List[Dict[str, str]]:
    """Read a simple .xlsx first sheet without pandas/openpyxl."""
    if not path.is_file():
        return []

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(".//m:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//m:t", ns)))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: List[List[str]] = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            values: Dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                idx = _col_to_idx(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t", "")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                else:
                    raw = cell.find("m:v", ns)
                    value = raw.text if raw is not None and raw.text is not None else ""
                    if cell_type == "s" and value:
                        try:
                            value = shared_strings[int(value)]
                        except Exception:
                            pass
                values[idx] = value
            if values:
                rows.append([values.get(i, "") for i in range(max(values) + 1)])

    if not rows:
        return []
    headers = [str(h).strip() for h in rows[0]]
    out: List[Dict[str, str]] = []
    for row in rows[1:]:
        record = {headers[i]: str(row[i]).strip() if i < len(row) else "" for i in range(len(headers)) if headers[i]}
        if any(record.values()):
            out.append(record)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for key in preferred:
        if key and key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _task_extractor_from_stamp_csv(path: Path) -> tuple[str, str]:
    # .../<case>/stamp/stamp_predictions/<task>/<extractor>/patient-preds.csv
    parts = path.parts
    try:
        idx = parts.index("stamp_predictions")
        return parts[idx + 1], parts[idx + 2]
    except Exception:
        return "", ""


def _case_from_stamp_csv(path: Path, row: Dict[str, str]) -> str:
    patient = _norm_key(row.get("PATIENT"))
    if patient:
        return patient
    try:
        return path.parents[4].name
    except Exception:
        return ""


def _collect_vlm_rows(run_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(run_dir.glob("*/agent_vlm_output.csv")):
        for row in _read_csv(csv_path):
            case_id = _stem(row.get("Patient")) or csv_path.parent.name
            out = dict(row)
            out.setdefault("case_id", case_id)
            out["case_id"] = case_id
            out["case_dir"] = csv_path.parent.name
            rows.append(out)
    return rows


def _collect_stamp_rows(run_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(run_dir.glob("*/stamp/stamp_predictions/*/*/patient-preds.csv")):
        task, extractor = _task_extractor_from_stamp_csv(csv_path)
        for row in _read_csv(csv_path):
            case_id = _case_from_stamp_csv(csv_path, row)
            out = dict(row)
            out["case_id"] = case_id
            out["stamp_task"] = task
            out["stamp_extractor"] = extractor
            out["prediction_csv"] = str(csv_path)
            rows.append(out)
    return rows


def _build_metadata_maps(
    *,
    slide_metadata: Path,
    clinical_metadata: Path,
) -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    slide_by_case: Dict[str, Dict[str, str]] = {}
    for row in _read_csv(slide_metadata):
        filename = row.get("FILENAME") or row.get("filename") or row.get("slide") or row.get("SLIDE")
        case_id = _stem(filename)
        if case_id:
            slide_by_case[case_id] = dict(row)

    clinical_by_patient: Dict[str, Dict[str, str]] = {}
    suffix = clinical_metadata.suffix.lower()
    if suffix == ".xlsx":
        clinical_rows = _read_xlsx_first_sheet(clinical_metadata)
    else:
        clinical_rows = _read_csv(clinical_metadata)
    for row in clinical_rows:
        patient = _norm_key(row.get("PATIENT") or row.get("patient") or row.get("Patient"))
        if patient:
            clinical_by_patient[patient] = dict(row)
    return slide_by_case, clinical_by_patient


def _metadata_for_case(
    case_id: str,
    *,
    slide_by_case: Dict[str, Dict[str, str]],
    clinical_by_patient: Dict[str, Dict[str, str]],
) -> tuple[Dict[str, str], Dict[str, str]]:
    slide = slide_by_case.get(case_id, {})
    patient_id = _norm_key(slide.get("PATIENT") or slide.get("patient") or slide.get("Patient"))
    clinical = clinical_by_patient.get(patient_id, {})
    return slide, clinical


def _merge_ground_truth(
    rows: Iterable[Dict[str, str]],
    *,
    slide_by_case: Dict[str, Dict[str, str]],
    clinical_by_patient: Dict[str, Dict[str, str]],
    gt_columns: Sequence[str],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for row in rows:
        case_id = _stem(row.get("case_id") or row.get("Patient") or row.get("PATIENT"))
        slide, clinical = _metadata_for_case(
            case_id,
            slide_by_case=slide_by_case,
            clinical_by_patient=clinical_by_patient,
        )
        out = dict(row)
        out["case_id"] = case_id
        out["slide_metadata_PATIENT"] = slide.get("PATIENT", "")
        out["slide_metadata_FILENAME"] = slide.get("FILENAME", "")
        for col in gt_columns:
            out[f"gt_{col}"] = clinical.get(col, "")
        merged.append(out)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate finished AML Agent VLM/STAMP outputs and merge ground truth post-hoc.")
    parser.add_argument("--run-dir", required=True, help="Completed run folder, e.g. .../Agent_ROI/20260605_..._UNI2_224px")
    parser.add_argument("--slide-metadata", default=DEFAULT_SLIDE_METADATA)
    parser.add_argument("--clinical-metadata", default=DEFAULT_CLINICAL_METADATA)
    parser.add_argument("--output-dir", default=None, help="Defaults to --run-dir")
    parser.add_argument(
        "--ground-truth-columns",
        nargs="+",
        default=list(DEFAULT_GT_COLUMNS),
        help="Clinical metadata columns to append with gt_ prefix.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    slide_by_case, clinical_by_patient = _build_metadata_maps(
        slide_metadata=Path(args.slide_metadata).expanduser().resolve(),
        clinical_metadata=Path(args.clinical_metadata).expanduser().resolve(),
    )

    vlm_rows = _collect_vlm_rows(run_dir)
    stamp_rows = _collect_stamp_rows(run_dir)
    vlm_gt_rows = _merge_ground_truth(
        vlm_rows,
        slide_by_case=slide_by_case,
        clinical_by_patient=clinical_by_patient,
        gt_columns=args.ground_truth_columns,
    )
    stamp_gt_rows = _merge_ground_truth(
        stamp_rows,
        slide_by_case=slide_by_case,
        clinical_by_patient=clinical_by_patient,
        gt_columns=args.ground_truth_columns,
    )

    _write_csv(out_dir / "agent_vlm_outputs.csv", vlm_rows, preferred=("case_id", "Patient", "final_decision"))
    _write_csv(
        out_dir / "agent_vlm_outputs_with_ground_truth.csv",
        vlm_gt_rows,
        preferred=("case_id", "Patient", "final_decision", "gt_WHO_CLASSE_SIMPLE", "gt_NPM1"),
    )
    _write_csv(
        out_dir / "agent_stamp_outputs.csv",
        stamp_rows,
        preferred=("case_id", "stamp_task", "stamp_extractor", "PATIENT", "pred"),
    )
    _write_csv(
        out_dir / "agent_stamp_outputs_with_ground_truth.csv",
        stamp_gt_rows,
        preferred=("case_id", "stamp_task", "stamp_extractor", "PATIENT", "pred", "gt_WHO_CLASSE_SIMPLE", "gt_NPM1"),
    )

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "vlm_rows": len(vlm_rows),
        "stamp_rows": len(stamp_rows),
        "slide_metadata_cases": len(slide_by_case),
        "clinical_metadata_patients": len(clinical_by_patient),
        "ground_truth_columns": list(args.ground_truth_columns),
        "outputs": {
            "vlm": str(out_dir / "agent_vlm_outputs.csv"),
            "vlm_with_ground_truth": str(out_dir / "agent_vlm_outputs_with_ground_truth.csv"),
            "stamp": str(out_dir / "agent_stamp_outputs.csv"),
            "stamp_with_ground_truth": str(out_dir / "agent_stamp_outputs_with_ground_truth.csv"),
        },
    }
    (out_dir / "agent_aggregation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
