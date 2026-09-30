#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SINGLE_SLIDE="${SCRIPT_DIR}/run_single_slide.py"
RUN_DATE_TIME="$(date +%Y%m%d_%H%M%S)"

CSV="/data/private/metadata/AML-HEALTHY-SLIDE-DX_test.csv"
SLIDES_ROOT="/data/private/wsi"
OUTPUT_DIR="/results/Agent_STAMP"
EXPERIMENT_ROOT="/results/Agent_STAMP"
MODEL="gemma-4-31B-it"
EXTRACTOR="uni2"
STAMP_EXTRACTOR="uni2"
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
BATCH_SIZE="512"
ROI_SIZE_PX="2048"
MAX_ACCEPTED_ROIS="5"
TARGET_ACCEPTED_ROIS="5"
DEFAULT_MPP_UM="0.159"
CUDA_DEVICE=""
STAMP_DEVICE="auto"
STAMP_ACCELERATOR="auto"
STAMP_NUM_WORKERS="0"
RESUME=true
PARALLEL_PREDICTIONS=false

PYTHON_BIN="python"
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

usage() {
    cat <<EOF
Usage: bash run_batch_aml_staged.sh [options]

Core:
  --csv PATH
  --slides-root PATH
  --output-dir PATH
  --experiment-root PATH
  --model NAME
  --extractor NAME
  --stamp-extractor NAME

Behavior:
  --parallel-predictions     Run VLM and STAMP concurrently after ROI selection.
  --sequential-predictions   Run VLM then STAMP after ROI selection (default).
  --no-resume                Rerun stages even when outputs exist.
  --cuda-device VALUE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        --slides-root) SLIDES_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --extractor) EXTRACTOR="$2"; shift 2 ;;
        --stamp-extractor) STAMP_EXTRACTOR="$2"; shift 2 ;;
        --tile-filter) TILE_FILTER="$2"; shift 2 ;;
        --tile-size-px) TILE_SIZE_PX="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --roi-size-px) ROI_SIZE_PX="$2"; shift 2 ;;
        --max-accepted-rois) MAX_ACCEPTED_ROIS="$2"; shift 2 ;;
        --target-accepted-rois) TARGET_ACCEPTED_ROIS="$2"; shift 2 ;;
        --default-mpp-um) DEFAULT_MPP_UM="$2"; shift 2 ;;
        --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
        --stamp-device) STAMP_DEVICE="$2"; shift 2 ;;
        --stamp-accelerator) STAMP_ACCELERATOR="$2"; shift 2 ;;
        --stamp-num-workers) STAMP_NUM_WORKERS="$2"; shift 2 ;;
        --parallel-predictions) PARALLEL_PREDICTIONS=true; shift ;;
        --sequential-predictions) PARALLEL_PREDICTIONS=false; shift ;;
        --no-resume) RESUME=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

extractor_tag() {
    case "$1" in
        uni2) echo "UNI2" ;;
        h_optimus_1) echo "H-optimus-1" ;;
        virchow2) echo "Virchow2" ;;
        dinobloom) echo "DinoBloom-S" ;;
        dinobloom_base) echo "DinoBloom-B" ;;
        dinobloom_large) echo "DinoBloom-L" ;;
        dinobloom_giant) echo "DinoBloom-G" ;;
        reddino) echo "RedDino-Small" ;;
        reddino_base) echo "RedDino-base" ;;
        reddino_large) echo "RedDino-large" ;;
        *) echo "$1" ;;
    esac
}

generated_output_leaf() {
    printf '%s_%s_%s_%spx\n' "$RUN_DATE_TIME" "$MODEL" "$(extractor_tag "$EXTRACTOR")" "$TILE_SIZE_PX"
}

is_output_parent_dir() {
    local base
    base="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "$base" =~ ^agent(_roi|_stamp)?$ ]]
}

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
fi
if is_output_parent_dir "$OUTPUT_DIR"; then
    OUTPUT_DIR="${OUTPUT_DIR}/$(generated_output_leaf)"
fi
if [[ -n "$EXPERIMENT_ROOT" && "$EXPERIMENT_ROOT" != /* ]]; then
    EXPERIMENT_ROOT="$(realpath -m "$EXPERIMENT_ROOT")"
fi
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/_logs_staged"
mkdir -p "$LOG_DIR"
VLM_BATCH_CSV="${OUTPUT_DIR}/agent_vlm_outputs.csv"

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

mapfile -t PATIENTS < <("$PYTHON_BIN" - "$CSV" <<'PY'
import csv
import sys
from pathlib import Path

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    fields = [str(f or "").strip() for f in (reader.fieldnames or [])]
    filename_key = next((f for f in fields if f.lower() in {"filename", "slide", "slide_id", "slide_name"}), None)
    patient_key = next((f for f in fields if f.lower() in {"patient", "patient_id", "case_id"}), None)
    for row in reader:
        raw = str(row.get(filename_key) if filename_key else row.get(patient_key, "")).strip()
        if raw:
            print(Path(raw).stem if raw.lower().endswith((".h5", ".csv")) else raw)
PY
)

append_agent_vlm_csv() {
    local src="$1"
    local dst="$2"
    [[ -f "$src" ]] || return 0
    if [[ ! -f "$dst" ]]; then
        cp "$src" "$dst"
    elif ! grep -q "^$(basename "$(dirname "$src")")," "$dst" 2>/dev/null; then
        tail -n +2 "$src" >> "$dst"
    fi
}

rebuild_agent_vlm_csv() {
    local root="$1"
    local dst="$2"
    "$PYTHON_BIN" - "$root" "$dst" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = []
fieldnames = None
for src in sorted(root.glob("*/agent_vlm_output.csv")):
    try:
        with src.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            current = list(reader)
            if not current:
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or current[0].keys())
            rows.extend(current)
    except Exception:
        continue
if not rows or fieldnames is None:
    raise SystemExit(0)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
PY
}

roi_count() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0)
    raise SystemExit(0)
try:
    data = json.loads(p.read_text())
except Exception:
    print(0)
    raise SystemExit(0)
rois = [r for r in data.get("accepted_rois", []) if isinstance(r, dict)]
print(len(rois))
PY
}

roi_complete() {
    local path="$1"
    "$PYTHON_BIN" - "$path" "$TARGET_ACCEPTED_ROIS" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
target = int(sys.argv[2])
if not p.is_file():
    raise SystemExit(1)
try:
    data = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
rois = [r for r in data.get("accepted_rois", []) if isinstance(r, dict)]
raise SystemExit(0 if len(rois) >= target else 1)
PY
}

roi_has_any() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
try:
    data = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
rois = [r for r in data.get("accepted_rois", []) if isinstance(r, dict)]
raise SystemExit(0 if len(rois) >= 1 else 1)
PY
}

stamp_ok() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
try:
    data = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("status") == "ok" else 1)
PY
}

vlm_ok() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import csv
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
try:
    with p.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
except Exception:
    raise SystemExit(1)
if not rows:
    raise SystemExit(1)
row = rows[0]
raw = str(row.get("raw_final_output", "") or "").lower()
decision = str(row.get("final_decision", "") or "").strip()
try:
    roi_count = int(float(str(row.get("selected_roi_count", "") or "0")))
except Exception:
    roi_count = 0
bad_markers = (
    "image not available",
    "images are unavailable",
    "roi images are unavailable",
    "no visual data",
)
if roi_count < 1 or not decision or any(marker in raw for marker in bad_markers):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

echo "═══════════════════════════════════════════════════════════════"
echo " AML staged batch"
echo " CSV:          $CSV"
echo " Output:       $OUTPUT_DIR"
echo " Model:        $MODEL"
echo " ROI extractor:$EXTRACTOR"
echo " STAMP model:  $STAMP_EXTRACTOR"
echo " Parallel pred:$PARALLEL_PREDICTIONS"
echo " Python:       $PYTHON_BIN"
echo " Patients:     ${#PATIENTS[@]}"
echo "═══════════════════════════════════════════════════════════════"

for idx in "${!PATIENTS[@]}"; do
    PATIENT="${PATIENTS[$idx]}"
    SLIDE="${SLIDES_ROOT}/${PATIENT}.mrxs"
    PATIENT_OUT="${OUTPUT_DIR}/${PATIENT}"
    ROI_JSON="${PATIENT_OUT}/roi_collection.json"
    VLM_CSV="${PATIENT_OUT}/agent_vlm_output.csv"
    STAMP_JSON="${PATIENT_OUT}/stamp/stamp_result.json"
    n=$((idx + 1))

    if [[ ! -f "$SLIDE" ]]; then
        echo "[$n/${#PATIENTS[@]}] SKIP $PATIENT - slide not found"
        continue
    fi

    echo ""
    echo "[$n/${#PATIENTS[@]}] $PATIENT"

    if $RESUME && roi_has_any "$ROI_JSON"; then
        roi_n="$(roi_count "$ROI_JSON")"
        if roi_complete "$ROI_JSON"; then
            echo "  ROI   skip (${roi_n}/${TARGET_ACCEPTED_ROIS})"
        else
            echo "  ROI   skip partial (${roi_n}/${TARGET_ACCEPTED_ROIS}); predictions will still run"
        fi
    else
        echo "  ROI   run"
        if ! "$PYTHON_BIN" "$RUN_SINGLE_SLIDE" \
            --agent aml_roi \
            --slide "$SLIDE" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-root "$EXPERIMENT_ROOT" \
            --model "$MODEL" \
            --extractor "$EXTRACTOR" \
            --tile-filter "$TILE_FILTER" \
            --tile-size-px "$TILE_SIZE_PX" \
            --batch-size "$BATCH_SIZE" \
            --roi-size-px "$ROI_SIZE_PX" \
            --max-accepted-rois "$MAX_ACCEPTED_ROIS" \
            --target-accepted-rois "$TARGET_ACCEPTED_ROIS" \
            --default-mpp-um "$DEFAULT_MPP_UM" \
            >"${LOG_DIR}/${PATIENT}.roi.log" 2>&1; then
            if roi_has_any "$ROI_JSON"; then
                roi_n="$(roi_count "$ROI_JSON")"
                echo "  ROI   partial after ROI-stage error (${roi_n}/${TARGET_ACCEPTED_ROIS}); predictions will still run"
            else
                echo "  ROI   failed"
            fi
        fi
    fi

    if ! roi_has_any "$ROI_JSON"; then
        echo "  ROI   failed with 0 accepted ROIs; skipping predictions"
        continue
    fi

    run_vlm() {
        if $RESUME && vlm_ok "$VLM_CSV"; then
            echo "  VLM   skip"
            append_agent_vlm_csv "$VLM_CSV" "$VLM_BATCH_CSV"
            return 0
        fi
        echo "  VLM   run"
        "$PYTHON_BIN" "$RUN_SINGLE_SLIDE" \
            --agent vlm_diagnosis \
            --roi-input-path "$ROI_JSON" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-root "$EXPERIMENT_ROOT" \
            --model "$MODEL" \
            --extractor "$EXTRACTOR" \
            --tile-filter "$TILE_FILTER" \
            --tile-size-px "$TILE_SIZE_PX" \
            --batch-size "$BATCH_SIZE" \
            --roi-size-px "$ROI_SIZE_PX" \
            --default-mpp-um "$DEFAULT_MPP_UM" \
            >"${LOG_DIR}/${PATIENT}.vlm.log" 2>&1
        append_agent_vlm_csv "$VLM_CSV" "$VLM_BATCH_CSV"
    }

    run_stamp() {
        if $RESUME && stamp_ok "$STAMP_JSON"; then
            echo "  STAMP skip"
            return 0
        fi
        echo "  STAMP run"
        "$PYTHON_BIN" "$RUN_SINGLE_SLIDE" \
            --agent aml_stamp \
            --slide "$SLIDE" \
            --roi-input-path "$ROI_JSON" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-root "$EXPERIMENT_ROOT" \
            --model "$MODEL" \
            --extractor "$EXTRACTOR" \
            --stamp-extractors "$STAMP_EXTRACTOR" \
            --tile-filter "$TILE_FILTER" \
            --tile-size-px "$TILE_SIZE_PX" \
            --batch-size "$BATCH_SIZE" \
            --roi-size-px "$ROI_SIZE_PX" \
            --default-mpp-um "$DEFAULT_MPP_UM" \
            --stamp-device "$STAMP_DEVICE" \
            --stamp-accelerator "$STAMP_ACCELERATOR" \
            --stamp-num-workers "$STAMP_NUM_WORKERS" \
            >"${LOG_DIR}/${PATIENT}.stamp.log" 2>&1
    }

    if $PARALLEL_PREDICTIONS; then
        run_vlm &
        vlm_pid=$!
        run_stamp &
        stamp_pid=$!
        wait "$vlm_pid" || echo "  VLM   failed"
        wait "$stamp_pid" || echo "  STAMP failed"
    else
        run_vlm || echo "  VLM   failed"
        run_stamp || echo "  STAMP failed"
    fi
done

echo ""
rebuild_agent_vlm_csv "$OUTPUT_DIR" "$VLM_BATCH_CSV"
echo "Done. Batch VLM CSV: $VLM_BATCH_CSV"
