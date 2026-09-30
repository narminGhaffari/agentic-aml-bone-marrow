#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_batch_aml.sh — Run AML detector on every patient in a CSV list.
#
# Usage:
#   bash run_batch_aml.sh \
#       [--csv /path/to/patients.csv] \
#       [--slides-root /path/to/ALL_WSIs] \
#       [--output-dir ./batch_outputs] \
#       [--model GLM-4.6V-FP8] \
#       [--extractor uni2] \
#       [--tile-filter hybrid] \
#       [--tile-size-px 224] \
#       [--roi-size-px 2048] \
#       [--default-mpp-um <from config>] \
#       [--batch-size 512] \
#       [--experiment-root /path/to/experiment] \
#       [--use-tile-cache] \
#       [--resume]
#
# The --resume flag skips patients that already have a summary.json
# with status=ok and a non-empty final_decision in summary.json.
# If a slide run exits with an error during the current batch, the
# script reruns that slide once automatically.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SINGLE_SLIDE="${SCRIPT_DIR}/run_single_slide.py"
RUN_DATE_TIME="$(date +%Y%m%d_%H%M%S)"

# ── Defaults ─────────────────────────────────────────────────────────
CSV="/data/private/metadata/AML-HEALTHY-SLIDE-DX_test.csv"
SLIDES_ROOT="/data/private/wsi"
BASE_OUTPUT_ROOT="/results/agent_runs/new_runs"
OUTPUT_DIR=""
EXPERIMENT_ROOT=""
CUDA_DEVICE=""
MODEL="GLM-4.6V-FP8"
EXTRACTOR="uni2"
INCLUDE_MODEL_IN_OUTPUT_NAME=false
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
BATCH_SIZE="512"
ROI_SIZE_PX="2048"
MAX_ACCEPTED_ROIS="5"
TARGET_ACCEPTED_ROIS="5"
DEFAULT_MPP_UM="$(python3 - <<'PY'
from pathlib import Path
import yaml
cfg = Path('configs/config.yaml')
default = '0.159'
try:
    data = yaml.safe_load(cfg.read_text()) or {}
    value = data.get('tools', {}).get('slide', {}).get('DEFAULT_MPP_UM', default)
    print(value)
except Exception:
    print(default)
PY
)"
CONFIG_CACHE_ROOT_DIR="$(python3 - <<'PY'
from pathlib import Path
import yaml
cfg = Path('configs/config.yaml')
try:
    data = yaml.safe_load(cfg.read_text()) or {}
    value = str(data.get('tools', {}).get('cache', {}).get('CACHE_ROOT_DIR', '') or '').strip()
    print(value)
except Exception:
    print('')
PY
)"
AGENT="aml"
STAMP_TRAIN_ROOT="/data/private/stamp_train/All_Tiles"
STAMP_REPO_ROOT="/code/source/AML_Agent_Project/STAMP"
STAMP_EXTRACTORS=()
STAMP_TASKS=("WHO_CLASSE_SIMPLE" "NPM1")
STAMP_DEVICE="auto"
STAMP_ACCELERATOR="auto"
STAMP_NUM_WORKERS="0"
ALLOW_STAMP_TRAIN_VALID_OVERLAP=false
RESUME=false
USE_TILE_CACHE=false
SLIDE_TIMEOUT=0
PYTHON_BIN="python"

if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

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


# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)          CSV="$2";         shift 2 ;;
        --slides-root)  SLIDES_ROOT="$2"; shift 2 ;;
        --base-output-root) BASE_OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --cuda-device)  CUDA_DEVICE="$2"; shift 2 ;;
        --model)        MODEL="$2";       shift 2 ;;
        --extractor)    EXTRACTOR="$2";   shift 2 ;;
        --include-model-in-output-name) INCLUDE_MODEL_IN_OUTPUT_NAME=true; shift ;;
        --tile-filter)  TILE_FILTER="$2"; shift 2 ;;
        --tile-size-px) TILE_SIZE_PX="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --roi-size-px)  ROI_SIZE_PX="$2"; shift 2 ;;
        --max-accepted-rois) MAX_ACCEPTED_ROIS="$2"; shift 2 ;;
        --target-accepted-rois) TARGET_ACCEPTED_ROIS="$2"; shift 2 ;;
        --default-mpp-um) DEFAULT_MPP_UM="$2"; shift 2 ;;
        --agent)        AGENT="$2";       shift 2 ;;
        --stamp-train-root) STAMP_TRAIN_ROOT="$2"; shift 2 ;;
        --stamp-repo-root) STAMP_REPO_ROOT="$2"; shift 2 ;;
        --stamp-extractors)
            shift
            STAMP_EXTRACTORS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                STAMP_EXTRACTORS+=("$1")
                shift
            done
            ;;
        --stamp-tasks)
            shift
            STAMP_TASKS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                STAMP_TASKS+=("$1")
                shift
            done
            ;;
        --stamp-device) STAMP_DEVICE="$2"; shift 2 ;;
        --stamp-accelerator) STAMP_ACCELERATOR="$2"; shift 2 ;;
        --stamp-num-workers) STAMP_NUM_WORKERS="$2"; shift 2 ;;
        --allow-stamp-train-valid-overlap) ALLOW_STAMP_TRAIN_VALID_OVERLAP=true; shift ;;
        --slide-timeout) SLIDE_TIMEOUT="$2"; shift 2 ;;
        --use-tile-cache) USE_TILE_CACHE=true; shift ;;
        --resume)       RESUME=true;      shift   ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Force all output-related paths to absolute
if [[ -n "$BASE_OUTPUT_ROOT" && "$BASE_OUTPUT_ROOT" != /* ]]; then
    BASE_OUTPUT_ROOT="$(realpath -m "$BASE_OUTPUT_ROOT")"
fi
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
fi
if [[ -n "$EXPERIMENT_ROOT" && "$EXPERIMENT_ROOT" != /* ]]; then
    EXPERIMENT_ROOT="$(realpath -m "$EXPERIMENT_ROOT")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${BASE_OUTPUT_ROOT}/$(generated_output_leaf)"
    BASE_OUTPUT_ROOT="$(dirname "$OUTPUT_DIR")"
elif is_output_parent_dir "$OUTPUT_DIR"; then
    OUTPUT_DIR="${OUTPUT_DIR}/$(generated_output_leaf)"
fi

if [[ -z "$EXPERIMENT_ROOT" ]]; then
    EXPERIMENT_ROOT="$OUTPUT_DIR"
fi
if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

format_elapsed() {
    local total_seconds="${1:-0}"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf '%02dh:%02dm:%02ds' "$hours" "$minutes" "$seconds"
}

print_failure_reason() {
    local summary_path="$1"
    local log_path="$2"
    local reason=""

    if [[ -f "$summary_path" ]]; then
        reason=$("$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]

try:
    with open(summary_path) as handle:
        summary = json.load(handle)
except Exception as exc:
    print(f"Could not parse summary.json: {type(exc).__name__}: {exc}")
    raise SystemExit

error = str(summary.get("error") or "").strip()
status = str(summary.get("status") or "").strip()
if error:
    print(error)
elif status:
    print(f"Run ended with status={status!r} but no explicit error message was recorded")
PY
)
    fi

    if [[ -n "$reason" ]]; then
        echo "      reason: $reason"
        return
    fi

    if [[ -f "$log_path" ]]; then
        echo "      reason: summary.json had no error; showing tail of ${log_path}"
        tail -n 20 "$log_path" | sed 's/^/      | /'
        return
    fi

    echo "      reason: no summary.json error and no retry log found"
}

next_log_path() {
    local patient="$1"
    local candidate="${LOG_DIR}/${patient}.log"
    local retry_number=2

    if [[ ! -e "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
    fi

    while true; do
        candidate="${LOG_DIR}/${patient}.retry${retry_number}.log"
        if [[ ! -e "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
        retry_number=$((retry_number + 1))
    done
}

output_patient_name() {
    local patient="$1"
    if ! $INCLUDE_MODEL_IN_OUTPUT_NAME; then
        printf '%s\n' "$patient"
        return
    fi
    "$PYTHON_BIN" - "$MODEL" "$patient" <<'PY'
import re
import sys

def clean(value, default):
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "-", text)
    text = text.strip(".-_")
    return text or default

print(f"{clean(sys.argv[1], 'model')}_{clean(sys.argv[2], 'slide')}")
PY
}

# ── Read slide list. Prefer a FILENAME column and strip feature suffixes (.h5).
mapfile -t PATIENTS < <("$PYTHON_BIN" - "$CSV" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = sys.argv[1]
with open(csv_path, newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames:
        fieldnames = [str(f or "").strip() for f in reader.fieldnames]
        filename_key = next((f for f in fieldnames if f.lower() in {"filename", "slide", "slide_id", "slide_name"}), None)
        patient_key = next((f for f in fieldnames if f.lower() in {"patient", "patient_id", "case_id"}), None)
        for row in reader:
            raw = str(row.get(filename_key) if filename_key else row.get(patient_key, "")).strip()
            if not raw:
                continue
            stem = Path(raw).stem if raw.lower().endswith((".h5", ".csv")) else raw
            print(stem)
    else:
        handle.seek(0)
        for line in handle:
            raw = line.strip().split(",", 1)[0].strip().strip('"')
            if raw:
                print(Path(raw).stem if raw.lower().endswith(".h5") else raw)
PY
)
TOTAL=${#PATIENTS[@]}

echo "═══════════════════════════════════════════════════════════════"
echo " Slide-Agent batch AML run"
echo " CSV:         $CSV"
echo " Slides root: $SLIDES_ROOT"
echo " Base root:   $BASE_OUTPUT_ROOT"
echo " Output dir:  $OUTPUT_DIR"
echo " Agent:       $AGENT"
echo " Model:       $MODEL   Extractor: $EXTRACTOR   Filter: $TILE_FILTER"
echo " Output names include model: $INCLUDE_MODEL_IN_OUTPUT_NAME"
echo " Tile size:   ${TILE_SIZE_PX}px"
echo " Batch size:  $BATCH_SIZE"
echo " ROI size:    ${ROI_SIZE_PX}px"
echo " ROI target:  ${TARGET_ACCEPTED_ROIS} / max ${MAX_ACCEPTED_ROIS}"
echo " Default MPP: ${DEFAULT_MPP_UM}"
echo " Tile cache:  $USE_TILE_CACHE"
echo " Slide timeout: ${SLIDE_TIMEOUT}s (0=disabled)"
echo " CUDA devices:${CUDA_VISIBLE_DEVICES:+ }${CUDA_VISIBLE_DEVICES:-all}"
echo " Experiment root: $EXPERIMENT_ROOT"
echo " Config cache dir:${CONFIG_CACHE_ROOT_DIR:+ }${CONFIG_CACHE_ROOT_DIR:-<empty>}"
echo " Patients:    $TOTAL"
echo " Resume:      $RESUME"
if [[ "$AGENT" == "aml_auto_stamp" || "$AGENT" == "aml_stamp" ]]; then
    echo " STAMP root:  $STAMP_TRAIN_ROOT"
    echo " STAMP ext:   ${STAMP_EXTRACTORS[*]:-${EXTRACTOR}}"
fi
echo "═══════════════════════════════════════════════════════════════"

LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "$LOG_DIR"
VLM_BATCH_CSV="${OUTPUT_DIR}/agent_vlm_outputs.csv"

PASSED=0
FAILED=0
SKIPPED=0
TIMEDOUT=0

_run_slide() {
    if (( SLIDE_TIMEOUT > 0 )); then
        timeout --kill-after=15s "$SLIDE_TIMEOUT" "${RUN_CMD[@]}" "$@"
    else
        "${RUN_CMD[@]}" "$@"
    fi
}

append_agent_vlm_csv() {
    local src="$1"
    local dst="$2"
    [[ -f "$src" ]] || return 0
    if [[ ! -f "$dst" ]]; then
        cp "$src" "$dst"
    else
        tail -n +2 "$src" >> "$dst"
    fi
}

for i in "${!PATIENTS[@]}"; do
    ENTRY="${PATIENTS[$i]%%,*}"
    ENTRY="${ENTRY%\"}"
    ENTRY="${ENTRY#\"}"
    IDX=$((i + 1))

    # ── Resolve slide path (.mrxs) ──────────────────────────────────
    if [[ "$ENTRY" == *.mrxs ]]; then
        SLIDE="$ENTRY"
        PATIENT="$(basename "${SLIDE%.mrxs}")"
    else
        PATIENT="$ENTRY"
        SLIDE="${SLIDES_ROOT}/${PATIENT}.mrxs"
    fi
    if [[ ! -f "$SLIDE" ]]; then
        echo "[$IDX/$TOTAL] SKIP  $PATIENT — .mrxs not found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    OUTPUT_PATIENT="$(output_patient_name "$PATIENT")"
    SUMMARY="${OUTPUT_DIR}/${OUTPUT_PATIENT}/summary.json"

    # ── Resume: skip if already completed ───────────────────────────
    if $RESUME && [[ -f "$SUMMARY" ]]; then
        RESUME_STATE=$("$PYTHON_BIN" - "$SUMMARY" <<'PY'
import json
import sys

summary_path = sys.argv[1]

try:
    with open(summary_path) as handle:
        summary = json.load(handle)
except Exception:
    print("rerun:invalid summary.json")
    raise SystemExit

if str(summary.get("status") or "").strip() != "ok":
    print("rerun")
    raise SystemExit

summary_final_decision = str(summary.get("final_decision") or "").strip()
if summary_final_decision:
    print("skip")
    raise SystemExit

print("rerun:status=ok but final_decision is missing")
PY
)
        if [[ "$RESUME_STATE" == "skip" ]]; then
            echo "[$IDX/$TOTAL] DONE  $PATIENT (already completed, skipping)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if [[ "$RESUME_STATE" == rerun:* ]]; then
            echo "[$IDX/$TOTAL] RERUN $PATIENT — ${RESUME_STATE#rerun:}"
        fi
    fi

    # ── Run (suppress all python output) ───────────────────────────
    LOG="$(next_log_path "$PATIENT")"
    if [[ "$LOG" != "${LOG_DIR}/${PATIENT}.log" ]]; then
        echo "[$IDX/$TOTAL] LOG   $PATIENT -> ${LOG}"
    fi
    SLIDE_STARTED_EPOCH="$(date +%s)"

    RUN_CMD=(
        "$PYTHON_BIN"
        "$RUN_SINGLE_SLIDE"
        --slide "$SLIDE"
        --output-dir "$OUTPUT_DIR"
        --model "$MODEL"
        --extractor "$EXTRACTOR"
        --tile-filter "$TILE_FILTER"
        --tile-size-px "$TILE_SIZE_PX"
        --batch-size "$BATCH_SIZE"
        --roi-size-px "$ROI_SIZE_PX"
        --max-accepted-rois "$MAX_ACCEPTED_ROIS"
        --target-accepted-rois "$TARGET_ACCEPTED_ROIS"
        --default-mpp-um "$DEFAULT_MPP_UM"
        --agent "$AGENT"
    )
    if [[ "$AGENT" == "aml_auto_stamp" || "$AGENT" == "aml_stamp" ]]; then
        RUN_CMD+=(
            --stamp-train-root "$STAMP_TRAIN_ROOT"
            --stamp-repo-root "$STAMP_REPO_ROOT"
            --stamp-device "$STAMP_DEVICE"
            --stamp-accelerator "$STAMP_ACCELERATOR"
            --stamp-num-workers "$STAMP_NUM_WORKERS"
        )
        if [[ ${#STAMP_EXTRACTORS[@]} -gt 0 ]]; then
            RUN_CMD+=(--stamp-extractors "${STAMP_EXTRACTORS[@]}")
        else
            RUN_CMD+=(--stamp-extractors "$EXTRACTOR")
        fi
        if [[ ${#STAMP_TASKS[@]} -gt 0 ]]; then
            RUN_CMD+=(--stamp-tasks "${STAMP_TASKS[@]}")
        fi
        if $ALLOW_STAMP_TRAIN_VALID_OVERLAP; then
            RUN_CMD+=(--allow-stamp-train-valid-overlap)
        fi
    fi
    if $INCLUDE_MODEL_IN_OUTPUT_NAME; then
        RUN_CMD+=(--include-model-in-output-name)
    fi
    if [[ -n "$CUDA_DEVICE" ]]; then
        RUN_CMD+=(--cuda-device "$CUDA_DEVICE")
    fi
    RUN_CMD+=(--experiment-root "$EXPERIMENT_ROOT")
    if $USE_TILE_CACHE; then
        RUN_CMD+=(--use-tile-cache)
    fi

    _run_slide >"$LOG" 2>&1 && RUN_EXIT=0 || RUN_EXIT=$?

    SLIDE_ELAPSED_SECONDS=$(( $(date +%s) - SLIDE_STARTED_EPOCH ))
    if (( RUN_EXIT == 0 )); then
        PASSED=$((PASSED + 1))
        append_agent_vlm_csv "${OUTPUT_DIR}/${OUTPUT_PATIENT}/agent_vlm_output.csv" "$VLM_BATCH_CSV"
        echo "[$IDX/$TOTAL] OK    $PATIENT elapsed=$(format_elapsed "$SLIDE_ELAPSED_SECONDS")"
    elif (( RUN_EXIT == 124 )); then
        TIMEDOUT=$((TIMEDOUT + 1))
        echo "[$IDX/$TOTAL] TIMEOUT $PATIENT — exceeded ${SLIDE_TIMEOUT}s, skipping"
    else
        FAILED=$((FAILED + 1))
        echo "[$IDX/$TOTAL] FAIL  $PATIENT (exit=${RUN_EXIT}) elapsed=$(format_elapsed "$SLIDE_ELAPSED_SECONDS"), skipping"
        print_failure_reason "$SUMMARY" "$LOG"
    fi

    echo "[$IDX/$TOTAL] Progress: $PASSED ok / $FAILED fail / $TIMEDOUT timeout / $SKIPPED skip"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " BATCH COMPLETE"
echo " Total: $TOTAL  |  OK: $PASSED  |  FAIL: $FAILED  |  TIMEOUT: $TIMEDOUT  |  SKIP: $SKIPPED"
echo "═══════════════════════════════════════════════════════════════"
