# import necessary packages
from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import random
import base64
import torch
import json
import csv
import re
import os
import sys
import time
import yaml


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

# Function to get video names using multiple filters 
def get_patient_names(clini_table):
    if '.xlsx' in  clini_table:
        data = pd.read_excel(clini_table)
    else:
        data = pd.read_csv(clini_table)
    patients = data["PATIENT"].dropna().astype(str).drop_duplicates().tolist()
    return patients

def load_images(image_dir, patient, slide_table, expected_count=10):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    slide_table = pd.read_csv(slide_table)
    slide_table = slide_table[slide_table["PATIENT"].astype(str) == str(patient)]

    if slide_table.empty:
        print(f"Warning: no slide found for patient {patient}. Skipping.")
        return None

    slide = slide_table["FILENAME"].values[0].replace('.h5', '')
    images = os.listdir(image_dir)
    images = [i for i in images if f"{slide}_" in i and i.lower().endswith(valid_exts)]
    images = sorted(os.path.join(image_dir, i) for i in images)

    if expected_count is not None and len(images) > expected_count:
        rng = random.Random(42)
        images = rng.sample(images, expected_count)
    elif expected_count is not None and len(images) < expected_count:
        print(f"Warning: found only {len(images)} images for patient {patient}, expected {expected_count}. Skipping.")
        return None

    return images, slide
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_encoded_images(image_paths):
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{'png' if path.lower().endswith('.png') else 'jpeg'};base64,{encode_image(path)}"
            },
        }
        for path in image_paths
    ]


def generate_diagnose(encoded_images, client, model_name, system_prompt, user_prompt_template):
    n_images = len(encoded_images)
    roi_label = "ROI-Bilder" if n_images != 1 else "ROI-Bild"
    user_prompt = (
        user_prompt_template
        .replace("{n_images}", str(n_images))
        .replace("{roi_label}", roi_label)
    )
    messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": user_prompt
                    },
                
                    *encoded_images  
                ]
            }
        ]
    response = client.chat.completions.create(
                    model=model_name,
                    messages = messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
    message = response.choices[0].message
    file_content = getattr(message, "content", None)
    if not file_content:
        file_content = getattr(message, "reasoning_content", None)
    if isinstance(file_content, list):
        parts = []
        for item in file_content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                else:
                    parts.append(str(item))
        file_content = "\n".join(parts).strip()
    elif file_content is None:
        file_content = ""
    return file_content


def parse_diagnose_json(raw_diagnose):
    if raw_diagnose is None:
        return None
    if isinstance(raw_diagnose, dict):
        return raw_diagnose
    if isinstance(raw_diagnose, list):
        return raw_diagnose
    
    raw_diagnose = raw_diagnose.strip()
    
    # Remove wrapper annotations and ANSI escape codes
    raw_diagnose = raw_diagnose.replace('<|begin_of_box|>', '')
    raw_diagnose = raw_diagnose.replace('<|end_of_box|>', '')
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    raw_diagnose = ansi_escape.sub('', raw_diagnose)
    
    # Remove markdown code blocks
    if raw_diagnose.startswith("```json"):
        raw_diagnose = raw_diagnose.strip("`").strip()
        first_newline = raw_diagnose.find('\n')
        if first_newline > 0:
            raw_diagnose = raw_diagnose[first_newline+1:]
        if raw_diagnose.endswith("```"):
            raw_diagnose = raw_diagnose.rsplit("```", 1)[0]
    
    # Trim everything before the first JSON object/array
    first_brace = min([idx for idx in [raw_diagnose.find('{'), raw_diagnose.find('[')] if idx != -1], default=-1)
    if first_brace > 0:
        raw_diagnose = raw_diagnose[first_brace:]
    
    raw_diagnose = raw_diagnose.strip()
    
    # Try direct parse first
    try:
        return json.loads(raw_diagnose)
    except json.JSONDecodeError:
        pass
    
    # GLM-specific fix: handle stringified arrays
    # Problem: "roi_assessment": "[
    #    {""roi_index": 1, ...}
    # ]" instead of "roi_assessment": [{...}]
    if '"roi_assessment": "[' in raw_diagnose:
        start_key = raw_diagnose.index('"roi_assessment": "[')
        array_start = raw_diagnose.index('"[', start_key) + 1
        array_end = raw_diagnose.rfind(']"')
        if array_start != -1 and array_end != -1 and array_end > array_start:
            stringified_array = raw_diagnose[array_start:array_end+1]
            stringified_array = stringified_array.replace('""', '"')
            stringified_array = stringified_array.replace('\\"', '"')
            stringified_array = stringified_array.replace('\\n', '\n')
            raw_diagnose = raw_diagnose[:array_start] + stringified_array + raw_diagnose[array_end+2:]
    
    # Fix 2: Double-quoted keys: ""roi_index" -> "roi_index"
    raw_diagnose = re.sub(r'"{2,}([^"]*?)"{2,}', r'"\1"', raw_diagnose)
    
    # Fix 3: Remove trailing commas before closing brackets/braces
    raw_diagnose = re.sub(r',\s*([}\]])', r'\1', raw_diagnose)
    
    # Try parse again
    try:
        return json.loads(raw_diagnose)
    except json.JSONDecodeError as e:
        print(f"❌ JSON error: {e.msg} at line {e.lineno}, col {e.colno}")
        print(f"   Content around error: ...{raw_diagnose[max(0, e.pos-40):min(len(raw_diagnose), e.pos+40)]}...")
        return None


# Removed: fix_json_string() - truncated responses cannot be reliably fixed.
# Instead, rely on max_tokens and batch size reduction to prevent truncation.

def sanity_check(entry):
    raw_diagnose = entry.get("Results", entry.get("Ergebnis"))
    if raw_diagnose is None:
        print(f"Warning: 'Results' field is missing.")
        return False
    diagnose_dict = parse_diagnose_json(raw_diagnose)
    return diagnose_dict is not None


def run_diagnose_with_retry(encoded_images, client, model_name, system_prompt, user_prompt, max_tries):
    tries = 0
    while tries < max_tries:
        tries += 1
        try:
            diagnose = generate_diagnose(
                encoded_images=encoded_images,
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt,
            )
        except Exception as exc:
            print(f"Warning: request failed (try {tries}/{max_tries}): {exc}")
            if tries < max_tries:
                time.sleep(min(tries * 2, 10))
            continue
        diagnose_dict = parse_diagnose_json(diagnose)
        if diagnose_dict is not None:
            return diagnose_dict
        if tries < max_tries:
            time.sleep(min(tries * 2, 10))
    return None


def write_json_atomic(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def process_patient(patient, image_dir, image_mode, expected_image_count, max_tries, client, model_name, system_prompt, user_prompt, slide_table):
    loaded_images = load_images(
        image_dir=image_dir,
        patient=patient,
        expected_count=expected_image_count if image_mode == "multiple" else None,
        slide_table=slide_table,
    )
    if loaded_images is None:
        return None

    images, slide = loaded_images
    total_image_mb = sum(os.path.getsize(image_path) for image_path in images) / (1024 * 1024)
    print(f"Patient {patient}: slide={slide}, images={len(images)}, total_image_mb={total_image_mb:.1f}")

    if image_mode == "multiple":
        roi_files = [os.path.basename(image_path) for image_path in images]
        encoded_images = build_encoded_images(images)
        diagnose = run_diagnose_with_retry(
            encoded_images=encoded_images,
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tries=max_tries,
        )
        if diagnose is None:
            print(f"Warning: max tries ({max_tries}) reached for patient {patient}. Skipping.")
            return None
        return {
            "Patient": patient,
            "FILENAME": slide,
            "ROI_FILES": roi_files,
            "Results": diagnose,
        }

    image_diagnosen = []
    for image_path in images:
        encoded_images = build_encoded_images([image_path])
        diagnose = run_diagnose_with_retry(
            encoded_images=encoded_images,
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tries=max_tries,
        )
        if diagnose is None:
            print(f"Warning: max tries ({max_tries}) reached for image {os.path.basename(image_path)} of patient {patient}. Skipping image.")
            continue
        image_diagnosen.append({"Image": os.path.basename(image_path), "Results": diagnose})
    return {
        "Patient": patient,
        "FILENAME": slide,
        "Image_Diagnosen": image_diagnosen,
    }

def write_results_to_csv(input_dir, output_file):
    input_dir = Path(input_dir)
    rows = []
    all_keys = set()

    def _append_payload(payload, patient_id, image_name=None, slide=None, roi_files=None):
        if isinstance(payload, dict):
            row = dict(payload)
            row["Patient"] = patient_id
            if slide is not None:
                row["FILENAME"] = slide
            if roi_files is not None:
                row["ROI_FILES"] = ", ".join(roi_files)
            if image_name is not None:
                row["Image"] = image_name
            all_keys.update(row.keys())
            rows.append(row)
            return

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["Patient"] = patient_id
                if slide is not None:
                    row["FILENAME"] = slide
                if roi_files is not None:
                    row["ROI_FILES"] = ", ".join(roi_files)
                if image_name is not None:
                    row["Image"] = image_name
                all_keys.update(row.keys())
                rows.append(row)
            return

        print(f"Warning: parsed diagnose payload is neither dict nor list for patient ID: {patient_id}")

    # Loop through all JSON files
    for file in input_dir.glob("*.json"):
        print(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        patient_id = data.get("Patient", file.stem)
        patient_id = patient_id.replace('_ROIs.json', '')
        slide = data.get("FILENAME")
        roi_files = data.get("ROI_FILES")
        
        # Handle single-image mode output first (one patient JSON with per-image diagnoses).
        if "Image_Diagnosen" in data and isinstance(data["Image_Diagnosen"], list):
            for idx, image_item in enumerate(data["Image_Diagnosen"], start=1):
                raw_diagnose = image_item.get("Results", image_item.get("Ergebnis"))
                image_name = image_item.get("Image", f"image_{idx}")
                if raw_diagnose is None:
                    print(f"Warning: missing image diagnosis for patient ID: {patient_id}, image: {image_name}")
                    continue
                diagnose_dict = parse_diagnose_json(raw_diagnose)
                if diagnose_dict is not None:
                    _append_payload(
                        diagnose_dict,
                        patient_id=patient_id,
                        image_name=image_name,
                        slide=slide,
                        roi_files=roi_files,
                    )
            continue

        # Step 1: Get the Results or legacy Ergebnis field
        raw_diagnose = data.get("Results", data.get("Ergebnis"))
        if raw_diagnose is not None:
            # Proceed with further processing.
            if isinstance(raw_diagnose, str):
                raw_diagnose = raw_diagnose.strip()
        else:
            print(f"Warning: 'Results' field is missing for patient ID: {patient_id}")
            continue  # Skip to the next file if no diagnose is found

        diagnose_dict = parse_diagnose_json(raw_diagnose)
            
        if diagnose_dict is not None:
            _append_payload(
                diagnose_dict,
                patient_id=patient_id,
                slide=slide,
                roi_files=roi_files,
            )

    # Assuming you already have a list of dicts in `rows` and all keys in `fieldnames`
    df = pd.DataFrame(rows)

    df = df[["Patient"] + [col for col in df.columns if col != "Patient"]]
    df.to_excel(output_file, index=False)

        
# main function
if __name__ == "__main__":
    roi_count = os.environ.get("VLM_ROI_COUNT", "10")
    project_root = Path(__file__).resolve().parents[1]
    
    SETTINGS = {
    "MAIN_DIR": os.environ.get("VLM_MAIN_DIR", f"/results/VLM/{roi_count}_ROIs"),
    "IMG_DIR": os.environ.get("VLM_IMG_DIR", "/data/private/manual_rois"),
    "PROMPT_FILE": os.environ.get("VLM_PROMPT_FILE", str(project_root / "Prompts" / "VLM_Prompts.yaml")),
    "BASE_URL": os.environ.get("VLM_BASE_URL", "https://YOUR_VLM_ENDPOINT/v1"),
    "API_KEY_FILE": os.environ.get("VLM_API_KEY_FILE", "/data/private/secrets/vlm_api_key.json"),
    "API_KEY_NAME": os.environ.get("VLM_API_KEY_NAME", "Pluto"),
    "PATIENT_TABLE": os.environ.get("VLM_PATIENT_TABLE", "/data/private/metadata/AML_HEALTHY-CLINI-DX_test.xlsx"),
    "SLIDE_TABLE": os.environ.get("VLM_SLIDE_TABLE", "/data/private/metadata/AML-HEALTHY-SLIDE-DX.csv"),
    "EXPECTED_IMAGE_COUNT_MULTIPLE": int(roi_count),
    "MAX_TRIES": 5,
    "REQUEST_TIMEOUT_SECONDS": 300,
    "ONLY_GENERATE_CSV": False,
    "AML_HEALTHY_EXPERIMENT": True,
    
    "MODEL_NAME": os.environ.get("VLM_MODEL_NAME", "Qwen3.5-397B-A17B-FP8"),
    "CSV_ONLY_OUT_DIR": os.environ.get("VLM_CSV_ONLY_OUT_DIR", "/results/vlm_existing_run"),
    "RESUME_OUT_DIR": '',  # e.g. "/results/vlm/run_YYYY-MM-DD_HH-MM-SS_MODEL_SUFFIX"
    }

    #for model_name in ["GLM-4.6V-FP8", "gemma-4-31B-it-h200", "Qwen3.5-397B-A17B-FP8"]: # SETTINGS["MODEL_NAME"]: "Qwen3.5-397B-A17B-FP8" or "GLM-4.6V-FP8"
    for model_name in [SETTINGS["MODEL_NAME"]]: # SETTINGS["MODEL_NAME"]: 
        for run_suffix in ["Simple prompt"]:
            for image_mode in ["multiple"]: #"IMAGE_MODE": "multiple" or "single"
                SETTINGS["MODEL_NAME"] = model_name
                SETTINGS["RUN_NAME_SUFFIX"] = f"{run_suffix}_{image_mode}"
                SETTINGS["IMAGE_MODE"] = image_mode
                if SETTINGS["ONLY_GENERATE_CSV"]:
                    out_dir = SETTINGS["CSV_ONLY_OUT_DIR"]
                else:
                    if SETTINGS["RESUME_OUT_DIR"]:
                        out_dir = SETTINGS["RESUME_OUT_DIR"]
                    else:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        folder_name = f"run_{timestamp}_{SETTINGS['MODEL_NAME']}_{SETTINGS['RUN_NAME_SUFFIX']}"
                        out_dir = os.path.join(SETTINGS["MAIN_DIR"], folder_name)
                os.makedirs(out_dir, exist_ok=True)
                log_path = os.path.join(out_dir, "run_report.txt")

                with open(log_path, "a", encoding="utf-8") as log_file:
                    stdout_tee = TeeStream(sys.stdout, log_file)
                    stderr_tee = TeeStream(sys.stderr, log_file)
                    with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                        print("=" * 80)
                        print(f"Run started: {datetime.now().isoformat()}")
                        print(f"Running with model: {model_name}, prompt: {run_suffix}, image mode: {image_mode}")

                        with open(SETTINGS["PROMPT_FILE"], "r") as f:
                            prompts = yaml.safe_load(f)
                        
                        SYSTEM_PROMPT = prompts.get("system_prompt", "")
                        #if SETTINGS["AML_HEALTHY_EXPERIMENT"]:
                        #    USER_PROMPT = prompts.get("user_prompt_aml_healthy", "")
                        #else:
                        #    USER_PROMPT = prompts.get("user_prompt_npm1", "")
                        
                        USER_PROMPT = prompts.get("user_prompt_combined", "")
                        if SETTINGS["ONLY_GENERATE_CSV"]:
                            diagnose_dir = os.path.join(out_dir, "diagnose")
                            write_results_to_csv(
                                input_dir=diagnose_dir,
                                output_file=os.path.join(out_dir, "results.xlsx"),
                            )
                        else:
                            if SETTINGS["RESUME_OUT_DIR"]:
                                print(f"Resuming existing run directory: {out_dir}")

                            diagnose_dir = os.path.join(out_dir, "diagnose")
                            os.makedirs(diagnose_dir, exist_ok=True)

                            code_copy_target = os.path.join(out_dir, "code_copy.txt")
                            shutil.copy(__file__, code_copy_target)
                            prompt_copy_target = os.path.join(out_dir, "VLM_Prompts.yaml")
                            shutil.copy(SETTINGS["PROMPT_FILE"], prompt_copy_target)

                            patients = get_patient_names(clini_table=SETTINGS["PATIENT_TABLE"])
                            completed_patients = {p.stem for p in Path(diagnose_dir).glob("*.json")}
                            if completed_patients:
                                print(f"Resume mode: skipping {len(completed_patients)} already completed patients.")

                            with open(SETTINGS["API_KEY_FILE"], "r") as f:
                                key_config = json.load(f)
                            client = OpenAI(
                                api_key=key_config[SETTINGS["API_KEY_NAME"]],
                                base_url=SETTINGS["BASE_URL"],
                                timeout=SETTINGS["REQUEST_TIMEOUT_SECONDS"],
                                max_retries=0,
                            )

                            print(f"Using user prompt template: {USER_PROMPT}")
                            for patient in tqdm(patients):
                                if patient in completed_patients:
                                    continue
                                print(f"Processing patient {patient}...")
                                entry = process_patient(
                                    patient=patient,
                                    image_dir=SETTINGS["IMG_DIR"],
                                    image_mode=SETTINGS["IMAGE_MODE"],
                                    expected_image_count=SETTINGS["EXPECTED_IMAGE_COUNT_MULTIPLE"],
                                    max_tries=SETTINGS["MAX_TRIES"],
                                    client=client,
                                    model_name=SETTINGS["MODEL_NAME"],
                                    system_prompt=SYSTEM_PROMPT,
                                    user_prompt=USER_PROMPT,
                                    slide_table=SETTINGS["SLIDE_TABLE"],
                                )
                                if entry is None:
                                    continue
                                diagnose_file = os.path.join(diagnose_dir, f"{patient}.json")
                                write_json_atomic(diagnose_file, entry)
                                completed_patients.add(patient)

                            write_results_to_csv(
                                input_dir=diagnose_dir,
                                output_file=os.path.join(out_dir, "results.xlsx"),
                            )

                        print(f"Run finished: {datetime.now().isoformat()}")
                        print(f"Saved text report to: {log_path}")
        
