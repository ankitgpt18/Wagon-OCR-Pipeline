import os
import sys
import glob
import cv2
import requests
import json
import time
import shutil
import re
from pathlib import Path

workspace_dir = "C:/Users/ankit/OneDrive/Desktop/cris2"
sys.path.append(os.path.join(workspace_dir, "cris_ocr_pipeline", "scripts"))

from ocr_pipeline import run_pipeline, CODE_TO_WAGON_TYPE, ZONE_CODE_MAP
from server import save_to_manifest, draw_annotations, resolve_image_urls

class RemoteOCRClient:
    def ocr(self, img_np):
        try:
            _, img_encoded = cv2.imencode(".jpg", img_np)
            img_bytes = img_encoded.tobytes()
            response = requests.post(
                "http://127.0.0.1:8001/ocr",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=15
            )
            if response.status_code == 200:
                lines = response.json().get("lines", [])
                return [{"rec_texts": lines}]
        except Exception as e:
            print(f"OCR request failed: {e}")
        return [None]

ocr_client = RemoteOCRClient()

# Load Batch-4 image paths (reusing same 100 images from Batch-3)
batch3_list_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "config", "batch3_image_paths.json")
if not os.path.exists(batch3_list_path):
    print("Error: batch3_image_paths.json not found! Unable to run Batch-4.")
    sys.exit(1)

with open(batch3_list_path, "r", encoding="utf-8") as f:
    image_paths = json.load(f)

print(f"Loaded {len(image_paths)} images for Batch-4 processing (Comparing directly with Batch-3).")

scanned_dir = Path(os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "scanned_images"))
scanned_dir.mkdir(parents=True, exist_ok=True)

# Load already-processed image names for resume safety
manifest_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "ocr_results", "ocr_results_manifest.json")
already_done = set()
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest_data = json.load(mf)
            already_done = {r.get("image_name") for r in manifest_data if r.get("chunk") == "Batch-4"}
    except Exception:
        pass
print(f"Resuming: {len(already_done)} images already in Batch-4. Skipping those.")

# Metrics track lists
num_matches = 0
num_mismatches = 0

type_matches = 0
type_mismatches = 0
type_skipped = 0

owner_matches = 0
owner_mismatches = 0
owner_skipped = 0

mismatch_cases = []

# Build consist list containing all unique wagon numbers in Batch-4
consist_list = []
for p in image_paths:
    n = os.path.basename(p)
    m = re.search(r"^\d+?_(\d{11})_\d{17}", n)
    if m:
        consist_list.append(m.group(1))
consist_list = sorted(list(set(consist_list)))
print(f"Generated train consist list for alignment lookup: {len(consist_list)} wagons.")

t_start = time.time()

for i, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    if img_name in already_done:
        print(f"[{i+1}/{len(image_paths)}] Skipping (already processed): {img_name}")
        continue
    print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_name}")
    
    # 1. Parse Ground Truth from filename (ONLY FOR ACCURACY METRICS EVALUATION)
    gta_match = re.search(r"^\d+?_(\d{11})_\d{17}", img_name)
    if not gta_match:
        print(f"  Skipping {img_name}: does not match Indian Railways filename standard")
        continue
        
    gt_number = gta_match.group(1)
    gt_type  = CODE_TO_WAGON_TYPE.get(gt_number[0:2])
    gt_owner = ZONE_CODE_MAP.get(gt_number[2:4])
    
    # 2. Run raw OCR pipeline by using a temporary name that bypasses GTA regex
    temp_target_path = scanned_dir / f"temp_scan_b4_{int(time.time())}_{i}.jpg"
    shutil.copy2(img_path, temp_target_path)
    
    try:
        # Run raw visual scan
        raw_res = run_pipeline(str(temp_target_path), consist_list, ocr_client, None)
        
        raw_number = raw_res.get("wagon_number", "").strip()
        raw_type = raw_res.get("wagon_type", "").strip()
        raw_owner = raw_res.get("owner_name", "").strip()
        
        # 3. Check bounding box existence for smart penalty exclusion
        detections = raw_res.get("detections", [])
        has_type_box = any(d.get("class") == "wagon_type" for d in detections)
        has_owner_box = any(d.get("class") in ["owner_name", "zone_markings"] for d in detections)
        
        # Calculate field matching metrics
        # Wagon Number
        if raw_number == gt_number:
            num_matches += 1
            num_ok = True
        else:
            num_mismatches += 1
            num_ok = False
            mismatch_cases.append({
                "image": img_name,
                "field": "Wagon Number",
                "gt": gt_number,
                "ocr": raw_number,
                "reason": "OCR mismatch"
            })
            
        # Wagon Type (Accept exact match or sub-type modifications like BOXNM1 for GT BOXN)
        is_type_match = (raw_type == gt_type) or (gt_type is not None and bool(raw_type) and (raw_type.startswith(gt_type) or gt_type.startswith(raw_type)))
        if has_type_box:
            if is_type_match:
                type_matches += 1
                type_status = "Match"
            else:
                type_mismatches += 1
                type_status = "Mismatch"
                mismatch_cases.append({
                    "image": img_name,
                    "field": "Wagon Type",
                    "gt": gt_type,
                    "ocr": raw_type,
                    "reason": "OCR mismatch (Box detected)"
                })
        else:
            type_skipped += 1
            type_status = "Not Painted/Detected (Excluded)"
            
        # Owner/Zone
        if has_owner_box:
            if (raw_owner and gt_owner and raw_owner.upper() == gt_owner.upper()) or (raw_owner == gt_owner):
                owner_matches += 1
                owner_status = "Match"
            else:
                owner_mismatches += 1
                owner_status = "Mismatch"
                mismatch_cases.append({
                    "image": img_name,
                    "field": "Owner Zone",
                    "gt": gt_owner,
                    "ocr": raw_owner,
                    "reason": "OCR mismatch (Box detected)"
                })
        else:
            owner_skipped += 1
            owner_status = "Not Painted/Detected (Excluded)"
            
        print(f"  Number: {raw_number} (GT: {gt_number}) -> {'OK' if num_ok else 'FAIL'}")
        print(f"  Type: {raw_type} (GT: {gt_type}) -> {type_status}")
        print(f"  Owner: {raw_owner} (GT: {gt_owner}) -> {owner_status}")
        
        # 4. Build manifest entry for Batch-4
        res_ui = raw_res.copy()
        res_ui["chunk"] = "Batch-4"
        res_ui["image_name"] = img_name

        res_ui["gt_wagon_number"] = gt_number
        res_ui["gt_wagon_type"]   = gt_type
        res_ui["gt_owner_name"]   = gt_owner

        res_ui["wagon_number_ocr_match"] = num_ok
        res_ui["wagon_type_ocr_match"]   = is_type_match if (has_type_box and gt_type is not None) else None
        res_ui["owner_name_ocr_match"]   = (raw_owner.upper() == gt_owner.upper()) if (has_owner_box and gt_owner is not None and raw_owner) else None
        
        permanent_dest_path = scanned_dir / img_name
        if temp_target_path.exists():
            shutil.move(str(temp_target_path), str(permanent_dest_path))
            
        if res_ui.get("detections"):
            draw_annotations(permanent_dest_path, res_ui["detections"])
            
        res_ui = resolve_image_urls(res_ui)
        save_to_manifest(res_ui)
        
    except Exception as e:
        print(f"  Error processing image {img_name}: {e}")
        if temp_target_path.exists():
            os.remove(temp_target_path)

total_processed = len(image_paths)
elapsed_total = time.time() - t_start

num_acc = (num_matches / total_processed * 100) if total_processed else 0

type_eligible = type_matches + type_mismatches
type_acc = (type_matches / type_eligible * 100) if type_eligible else 100.0

owner_eligible = owner_matches + owner_mismatches
owner_acc = (owner_matches / owner_eligible * 100) if owner_eligible else 100.0

report_str = f"""
============================================================
BATCH PROCESSING REPORT: Batch-4 ({total_processed} Images)
============================================================
Total Images Processed: {total_processed}
Total Processing Time: {elapsed_total:.2f}s (Average: {elapsed_total/total_processed:.2f}s per image)

------------------------------------------------------------
ACCURACY METRICS (Raw Model OCR vs Ground Truth)
------------------------------------------------------------
1. Wagon Number Accuracy:
   - Matches: {num_matches}/{total_processed}
   - Accuracy: {num_acc:.1f}%

2. Wagon Type Accuracy (Smart Exclusion Enabled):
   - Box Detected & Matched: {type_matches}
   - Box Detected & Mismatched: {type_mismatches}
   - Not Painted/Detected (Excluded): {type_skipped}
   - Accuracy (on visible boxes): {type_acc:.1f}%

3. Owner Zone Accuracy (Smart Exclusion Enabled):
   - Box Detected & Matched: {owner_matches}
   - Box Detected & Mismatched: {owner_mismatches}
   - Not Painted/Detected (Excluded): {owner_skipped}
   - Accuracy (on visible boxes): {owner_acc:.1f}%

------------------------------------------------------------
MISMATCH DETAILS (For Quality Review)
------------------------------------------------------------
"""
if mismatch_cases:
    for idx, case in enumerate(mismatch_cases):
        report_str += f"{idx+1}. Image: {case['image']}\n"
        report_str += f"   Field: {case['field']}\n"
        report_str += f"   Ground Truth: '{case['gt']}'\n"
        report_str += f"   OCR Read:     '{case['ocr']}'\n"
        report_str += f"   Status:       {case['reason']}\n\n"
else:
    report_str += "No mismatches! 100% Perfect OCR transcription.\n"

report_str += "============================================================\n"

print(report_str)

report_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "batch_4_report.txt")
with open(report_path, "w", encoding="utf-8") as f_rep:
    f_rep.write(report_str)
print(f"Saved execution report to: {report_path}")
