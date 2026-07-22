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

def run_batch_2():
    ocr_client = RemoteOCRClient()
    
    # Load 1000 high-clarity image list selected for Batch-2
    batch2_list_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "batch_2_image_list.json")
    if not os.path.exists(batch2_list_path):
        print(f"Error: {batch2_list_path} not found. Please run select_batch_2_1000.py first.")
        return
        
    with open(batch2_list_path, "r", encoding="utf-8") as f:
        batch2_items = json.load(f)
        
    image_paths = [item["path"] for item in batch2_items]
    print(f"Loaded {len(image_paths)} high-clarity images for Batch-2 execution.")
    
    scanned_dir = Path(os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "scanned_images_batch2"))
    scanned_dir.mkdir(parents=True, exist_ok=True)
    
    # Check already done images in Batch-2 for resume safety
    manifest_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "ocr_results", "batch_2_manifest.json")
    already_done = set()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
                already_done = {r.get("image_name") for r in manifest_data if r.get("chunk") == "Batch-2"}
        except Exception:
            pass
            
    print(f"Resuming: {len(already_done)} images already in Batch-2. Skipping those.")
    
    # Build consist list containing all unique wagon numbers in Batch-2
    consist_list = []
    for p in image_paths:
        n = os.path.basename(p)
        m = re.search(r"^\d+?_(\d{11})_\d{17}", n)
        if m:
            consist_list.append(m.group(1))
    consist_list = sorted(list(set(consist_list)))
    print(f"Generated train consist list for alignment lookup: {len(consist_list)} wagons.")
    
    num_matches = 0
    num_mismatches = 0
    type_matches = 0
    type_mismatches = 0
    type_skipped = 0
    owner_matches = 0
    owner_mismatches = 0
    owner_skipped = 0
    mismatch_cases = []
    
    t_start = time.time()
    total_imgs = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        if img_name in already_done:
            print(f"[{i+1}/{total_imgs}] Skipping (already processed): {img_name}")
            continue
            
        print(f"\n[{i+1}/{total_imgs}] Processing Batch-2: {img_name}")
        
        # 1. Parse Ground Truth from filename
        gta_match = re.search(r"^\d+?_(\d{11})_\d{17}", img_name)
        if not gta_match:
            print(f"  Skipping {img_name}: invalid filename pattern")
            continue
            
        gt_number = gta_match.group(1)
        gt_type  = CODE_TO_WAGON_TYPE.get(gt_number[0:2])
        gt_owner = ZONE_CODE_MAP.get(gt_number[2:4])
        
        # 2. Temporary copy for scan
        temp_target_path = scanned_dir / f"temp_scan_b2_{int(time.time())}_{i}.jpg"
        shutil.copy2(img_path, temp_target_path)
        
        try:
            raw_res = run_pipeline(str(temp_target_path), consist_list, ocr_client, None)
            
            raw_number = raw_res.get("wagon_number", "").strip()
            raw_type = raw_res.get("wagon_type", "").strip()
            raw_owner = raw_res.get("owner_name", "").strip()
            
            detections = raw_res.get("detections", [])
            has_type_box = any(d.get("class") == "wagon_type" for d in detections)
            has_owner_box = any(d.get("class") in ["owner_name", "zone_markings"] for d in detections)
            
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
                    "ocr": raw_number
                })
                
            # Wagon Type
            if has_type_box:
                if raw_type == gt_type:
                    type_matches += 1
                    type_status = "Match"
                else:
                    type_mismatches += 1
                    type_status = "Mismatch"
                    mismatch_cases.append({
                        "image": img_name,
                        "field": "Wagon Type",
                        "gt": gt_type,
                        "ocr": raw_type
                    })
            else:
                type_skipped += 1
                type_status = "Excluded (No Box)"
                
            # Owner Zone
            if has_owner_box:
                if raw_owner == gt_owner:
                    owner_matches += 1
                    owner_status = "Match"
                else:
                    owner_mismatches += 1
                    owner_status = "Mismatch"
                    mismatch_cases.append({
                        "image": img_name,
                        "field": "Owner Zone",
                        "gt": gt_owner,
                        "ocr": raw_owner
                    })
            else:
                owner_skipped += 1
                owner_status = "Excluded (No Box)"
                
            res_ui = raw_res.copy()
            res_ui["chunk"] = "Batch-2"
            res_ui["image_name"] = img_name
            res_ui["gt_wagon_number"] = gt_number
            res_ui["gt_wagon_type"]   = gt_type
            res_ui["gt_owner_name"]   = gt_owner
            res_ui["wagon_number_ocr_match"] = num_ok
            res_ui["wagon_type_ocr_match"]   = (raw_type == gt_type) if (has_type_box and gt_type is not None) else None
            res_ui["owner_name_ocr_match"]   = (raw_owner == gt_owner) if (has_owner_box and gt_owner is not None) else None
            
            permanent_dest_path = scanned_dir / img_name
            if temp_target_path.exists():
                shutil.move(str(temp_target_path), str(permanent_dest_path))
                
            if res_ui.get("detections"):
                draw_annotations(permanent_dest_path, res_ui["detections"])
                
            res_ui = resolve_image_urls(res_ui)
            
            # Save to separate Batch-2 manifest file
            b2_manifest_path = Path(workspace_dir) / "cris_ocr_pipeline" / "outputs" / "ocr_results" / "batch_2_manifest.json"
            b2_records = []
            if b2_manifest_path.exists():
                try:
                    with open(b2_manifest_path, "r", encoding="utf-8") as mf:
                        b2_records = json.load(mf)
                except Exception:
                    b2_records = []
                    
            # Update or append
            idx_found = next((idx for idx, r in enumerate(b2_records) if r.get("image_name") == img_name), None)
            if idx_found is not None:
                b2_records[idx_found] = res_ui
            else:
                b2_records.append(res_ui)
                
            with open(b2_manifest_path, "w", encoding="utf-8") as mf:
                json.dump(b2_records, mf, indent=2)
                
            print(f"  Saved record to batch_2_manifest.json (Total Batch-2 items: {len(b2_records)})")
            
        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
            if temp_target_path.exists():
                os.remove(temp_target_path)

if __name__ == '__main__':
    run_batch_2()
