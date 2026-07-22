import torch
import os
import sys
import glob
import cv2
import requests
import json
import time

workspace_dir = "C:/Users/ankit/OneDrive/Desktop/cris2"
sys.path.append(os.path.join(workspace_dir, "cris_ocr_pipeline", "scripts"))

from ocr_pipeline import run_pipeline

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

# Get first 20 images
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
raw_images_dir = os.path.join(workspace_dir, "raw_images")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(raw_images_dir, ext)))
    image_paths.extend(glob.glob(os.path.join(raw_images_dir, ext.upper())))

image_paths = sorted(list(set(image_paths)))[:20]

print(f"Found {len(image_paths)} images for batch processing.")

results = []
out_txt_path = os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "batch_results.txt")
os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

with open(out_txt_path, "w", encoding="utf-8") as f:
    f.write("Batch Processing OCR Results (First 20 Images)\n")
    f.write("==============================================\n\n")

for i, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    print(f"[{i+1}/20] Processing {img_name}...")
    t0 = time.time()
    
    try:
        # Run pipeline
        res = run_pipeline(img_path, None, ocr_client, None)
        elapsed = time.time() - t0
        res["processing_time_sec"] = round(elapsed, 2)
        
        # Save output to text file
        output_str = f"Image: {img_name}\n"
        output_str += f"  Wagon Number: {res.get('wagon_number')} (Valid: {res.get('wagon_number_checksum_valid')}, Verified: {res.get('wagon_number_verified')})\n"
        output_str += f"  Wagon Type: {res.get('wagon_type')} (Verified: {res.get('wagon_type_verified')})\n"
        output_str += f"  Owner/Zone: {res.get('owner_name')} (Verified: {res.get('owner_name_verified')})\n"
        output_str += f"  Tare Details: {res.get('tare_load_details')} (Verified: {res.get('tare_load_details_verified')})\n"
        output_str += f"  Safety Stencil: {res.get('stencil')}\n"
        output_str += f"  Processing Time: {res.get('processing_time_sec')}s\n"
        output_str += "----------------------------------------------\n"
        
        with open(out_txt_path, "a", encoding="utf-8") as f_out:
            f_out.write(output_str)
            
        results.append(res)
        print(f"  Result: {res.get('wagon_number')} | Type: {res.get('wagon_type')} | Time: {res.get('processing_time_sec')}s")
    except Exception as e:
        err_str = f"Image: {img_name}\n  Error: {e}\n----------------------------------------------\n"
        with open(out_txt_path, "a", encoding="utf-8") as f_out:
            f_out.write(err_str)
        print(f"  Failed to process {img_name}: {e}")

# Summarize metrics
total = len(results)
verified_nums = sum(1 for r in results if r.get("wagon_number_verified"))
verified_types = sum(1 for r in results if r.get("wagon_type_verified"))
verified_owners = sum(1 for r in results if r.get("owner_name_verified"))
verified_tares = sum(1 for r in results if r.get("tare_load_details_verified"))

summary_str = f"\nSummary of Batch Run:\n"
summary_str += f"  Total Images Processed: {total}\n"
summary_str += f"  Wagon Numbers Verified: {verified_nums}/{total} ({round(verified_nums/total*100, 2) if total else 0}%)\n"
summary_str += f"  Wagon Types Verified: {verified_types}/{total} ({round(verified_types/total*100, 2) if total else 0}%)\n"
summary_str += f"  Owner Names Verified: {verified_owners}/{total} ({round(verified_owners/total*100, 2) if total else 0}%)\n"
summary_str += f"  Tare Details Verified: {verified_tares}/{total} ({round(verified_tares/total*100, 2) if total else 0}%)\n"

with open(out_txt_path, "a", encoding="utf-8") as f_out:
    f_out.write(summary_str)

print("\nBatch process completed successfully!")
print(summary_str)
