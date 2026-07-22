import torch
import os
import sys
import glob
import cv2
import requests
import json
import time
import shutil
from pathlib import Path

workspace_dir = "C:/Users/ankit/OneDrive/Desktop/cris2"
sys.path.append(os.path.join(workspace_dir, "cris_ocr_pipeline", "scripts"))

from ocr_pipeline import run_pipeline
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

# Get first 20 images
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
raw_images_dir = os.path.join(workspace_dir, "raw_images")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(raw_images_dir, ext)))
    image_paths.extend(glob.glob(os.path.join(raw_images_dir, ext.upper())))

image_paths = sorted(list(set(image_paths)))[:20]

print(f"Importing {len(image_paths)} images to the UI database...")

scanned_dir = Path(os.path.join(workspace_dir, "cris_ocr_pipeline", "outputs", "scanned_images"))
scanned_dir.mkdir(parents=True, exist_ok=True)

for i, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    print(f"[{i+1}/20] Importing {img_name}...")
    
    # Permanent target path for the UI server
    dest_path = scanned_dir / img_name
    shutil.copy2(img_path, dest_path)
    
    try:
        # Run the full OCR pipeline
        res = run_pipeline(str(dest_path), None, ocr_client, None)
        
        # Draw annotated box overlays
        if res.get("detections"):
            draw_annotations(dest_path, res["detections"])
            
        # Resolve static URLs
        res = resolve_image_urls(res)
        
        # Save to database manifest
        save_to_manifest(res)
        print(f"  Successfully imported: {res.get('wagon_number')} | Valid: {res.get('wagon_number_checksum_valid')}")
    except Exception as e:
        print(f"  Failed to import {img_name}: {e}")

print("\nBatch UI import completed!")
