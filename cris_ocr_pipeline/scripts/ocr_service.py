import os
import sys
import site
from pathlib import Path
from unittest.mock import MagicMock

# CRITICAL: Disable PaddlePaddle PIR and oneDNN to prevent runtime crash
# "ConvertPirAttribute2RuntimeAttribute not support" error on Windows
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

# Step 1: Mock modelscope to prevent paddlex from importing torch in this process
sys.modules['modelscope'] = MagicMock()

# Step 2: Add CUDA DLL search paths on Windows BEFORE importing paddle
if os.name == "nt":
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            for p in nvidia_path.rglob("bin"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR

app = FastAPI(title="CRIS GPU OCR Service")

print("Initializing PaddleOCR models on CPU (Hindi and English)...")
try:
    ocr_hi = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="hi",
        use_gpu=False,
        enable_mkldnn=False
    )
    ocr_en = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        use_gpu=False,
        enable_mkldnn=False
    )
    print("PaddleOCR models initialized successfully.")
except Exception as e:
    print(f"Failed to initialize PaddleOCR models: {e}")
    sys.exit(1)

def parse_result(result):
    lines_data = []
    if result is None:
        return lines_data

    # Handle PP-OCRv6 / PaddleX predict() output format
    if isinstance(result, dict):
        rec_texts = result.get("rec_texts", result.get("texts", []))
        if rec_texts:
            for i, text in enumerate(rec_texts):
                lines_data.append((i, 0, str(text), 1.0, None))
        elif "res" in result:
            inner = result["res"]
            if isinstance(inner, dict):
                inner_texts = inner.get("rec_texts", inner.get("texts", []))
                for i, text in enumerate(inner_texts):
                    lines_data.append((i, 0, str(text), 1.0, None))
            elif isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        inner_texts = item.get("rec_texts", item.get("texts", []))
                        for i, text in enumerate(inner_texts):
                            lines_data.append((i, 0, str(text), 1.0, None))
    elif isinstance(result, list):
        for item in result:
            if item is None:
                continue
            if isinstance(item, dict) or hasattr(item, "keys"):
                if not isinstance(item, dict):
                    try:
                        item = dict(item)
                    except Exception:
                        pass
                if "text_rec_res" in item:
                    rec_texts = item["text_rec_res"].get("texts", item["text_rec_res"].get("rec_texts", []))
                    rec_confs = item["text_rec_res"].get("scores", [])
                    rec_boxes = item.get("text_det_res", {}).get("boxes", item.get("text_det_res", {}).get("dt_polys", []))
                else:
                    rec_texts = item.get("rec_texts", item.get("texts", []))
                    rec_confs = item.get("scores", [])
                    rec_boxes = item.get("rec_boxes", item.get("dt_polys", []))
                
                if rec_texts:
                    for i, text in enumerate(rec_texts):
                        conf = rec_confs[i] if i < len(rec_confs) else 1.0
                        if i < len(rec_boxes):
                            box = rec_boxes[i]
                            try:
                                y_center = sum([pt[1] for pt in box]) / max(len(box), 1)
                                x_center = sum([pt[0] for pt in box]) / max(len(box), 1)
                                min_x = min(pt[0] for pt in box)
                                max_x = max(pt[0] for pt in box)
                                min_y = min(pt[1] for pt in box)
                                max_y = max(pt[1] for pt in box)
                                bbox = [min_x, min_y, max_x, max_y]
                            except (TypeError, IndexError):
                                y_center, x_center = i, 0
                                bbox = None
                            lines_data.append((y_center, x_center, str(text), conf, bbox))
                        else:
                            lines_data.append((i, 0, str(text), conf, None))
            elif isinstance(item, list):
                for line in item:
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        box = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) > 1:
                            text = text_info[0]
                            conf = text_info[1]
                        else:
                            text = str(text_info)
                            conf = 1.0
                        try:
                            y_center = sum([pt[1] for pt in box]) / max(len(box), 1)
                            x_center = sum([pt[0] for pt in box]) / max(len(box), 1)
                            min_x = min(pt[0] for pt in box)
                            max_x = max(pt[0] for pt in box)
                            min_y = min(pt[1] for pt in box)
                            max_y = max(pt[1] for pt in box)
                            bbox = [min_x, min_y, max_x, max_y]
                        except (TypeError, IndexError):
                            y_center, x_center = 0, 0
                            bbox = None
                        lines_data.append((y_center, x_center, str(text), conf, bbox))
    return lines_data

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run legacy .ocr() for both Hindi and English models
        result_hi = None
        result_en = None
        try:
            result_hi = ocr_hi.ocr(img)
        except Exception as ocr_err:
            print(f"PaddleOCR (hi) .ocr() failed: {ocr_err}")

        try:
            result_en = ocr_en.ocr(img)
        except Exception as ocr_err:
            print(f"PaddleOCR (en) .ocr() failed: {ocr_err}")

        # Parse and combine results
        lines_data_hi = parse_result(result_hi)
        lines_data_en = parse_result(result_en)
        combined = lines_data_hi + lines_data_en

        # Deduplicate overlapping boxes spatially (NMS-style)
        # Sort by confidence score descending
        combined.sort(key=lambda item: item[3], reverse=True)

        keep_lines = []
        for y, x, text, conf, bbox in combined:
            text_clean = text.strip()
            if not text_clean:
                continue
            
            # Check overlap against already kept lines
            is_duplicate = False
            if bbox is not None:
                for ky, kx, ktext, kconf, kbbox in keep_lines:
                    if kbbox is not None:
                        xA = max(bbox[0], kbbox[0])
                        yA = max(bbox[1], kbbox[1])
                        xB = min(bbox[2], kbbox[2])
                        yB = min(bbox[3], kbbox[3])
                        
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        if interArea > 0:
                            areaA = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            areaB = (kbbox[2] - kbbox[0]) * (kbbox[3] - kbbox[1])
                            overlap = interArea / min(areaA, areaB)
                            if overlap > 0.4:
                                is_duplicate = True
                                break
            
            if not is_duplicate:
                text_key = text_clean.upper()
                if any(k[2].upper() == text_key for k in keep_lines):
                    is_duplicate = True
                    
            if not is_duplicate:
                keep_lines.append((y, x, text_clean, conf, bbox))

        # Sort spatially top-to-bottom, then left-to-right
        keep_lines.sort(key=lambda item: (item[0], item[1]))
        lines = [item[2] for item in keep_lines]

        return {"lines": lines}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
