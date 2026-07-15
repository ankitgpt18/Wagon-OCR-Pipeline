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

print("Initializing PaddleOCR on GPU...")
try:
    ocr_client = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
    print("PaddleOCR initialized successfully on GPU.")
except Exception as e:
    print(f"Failed to initialize PaddleOCR on GPU: {e}")
    sys.exit(1)

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Use .predict() (modern API) with fallback to .ocr() (legacy)
        result = None
        try:
            result = ocr_client.predict(img)
        except Exception as pred_err:
            print(f"predict() failed: {pred_err}, trying legacy .ocr()")
            try:
                result = ocr_client.ocr(img)
            except Exception as ocr_err:
                print(f"Legacy .ocr() also failed: {ocr_err}")
                raise ocr_err

        # Format response with spatial sorting (Top-to-Bottom, Left-to-Right)
        lines_data = []
        lines = []

        if result is None:
            return {"lines": []}

        # Handle PP-OCRv6 / PaddleX predict() output format
        # Could be: dict with 'rec_texts', list of dicts, list of lists, etc.
        if isinstance(result, dict):
            # Direct dict: {'rec_texts': [...], 'rec_boxes': [...], ...}
            rec_texts = result.get("rec_texts", result.get("texts", []))
            if rec_texts:
                lines = list(rec_texts)
            elif "res" in result:
                # Nested: {'res': {'rec_texts': [...]}}
                inner = result["res"]
                if isinstance(inner, dict):
                    lines = list(inner.get("rec_texts", inner.get("texts", [])))
                elif isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict):
                            lines.extend(item.get("rec_texts", item.get("texts", [])))
        elif isinstance(result, list):
            for item in result:
                if item is None:
                    continue
                if isinstance(item, dict) or hasattr(item, "keys"):
                    # Dict-like item (including custom OCRResult object)
                    if not isinstance(item, dict):
                        try:
                            item = dict(item)
                        except Exception:
                            pass
                    if "text_rec_res" in item:
                        rec_texts = item["text_rec_res"].get("texts", item["text_rec_res"].get("rec_texts", []))
                        rec_boxes = item.get("text_det_res", {}).get("boxes", item.get("text_det_res", {}).get("dt_polys", []))
                    else:
                        rec_texts = item.get("rec_texts", item.get("texts", []))
                        rec_boxes = item.get("rec_boxes", item.get("dt_polys", []))
                    
                    if rec_texts:
                        for i, text in enumerate(rec_texts):
                            if i < len(rec_boxes):
                                box = rec_boxes[i]
                                try:
                                    # Box could be list of points [[x, y], ...] or dict points
                                    y_center = sum([pt[1] for pt in box]) / max(len(box), 1)
                                    x_center = sum([pt[0] for pt in box]) / max(len(box), 1)
                                except (TypeError, IndexError):
                                    y_center, x_center = i, 0
                                lines_data.append((y_center, x_center, str(text)))
                            else:
                                lines_data.append((i, 0, str(text)))
                elif isinstance(item, list):
                    # Legacy format: [[box, (text, conf)], ...]
                    for line in item:
                        if isinstance(line, (list, tuple)) and len(line) > 1:
                            box = line[0]
                            text_info = line[1]
                            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                            try:
                                y_center = sum([pt[1] for pt in box]) / max(len(box), 1)
                                x_center = sum([pt[0] for pt in box]) / max(len(box), 1)
                            except (TypeError, IndexError):
                                y_center, x_center = 0, 0
                            lines_data.append((y_center, x_center, str(text)))

        # Build final lines list
        if lines_data:
            lines_data.sort(key=lambda item: (item[0], item[1]))
            lines = [item[2] for item in lines_data]

        if not lines:
            # Last resort: try to stringify result
            print(f"OCR result format unrecognized. Type: {type(result)}, preview: {str(result)[:300]}")

        return {"lines": lines}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
