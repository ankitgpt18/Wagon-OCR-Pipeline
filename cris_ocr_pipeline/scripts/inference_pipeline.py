#!/usr/bin/env python3
"""
Wagon OCR Inference Pipeline
Uses Roboflow model for text region detection and PaddleOCR for text recognition
"""
import argparse
import os
import json
from pathlib import Path
from PIL import Image
from inference_sdk import InferenceHTTPClient
from paddleocr import PaddleOCR
import csv


def load_config():
    """Load pipeline configuration"""
    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_models(config, api_key):
    """Initialize Roboflow and PaddleOCR models"""
    # Initialize Roboflow client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False
    )
    
    return client, ocr


def run_inference(image_path, client, config, ocr):
    """Run full inference pipeline on a single image"""
    results = {
        "image_path": str(image_path),
        "detections": [],
        "ocr_results": []
    }
    
    try:
        # Step 1: Detect text regions with Roboflow
        roboflow_result = client.infer(
            str(image_path),
            model_id=config["roboflow_model_id"]
        )
        
        results["detections"] = roboflow_result[0]["predictions"]
        
        # Step 2: OCR on each detected region
        image = Image.open(image_path).convert("RGB")
        
        for det in results["detections"]:
            # Extract bounding box
            x1 = int(det["x"] - det["width"] / 2)
            y1 = int(det["y"] - det["height"] / 2)
            x2 = int(det["x"] + det["width"] / 2)
            y2 = int(det["y"] + det["height"] / 2)
            
            # Crop region
            crop = image.crop((x1, y1, x2, y2))
            
            # Run PaddleOCR
            ocr_result = ocr.ocr(np.array(crop), cls=True)
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    results["ocr_results"].append({
                        "class": det["class"],
                        "text": text,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
    
    except Exception as e:
        results["error"] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Wagon OCR Inference Pipeline")
    parser.add_argument("--images", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output JSON file")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Please provide Roboflow API key via --api-key or ROBOFLOW_API_KEY environment variable")
    
    # Initialize models
    client, ocr = init_models(config, api_key)
    
    # Process images
    image_path = Path(args.images)
    all_results = []
    
    if image_path.is_dir():
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in image_path.iterdir() if f.suffix.lower() in img_extensions]
        
        for img_file in image_files:
            print(f"Processing {img_file.name}...")
            result = run_inference(img_file, client, config, ocr)
            all_results.append(result)
    else:
        result = run_inference(image_path, client, config, ocr)
        all_results.append(result)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    import numpy as np
    main()
