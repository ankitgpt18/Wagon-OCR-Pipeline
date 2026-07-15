# CRIS Wagon OCR Pipeline

This pipeline is prepared for a high-accuracy wagon OCR system where the final goal is:

```text
wagon image -> wagon number OCR -> validated wagon number -> name/type/owner lookup
```

The current dataset is image-only. Training can start after annotations arrive, but useful preparation can happen now.

## Recommended Free Model Stack

Use **PaddleOCR / PP-OCRv5** as the primary OCR stack.

Why this choice:

- Apache-2.0 licensed, which is easier for government/enterprise use than AGPL-licensed alternatives.
- Strong two-stage OCR design: text detection + text recognition.
- Lightweight enough for deployment and retraining.
- Supports custom fine-tuning once wagon-specific annotations are ready.

Avoid making Ultralytics YOLO the default dependency for production unless licensing is approved, because recent Ultralytics packages use AGPL/commercial licensing.

## Target Architecture

```text
raw image
  -> image QA and normalization checks
  -> text/wagon-number region detector
  -> OCR recognizer
  -> wagon-number parser and checksum/rule validation
  -> wagon master lookup
  -> response: wagon_no, name, type, owner, confidence, evidence
```

## Annotation Contract

Ask the annotation team to return one CSV file using this format:

```csv
image,x1,y1,x2,y2,x3,y3,x4,y4,text,label
10002480190_12130632037_08062026105439440.jpg,10,20,180,20,180,70,10,70,12130632037,wagon_number
```

Rules:

- `image` must match an image file in the dataset folder.
- Coordinates are pixel coordinates in the original image.
- Use 4-point quadrilateral boxes, not just horizontal rectangles, because wagon text may be slanted.
- `label` should be one of:
  - `wagon_number`
  - `zone`
  - `wagon_type`
  - `load_tare`
  - `other_text`
- For the first model, prioritize `wagon_number`. Other labels can be secondary.

## Current Scripts

Run from the dataset folder:

```powershell
python .\cris_ocr_pipeline\scripts\audit_dataset.py --images .
python .\cris_ocr_pipeline\scripts\create_splits.py --audit .\cris_ocr_pipeline\outputs\dataset_audit.csv
python .\cris_ocr_pipeline\scripts\validate_annotations.py --images . --annotations .\annotations.csv
```

## New Pipeline Scripts

### Convert YOLO Annotations to CSV
Convert auto-generated YOLO annotations to our project's CSV annotation format:
```powershell
python .\cris_ocr_pipeline\scripts\convert_yolo_to_csv.py --labels .\cris_ocr_pipeline\outputs\auto_annotations_v1_1000\labels --images . --classes .\cris_ocr_pipeline\outputs\auto_annotations_v1_1000\classes.txt --output auto_annotations.csv
```

### Convert Annotations to PaddleOCR Format
Convert annotations to PaddleOCR training format:
```powershell
python .\cris_ocr_pipeline\scripts\convert_to_paddleocr.py --annotations annotations.csv --images . --output paddleocr_labels
```

### Inference Pipeline
Run the full inference pipeline (Roboflow detection + PaddleOCR recognition:
```powershell
$env:ROBOFLOW_API_KEY="YOUR_KEY"
python .\cris_ocr_pipeline\scripts\inference_pipeline.py --images . --output inference_results.json
```

## When Annotations Arrive

1. Put annotation CSV at the dataset root as `annotations.csv`.
2. Run annotation validation:
   ```powershell
   python .\cris_ocr_pipeline\scripts\validate_annotations.py --images . --annotations .\annotations.csv
   ```
3. Fix missing files, invalid boxes, blank text, and label mistakes.
4. Convert to PaddleOCR detection/recognition training format.
5. Fine-tune detection first, then recognition.
6. Evaluate on held-out wagon IDs, not random images only.

## Auto-Annotate With Roboflow Baseline
...
The script skips images inside `sample_200_for_annotation` by default, keeps raw model outputs for audit, and writes YOLO `.txt` files using the 5-class order in `classes.txt`.Outputs are written under:

```text
cris_ocr_pipeline/outputs/
```

## When Annotations Arrive

1. Put annotation CSV at the dataset root as `annotations.csv`.
2. Run annotation validation.
3. Fix missing files, invalid boxes, blank text, and label mistakes.
4. Convert to PaddleOCR detection/recognition training format.
5. Fine-tune detection first, then recognition.
6. Evaluate on held-out wagon IDs, not random images only.

## Auto-Annotate With Roboflow Baseline

After training a Roboflow baseline detector, use it to generate candidate labels for unreviewed images:

```powershell
pip install -U inference-sdk pillow
$env:ROBOFLOW_API_KEY="YOUR_KEY"
python .\cris_ocr_pipeline\scripts\roboflow_auto_annotate.py --images . --limit 50
```

Start with `--limit 50` first. Check the generated labels and raw JSON under:

```text
cris_ocr_pipeline/outputs/auto_annotations_v1/
```

When the first 50 look reasonable, run a larger batch:

```powershell
python .\cris_ocr_pipeline\scripts\roboflow_auto_annotate.py --images . --limit 1000
```

The script skips images inside `sample_200_for_annotation` by default, keeps raw model outputs for audit, and writes YOLO `.txt` files using the 5-class order in `classes.txt`.

## Accuracy Notes

For CRIS-grade accuracy, do not rely only on OCR confidence. Use these guardrails:

- Split train/val/test by wagon identity to avoid duplicate leakage.
- Keep a separate final test set from different dates/locations if possible.
- Validate wagon number length and pattern.
- Return low-confidence cases for manual review.
- Use a wagon master table for `name`, `type`, and `owner`; do not hallucinate those fields from images.
