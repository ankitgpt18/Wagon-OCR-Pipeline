# CRIS Wagon OCR Pipeline - AI Agent Context & Rules

This file provides deep architectural context, startup sequences, and operating rules for AI agents contributing to this codebase.

---

## 📂 Project Overview & Mission
This system is a production-grade Wagon OCR and Stencil audit pipeline designed for Indian Railways (CRIS). It localizes and extracts wagon details (number, type, owner, zone, tare) from high-resolution raw images, performs cross-validation via S4 Mod-10 checksum validation and ROAMS master registries, and classifies maintenance stencil presence.

---

## ⚙️ System Architecture

### 1. Model Configuration
*   **Localization model**: YOLOv11m (`cris_ocr_pipeline/models/best.pt`) custom-trained to localize 5 parameter classes:
    *   `wagon_number` (11-digit standard)
    *   `wagon_type` (e.g. `BOXN`, `BOBYN`)
    *   `owner_name` / `zone_markings` (e.g. `SEC`, `SE`, `SER`)
    *   `tare_load_details` (numeric text blocks)
*   **Stencil presence model**: YOLOv11m (`cris_ocr_pipeline/models/stencil_best.pt`) custom-trained to classify presence of maintenance safety stencils.
*   **Kaggle model format**: Custom weights under `cris_ocr_pipeline/models/best/` can be loaded or converted for fine-tuning/inference experiments.

### 2. Multi-Tier OCR Ensemble Strategy
To handle extreme rust, weathered paint, and text degradation, the pipeline executes a 5-tier fallback flow:
*   **Tier 1**: Qwen2.5-VL Vision-Language Model OCR on crops (supports bilingual English + Devanagari Hindi, split layout reading, and full code suffixes).
*   **Tier 2**: PaddleOCR on preprocessed crop (deskewed + contrast-boosted with LAB CLAHE + resized to 80px target height).
*   **Tier 3**: PaddleOCR on raw BGR crop (uses original bounding box coordinates).
*   **Tier 4**: Bounding Box Margin Iteration (adaptive 5%, 10%, 15% padding expansion for cut-off edge characters).
*   **Tier 5**: Full Image Scan (scans entire frame with PaddleOCR to recover 11-digit wagon numbers when crops are missed).

### 3. Checksum & Cross-Validation Logic
*   **S4 Mod-10 Checksum**: Matches Indian Railways standard for 11-digit wagon numbers.
    *   *Algorithm*: Sum weights of odd/even position digits, calculate modulo 10 remainder, and verify if it matches the 11th digit.
    *   *Combinatorial correction*: If a digit is misread, the pipeline generates valid candidates using Levenshtein distance against ROAMS zone-owner matrices to find the most probable match.

---

## 🚀 Local Development Startup Sequence

For anyone running the codebase, the processes should be executed in this order:

### Step 1: Pre-warming the GPU OCR Service
Always start the GPU PaddleOCR microservice first:
```bash
python cris_ocr_pipeline/scripts/ocr_service.py
```
*   Runs on `http://127.0.0.1:8001`
*   Requires oneDNN (MKLDNN) disabled to prevent crashes on Windows:
    *   `enable_mkldnn=False` parameter during `PaddleOCR()` initialization.
    *   `FLAGS_enable_pir_api=0` and `FLAGS_use_mkldnn=0` env flags.

### Step 2: Starting the FastAPI Server
The main web server connects the backend pipeline to the frontend:
```bash
python start_server.py
```
*   Runs on `http://127.0.0.1:8000`
*   Orchestrates file uploads, calls the OCR service, and matches the master ROAMS registry database.

### Step 3: Launching the Frontend React App
Run the user interface dashboard:
```bash
cd frontend
npm install
npm run dev
```
*   Runs on `http://localhost:5173`

---

## 📌 Path Rules & Conventions
1.  **Image dataset location**: All raw test images should be placed in `raw_images/` at the workspace root.
2.  **ROAMS Database**: Master registries and config parameters are loaded from `cris_ocr_pipeline/config/pipeline_config.json`.
3.  **Environment Setup**: VS Code automatically reads path resolutions from the root `.env` and `.vscode/settings.json`.

---

## 🤖 Agent Operating Rules
*   **CUDA Memory Safety**: Keep OCR context pre-warmed. Avoid re-initializing PaddleOCR or EasyOCR inside loops; always route requests through the microservice at `port 8001`.
*   **Humanized Document Outputs**: When generating markdown or report assets, keep the tone casual, concise, and to-the-point. Do not include academic padding, n-dashes, or generic AI summaries.
