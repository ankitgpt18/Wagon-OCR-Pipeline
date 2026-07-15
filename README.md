# CRIS Indian Railways Wagon OCR & Stencil Audit Pipeline

This is the production-grade Wagon OCR and safety stencil audit pipeline designed for Indian Railways (CRIS). It localizes and extracts wagon parameters (wagon number, type, owner, zone, tare details) from high-resolution raw images and classifies safety stencil presence.

---

## 📂 Project Structure

*   `cris_ocr_pipeline/` - Production OCR pipeline backend:
    *   `scripts/ocr_pipeline.py` - Core OCR execution pipeline with 5-tier fallback.
    *   `scripts/ocr_service.py` - Pre-warmed GPU PaddleOCR API microservice (runs on port 8001).
    *   `scripts/server.py` - FastAPI backend server connecting frontend to backend OCR (runs on port 8000).
    *   `models/` - Bounding box YOLO detection models (`best.pt` for parameters, `stencil_best.pt` for safety stencils).
    *   `config/pipeline_config.json` - Target matrices and master registry for cross-validation.
*   `frontend/` - Vite + React frontend dashboard web app (runs on port 5173).
*   `yolov11m.ipynb` - Kaggle Jupyter Notebook used to train the custom 5-class localization model.
*   `raw_images/` - Directory for original raw wagon images.
*   `.agents/` - Custom context and operating rules for AI coding agents.

---

## ⚙️ Quick Setup & Dependencies

To make sure that any developer or AI agent can run the code without path or import errors, follow these steps:

### 1. Python Environment Setup
Install the Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Python Path
To prevent `ModuleNotFoundError` when importing files from subdirectory trees:
*   We have placed a `.env` file at the root containing the portable relative `PYTHONPATH`.
*   A `.vscode/settings.json` file is configured so VS Code terminals and PyLance IntelliSense load the relative paths automatically.
*   **If running outside VS Code (standard terminal)**, run this command from the project root:
    *   **PowerShell**: `$env:PYTHONPATH=".;cris_ocr_pipeline/scripts"`
    *   **CMD**: `set PYTHONPATH=.;cris_ocr_pipeline/scripts`
    *   **Bash**: `export PYTHONPATH=".:cris_ocr_pipeline/scripts"`

### 3. Image Dataset
To keep the zip archive clean and small, the massive 21k raw images folder has been cleared. Place any raw wagon images you want to test in the `raw_images/` directory in the root of this workspace.

---

## 🚀 How to Run the System

To start the OCR backend and web interface, execute these commands in order:

### Step 1: Start the OCR GPU Microservice
This microservice pre-warms the GPU text recognition models to prevent CUDA memory thrashing.
```bash
python cris_ocr_pipeline/scripts/ocr_service.py
```
*   Runs on `http://127.0.0.1:8001`
*   *Note*: The service automatically disables oneDNN (MKLDNN) to prevent library thread crashes on Windows environments.

### Step 2: Start the FastAPI Backend Server
```bash
python start_server.py
```
*   Runs on `http://127.0.0.1:8000`
*   Handles upload logic, runs YOLO localization, calls the OCR service, executes Mod-10 checksum validation, and looks up registries.

### Step 3: Run the Frontend React App
Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
npm install
npm run dev
```
*   Runs on `http://localhost:5173`

---

## 🤖 AI Agent Integration
If you or your teammates use AI coding agents (such as Gemini, Claude, or other assistants), the project-scoped rules are rootly integrated. The `.agents/AGENTS.md` file contains complete context, architectural flows, and database models. When an agent opens this directory, it will automatically discover these guidelines and align with the codebase design system.

---

## 📦 Zip & Share Guidelines
When zipping this project folder to share with teammates:
1.  Keep `raw_images/` empty (they can drop their test images there).
2.  Delete `node_modules` inside the `frontend/` directory (it will be re-created via `npm install`).
3.  Exclude any `.git` folder or python `__pycache__` directories.
