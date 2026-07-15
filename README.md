# CRIS Indian Railways Wagon OCR and Stencil Audit Pipeline

This codebase runs the production-grade Wagon OCR and safety stencil audit pipeline designed for Indian Railways (CRIS). It localizes and extracts wagon details like wagon number, type, owner, zone, and tare details from high-resolution raw images. It also checks if safety stencils are present.

## Project Structure

*   cris_ocr_pipeline/ - Backend files
    *   scripts/ocr_pipeline.py - Core OCR pipeline with 5-tier fallback logic
    *   scripts/ocr_service.py - GPU PaddleOCR microservice (runs on port 8001)
    *   scripts/server.py - FastAPI server connecting frontend to backend (runs on port 8000)
    *   models/ - Custom YOLO models (best.pt for parameters and stencil_best.pt for stencils)
    *   config/pipeline_config.json - Master registry config for checksum validation
*   frontend/ - Vite and React web app dashboard (runs on port 5173)
*   yolov11m.ipynb - Kaggle notebook used to train the custom 5-class localization model
*   raw_images/ - Empty folder for raw test images
*   .agents/ - Context and rules for AI coding assistants

## Quick Setup and Dependencies

Follow these steps to get everything running on your local machine:

1. Install Python packages:
   pip install -r requirements.txt

2. Configure Python Path:
   We have added a .env file and .vscode/settings.json to configure PYTHONPATH automatically in VS Code. If you are using a standard command prompt outside VS Code, set it manually before running the scripts:
   - Windows PowerShell: $env:PYTHONPATH=".;cris_ocr_pipeline/scripts"
   - Windows CMD: set PYTHONPATH=.;cris_ocr_pipeline/scripts
   - Linux/Mac Bash: export PYTHONPATH=".:cris_ocr_pipeline/scripts"

3. Put Test Images:
   The massive 21k raw images folder has been cleared to keep the repository size small. Download the images from Google Drive and paste them directly into the "raw_images" folder in the project root.

## Running the Application

To start the system, double-click the batch files at the root of the project:

1. Run setup.bat first:
   This will install all required Python modules and frontend node packages with a single click.

2. Run run_project.bat:
   This will start the PaddleOCR service on port 8001, boot the FastAPI backend on port 8000, and launch the React frontend dashboard on port 5173. The dashboard will open in your browser automatically.

## Collaboration Workflow

To contribute code:
1. Always create a new branch for your task (e.g. git checkout -b feature/your-task).
2. Commit and push your changes to your branch, then create a Pull Request on GitHub.
3. Keep raw images only in your local folder. Do not add them to git.
