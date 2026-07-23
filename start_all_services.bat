@echo off
TITLE CRIS Wagon OCR Pipeline - 1-Click Automated Service Launcher
echo ======================================================================
echo   CRIS WAGON OCR PIPELINE - AUTOMATED 1-CLICK DEMO LAUNCHER
echo ======================================================================
echo.
echo [1/4] Starting PaddleOCR GPU Microservice on Port 8001...
start "1. PaddleOCR Service (Port 8001)" cmd /k "python cris_ocr_pipeline/scripts/ocr_service.py"

echo [2/4] Starting Qwen2.5-VL VLM Microservice on Port 8002...
start "2. Qwen VLM Service (Port 8002)" cmd /k "python cris_ocr_pipeline/scripts/vlm_ocr_service.py"

echo [3/4] Waiting 3 seconds for microservices to initialize...
timeout /t 3 /nobreak >nul

echo [4/4] Starting FastAPI Core Server on Port 8000...
start "3. FastAPI Core Server (Port 8000)" cmd /k "python start_server.py"

echo [5/5] Launching React UI Dashboard on Port 5173...
start "4. React UI Dashboard (Port 5173)" cmd /k "cd frontend && npm run dev"

echo.
echo ======================================================================
echo  SUCCESS: All 4 microservices launched in separate windows!
echo  Open your browser to: http://localhost:5173
echo ======================================================================
pause
