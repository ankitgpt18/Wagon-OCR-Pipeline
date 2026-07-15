@echo off
echo ============================================================
echo   CRIS Wagon OCR Project Launcher
echo ============================================================
echo.

echo 1. Starting PaddleOCR GPU Microservice on port 8001...
start "PaddleOCR Service (Port 8001)" cmd /k "set FLAGS_enable_pir_api=0&& set FLAGS_use_mkldnn=0&& set FLAGS_enable_pir_in_executor=0&& python cris_ocr_pipeline/scripts/ocr_service.py"
echo Waiting 8 seconds for OCR models to pre-warm...
timeout /t 8 >nul

echo 2. Starting FastAPI Backend Server on port 8000...
start "FastAPI Backend (Port 8000)" cmd /k "python start_server.py"
echo Waiting 3 seconds for server initialization...
timeout /t 3 >nul

echo 3. Starting React Frontend (Vite) on port 5173...
start "React Frontend Dashboard (Port 5173)" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo   All services launched!
echo   - Web Dashboard: http://localhost:5173
echo   - Backend Server: http://127.0.0.1:8000
echo ============================================================
echo.
pause
