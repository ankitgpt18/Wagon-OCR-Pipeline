@echo off
echo ============================================================
echo   CRIS Wagon OCR Project Setup
echo ============================================================
echo.
echo 1. Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python dependencies installation failed!
    pause
    exit /b %errorlevel%
)

echo.
echo 2. Installing React/Vite Frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Frontend dependencies installation failed!
    pause
    exit /b %errorlevel%
)
cd ..

echo.
echo ============================================================
echo   Setup completed successfully!
echo   To start the application, double-click 'run_project.bat'
echo ============================================================
pause
