@echo off
setlocal

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       GST Reconciliation System — Production Mode        ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: Check for .env file
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo Please copy .env.example to .env and configure your settings.
    pause
    exit /b 1
)

:: Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] No .venv found, using system Python.
)

:: 1. Start Backend
echo.
echo [1/2] Starting Backend on port 8001...
start "GST Backend" /B python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --log-level info

:: Wait for backend to be ready
echo [INFO] Waiting for backend to start...
timeout /t 5 /nobreak >nul

:: Quick health check
curl -s http://localhost:8001/ >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [OK] Backend is running at http://localhost:8001
) else (
    echo [WARNING] Backend may still be starting up...
)

:: 2. Start Frontend
echo.
echo [2/2] Starting Dashboard on port 8502...
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  Dashboard: http://localhost:8502                         ║
echo ║  API Docs:  http://localhost:8001/docs                    ║
echo ║  Press Ctrl+C to stop                                     ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
streamlit run frontend/dashboard.py --server.port 8502 --server.address 0.0.0.0

endlocal
