@echo off
title Semantic RAG Memory Engine - Startup
echo ============================================================
echo   Starting Semantic RAG Conversational Memory Engine
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Verify Docker is running
echo [*] Checking Docker status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

:: 2. Start PostgreSQL with pgvector container
echo [*] Starting PostgreSQL pgvector container...
docker compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Docker database container.
    echo.
    pause
    exit /b 1
)

:: 3. Check / Activate Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Virtual environment not found. Initializing .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [*] Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: 4. Launch browser after 2 seconds in background
echo.
echo [*] Database running on port 5432
echo [*] Launching API server on http://localhost:8000
echo [*] Opening Web Console in default browser...
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

:: 5. Run Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
