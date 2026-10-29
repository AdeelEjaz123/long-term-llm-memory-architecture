@echo off
title Semantic RAG Memory Engine - Shutdown
echo ============================================================
echo   Stopping Semantic RAG Conversational Memory Engine
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Stop any server process listening on port 8000
echo [*] Stopping API server running on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    echo [*] Stopping process (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Stop PostgreSQL pgvector container
echo [*] Stopping PostgreSQL Docker container...
docker compose stop >nul 2>&1
echo [*] Database container stopped.

echo.
echo ============================================================
echo   System stopped successfully.
echo ============================================================
timeout /t 2 >nul
exit /b 0
