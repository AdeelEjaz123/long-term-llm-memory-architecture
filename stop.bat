@echo off
title Semantic RAG Memory Engine - Shutdown
echo ============================================================
echo   Stopping Semantic RAG Conversational Memory Engine
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Force kill processes running uvicorn or listening on port 8000
echo [*] Terminating API server on port 8000...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*app.main:app*' -or $_.CommandLine -like '*uvicorn*app.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

:: 2. Terminate uvicorn / console windows if still open
taskkill /F /FI "WINDOWTITLE eq Semantic RAG Memory Engine - Startup*" /T >nul 2>&1
taskkill /F /IM uvicorn.exe /T >nul 2>&1

:: 3. Stop PostgreSQL Docker container
echo [*] Stopping PostgreSQL pgvector container...
docker stop semantic_rag_postgres >nul 2>&1
docker compose stop >nul 2>&1

:: 4. Verify status
echo.
powershell -NoProfile -Command "$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if ($listen) { Write-Host ' [!] Port 8000 is still active.' -ForegroundColor Yellow } else { Write-Host ' [OK] API server successfully stopped.' -ForegroundColor Green }"
powershell -NoProfile -Command "$running = docker ps -q --filter 'name=semantic_rag_postgres'; if ($running) { Write-Host ' [!] Database container still running.' -ForegroundColor Yellow } else { Write-Host ' [OK] Database container stopped.' -ForegroundColor Green }"

echo.
echo ============================================================
echo   System stopped successfully.
echo ============================================================
echo.
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
exit /b 0
