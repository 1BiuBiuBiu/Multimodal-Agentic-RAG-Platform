@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Enterprise Agentic RAG Platform - Starting API and UI ...
echo ============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed. Please check the error message above.
    pause
    exit /b 1
)
echo.
echo [OK] API and UI are running, and the browser has been opened.
timeout /t 3 /nobreak >nul
