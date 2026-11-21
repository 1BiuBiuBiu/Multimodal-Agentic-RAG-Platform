@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Enterprise Agentic RAG Platform - Stopping API and UI ...
echo ============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Stop failed. Please check the error message above.
    pause
    exit /b 1
)
echo.
echo [OK] All services stopped.
timeout /t 2 /nobreak >nul
