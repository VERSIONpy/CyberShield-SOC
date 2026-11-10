@echo off
echo ============================================================
echo   CyberShield - Network Threat Detection ^& Analysis
echo   Starting server at http://localhost:8000
echo ============================================================
echo.
cd /d "%~dp0"
python server.py
pause
