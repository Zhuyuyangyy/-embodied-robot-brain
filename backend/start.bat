@echo off
echo ====================================
echo EmbodiedRobotBrain Backend
echo ====================================

cd /d "%~dp0"

echo.
echo [1/3] Checking Python...
python --version

echo [2/3] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [3/3] Starting server on port 8013...
echo.
echo API: http://localhost:8013
echo Docs: http://localhost:8013/docs
echo.

python app.py

pause
