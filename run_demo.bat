@echo off
rem Starts the super-resolution web demo and opens it in the browser once it has loaded.
cd /d "%~dp0"
start "" /min cmd /c "timeout /t 15 >nul & start http://127.0.0.1:7860"
".venv\Scripts\python.exe" app.py
