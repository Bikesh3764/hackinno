@echo off
REM One-click local launcher for MARS-09 Odyssey (Windows).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
echo Open http://localhost:8000 in your browser
.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --port 8000
