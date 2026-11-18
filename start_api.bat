@echo off
set PYTHONPATH=.;.venv\Lib\site-packages
set PYTHONUTF8=1
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
