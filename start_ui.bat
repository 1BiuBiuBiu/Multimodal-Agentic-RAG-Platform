@echo off
set PYTHONPATH=.;.venv\Lib\site-packages
set PYTHONUTF8=1
python -m streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
