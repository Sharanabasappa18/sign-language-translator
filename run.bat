@echo off
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q
if not exist ml_model\gesture_classifier.pkl (
  python -m ml_model.generate_synthetic --train
)
python scripts\download_model.py
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
