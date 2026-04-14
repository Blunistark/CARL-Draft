@echo off
echo Recreating virtual environment for Tacview...
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
