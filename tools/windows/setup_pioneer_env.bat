@echo off
echo Setting up Pioneer environment...
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
echo Environment setup complete.
