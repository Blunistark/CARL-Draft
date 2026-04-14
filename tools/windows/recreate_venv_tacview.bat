@echo off
REM Recreates the virtual environment for use with the Tacview flight-data
REM visualisation tool (https://www.tacview.net/). Run this if the Tacview
REM integration dependencies become out of sync with the main environment.
echo Recreating virtual environment for Tacview...
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
