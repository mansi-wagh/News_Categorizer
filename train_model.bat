@echo off
echo ================================================
echo   Training News Categorizer Model
echo ================================================
echo.

cd /d "%~dp0"

echo Step 1: Installing dependencies...
pip install -r requirements.txt

echo.
echo Step 2: Training model on BBC News dataset...
python model_trainer.py

echo.
echo ================================================
echo   Model training complete!
echo ================================================
echo.
echo To run the web app:
echo   python app.py
echo.
pause

