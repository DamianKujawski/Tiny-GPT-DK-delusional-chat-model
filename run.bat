@echo off
setlocal
cd /d "%~dp0"
title Tiny DK Chat

:: Setup virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto error

:: Verify dependencies
echo Checking required libraries...
python -c "import torch, tokenizers, numpy" >nul 2>&1
if errorlevel 1 (
    python -m pip install torch tokenizers numpy
    if errorlevel 1 goto error
)

:: Run app
echo Starting Tiny DK Chat...
python run.py
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo An error occurred.
pause
exit /b 1
