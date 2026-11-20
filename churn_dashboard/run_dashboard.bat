@echo off
title Telecom Churn Analytics Dashboard
echo ========================================================
echo   Starting Telecom Churn Analytics Dashboard...
echo ========================================================
cd /d "%~dp0"
"C:\Users\visha\.local\bin\python3.15.exe" -m streamlit run app.py
pause

