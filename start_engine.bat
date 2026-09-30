@echo off
title OmniPulse CLV & Churn Intelligence Engine
echo ======================================================================
echo    OMNIPULSE | CLV & CHURN INTELLIGENCE ENGINE
echo ======================================================================
echo [*] Starting OmniPulse Engine and Opening Interactive Dashboard...
start "" "http://127.0.0.1:8000"
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
pause
