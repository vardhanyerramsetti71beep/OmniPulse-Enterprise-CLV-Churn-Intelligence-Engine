Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   OMNIPULSE | CLV & CHURN INTELLIGENCE ENGINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] Launching browser to http://127.0.0.1:8000..." -ForegroundColor Yellow
Start-Process "http://127.0.0.1:8000"
Write-Host "[*] Starting Uvicorn API server on port 8000..." -ForegroundColor Green
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
