"""
OmniPulse CLV & Churn Intelligence Engine - One-Click Launcher
Starts the FastAPI application and launches the interactive dashboard.
"""

import uvicorn
import webbrowser
import os
import sys

def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 70)
    print("   OMNIPULSE | CUSTOMER LIFETIME VALUE & CHURN INTELLIGENCE ENGINE")
    print("=" * 70)
    print(f"[*] Starting local server at {url}")
    print("[*] Interactive dashboard will be available at:")
    print(f"    --> {url}")
    print("=" * 70)
    
    # Try to open the browser automatically after 1 second
    try:
        import threading
        import time
        def open_browser():
            time.sleep(1.5)
            print(f"[OmniPulse] Opening interactive dashboard: {url}")
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()
    except Exception as e:
        print(f"Note: Could not automatically open browser: {e}")
        
    uvicorn.run("backend.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
