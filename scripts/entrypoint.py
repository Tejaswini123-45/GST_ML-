"""
Production Docker entrypoint for GST Reconciliation.
Coordinates and runs both FastAPI (backend) and Streamlit (frontend)
as concurrent subprocesses, forwarding signals and exiting if either fails.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

def main():
    # 1. Configuration
    import platform
    port = os.getenv("PORT", "8001")
    # Multiple workers are not cleanly supported on Windows by Uvicorn (causes WinError 10022)
    workers = "1" if platform.system() == "Windows" else os.getenv("WORKERS", "4")
    
    # 2. Command definitions
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0",
        "--port", port,
    ]
    if workers != "1":
        backend_cmd.extend(["--workers", workers])
    
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
        "--server.port", "8502",
        "--server.address", "0.0.0.0"
    ]
    
    print("[ENTRYPOINT] Starting backend process...", flush=True)
    backend_proc = subprocess.Popen(backend_cmd)
    
    # Wait briefly for the backend to start up
    time.sleep(3)
    
    print("[ENTRYPOINT] Starting frontend process...", flush=True)
    frontend_proc = subprocess.Popen(frontend_cmd)
    
    procs = [backend_proc, frontend_proc]
    
    def signal_handler(signum, frame):
        print(f"\n[ENTRYPOINT] Received signal {signum}, terminating child processes...", flush=True)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        
        # Wait up to 5s for children to shut down cleanly
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 3. Monitor loop
    try:
        while True:
            # Check if any process has exited
            for p in procs:
                exit_code = p.poll()
                if exit_code is not None:
                    name = "Backend" if p == backend_proc else "Frontend"
                    print(f"[ENTRYPOINT] ERROR: {name} process exited with code {exit_code}", flush=True)
                    # Trigger shutdown of the other process
                    signal_handler(signal.SIGTERM, None)
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
