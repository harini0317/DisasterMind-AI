"""One-command launcher:  python main.py
Trains the model automatically on first run, then starts the dashboard at http://127.0.0.1:8000"""
import os
import threading
import webbrowser
import uvicorn

if not os.path.exists("disasternet.pt"):
    print("First run: training DisasterNet (takes about a minute)...")
    import train
    train.main()

threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="warning")
