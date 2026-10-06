# DisasterMind AI - Multi-Agent Disaster Prediction & Emergency Response System

Working prototype for the project review. Uses PyTorch (DisasterNet), 4 cooperating agents,
FastAPI backend and a Leaflet.js map dashboard.

## How to run (Windows / Mac / Linux)

1. Install Python 3.9+ (tick "Add Python to PATH" on Windows).
2. Open a terminal / Command Prompt inside this folder and run:

       pip install -r requirements.txt
       python main.py

3. The dashboard opens at http://127.0.0.1:8000 (open it manually if it does not).
   The trained model is already included. If disasternet.pt is missing, it trains itself first (about 1 minute).

## Demo flow for the review
1. Choose a district (e.g. Karur) and a hazard (e.g. Flood) -> click "Run Agents".
2. Show the 4 agents lighting up one by one: Perception -> Prediction -> Coordination -> Response.
3. The map marker changes colour by alert level; the Event Log fills up.
4. Click "Scan All Districts" to show multi-district, multi-hazard monitoring.
5. Show the model test results card (accuracy / precision / recall / F1).

No browser? Run "python run_demo.py" for a console demo.

## Files
- data_generator.py : simulated satellite + IoT + weather data (5 classes)
- model.py          : DisasterNet (CNN + Dense + LSTM -> fusion -> hazard class + risk score)
- train.py          : Phase 2 training, validation and testing (python train.py)
- agents.py         : Perception, Prediction, Coordination, Response agents
- app.py / main.py  : FastAPI server and one-command launcher
- static/index.html : dashboard (Leaflet map needs internet for map tiles)

## Important
Data is SIMULATED (synthetic) because real satellite/IoT feeds are not connected. Alerts and dispatches are
simulated on screen; no real SMS/Firebase message is sent. Say this clearly in the review.
