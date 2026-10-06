"""FastAPI backend + dashboard.  Run: python main.py"""
import json
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agents import DisasterMind, REGIONS
from data_generator import CLASSES
from live_agents import run_live_all
 
app = FastAPI(title="DisasterMind AI")
dm = DisasterMind()
events = []
 
 
class Sim(BaseModel):
    region: str = "Karur"
    hazard: str = "auto"
 
 
@app.get("/")
def index():
    return FileResponse("static/index.html")
 
 
@app.get("/api/info")
def info():
    return {"regions": {k: {"lat": v[0], "lon": v[1]} for k, v in REGIONS.items()}, "hazards": CLASSES}
 
 
@app.post("/api/simulate")
def simulate(s: Sim):
    if s.region not in REGIONS or (s.hazard != "auto" and s.hazard not in CLASSES):
        return JSONResponse({"error": "invalid region or hazard"}, status_code=400)
    r = dm.run(s.region, s.hazard); events.insert(0, r)
    del events[200:]
    return r
 
 
@app.post("/api/simulate_all")
def simulate_all():
    out = [dm.run(region, "auto") for region in REGIONS]
    for r in out:
        events.insert(0, r)
    del events[200:]
    return out
 
 
@app.api_route("/api/live_all", methods=["GET", "POST"])
def live_all(refresh: bool = False):
    """Real-time risk for every district (Open-Meteo + GloFAS + USGS). Add ?refresh=true to skip the 10-min cache."""
    try:
        out = run_live_all(force=refresh)
    except Exception as e:
        return JSONResponse({"error": f"live data fetch failed: {e}"}, status_code=502)
    for r in out:
        events.insert(0, r)
    del events[200:]
    return out
 
 
@app.get("/api/events")
def get_events():
    return events
 
 
@app.get("/api/metrics")
def metrics():
    return json.load(open("metrics.json")) if os.path.exists("metrics.json") else {}
 
 
app.mount("/static", StaticFiles(directory="static"), name="static")
