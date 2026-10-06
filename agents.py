"""The four cooperating agents of DisasterMind AI."""
import os
import time
import numpy as np
import torch
from data_generator import make_sample, CLASSES, SENSOR_NAMES
from model import DisasterNet
 
REGIONS = {  # all 38 Tamil Nadu districts (approximate district headquarters)
    "Ariyalur": (11.1401, 79.0786), "Chengalpattu": (12.6819, 79.9888), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Cuddalore": (11.7480, 79.7714), "Dharmapuri": (12.1211, 78.1582),
    "Dindigul": (10.3673, 77.9803), "Erode": (11.3410, 77.7172), "Kallakurichi": (11.7384, 78.9639),
    "Kancheepuram": (12.8342, 79.7036), "Kanniyakumari": (8.1833, 77.4119), "Karur": (10.9574, 78.0809),
    "Krishnagiri": (12.5186, 78.2137), "Madurai": (9.9252, 78.1198), "Mayiladuthurai": (11.1018, 79.6550),
    "Nagapattinam": (10.7672, 79.8449), "Namakkal": (11.2194, 78.1677), "Nilgiris": (11.4916, 76.7337),
    "Perambalur": (11.2342, 78.8808), "Pudukkottai": (10.3833, 78.8001), "Ramanathapuram": (9.3639, 78.8395),
    "Ranipet": (12.9224, 79.3326), "Salem": (11.6643, 78.1460), "Sivaganga": (9.8433, 78.4809),
    "Tenkasi": (8.9594, 77.3152), "Thanjavur": (10.7870, 79.1378), "Theni": (10.0104, 77.4768),
    "Thoothukudi": (8.7642, 78.1348), "Tiruchirappalli": (10.7905, 78.7047), "Tirunelveli": (8.7139, 77.7567),
    "Tirupathur": (12.4961, 78.5730), "Tiruppur": (11.1085, 77.3411), "Tiruvallur": (13.1231, 79.9120),
    "Tiruvannamalai": (12.2253, 79.0747), "Tiruvarur": (10.7661, 79.6344), "Vellore": (12.9165, 79.1325),
    "Villupuram": (11.9401, 79.4861), "Virudhunagar": (9.5850, 77.9570)}
 
RESOURCES = {
    "Flood": ["Rescue boats", "NDRF teams", "Relief camps", "Medical units"],
    "Cyclone": ["Evacuation buses", "Cyclone shelters", "Power-restoration crews", "Medical units"],
    "Earthquake": ["Search & rescue teams", "Heavy machinery", "Medical units", "Temporary shelters"],
    "Wildfire": ["Fire tenders", "Forest dept. teams", "Air patrol", "Medical units"]}
 
 
class PerceptionAgent:
    """Senses live data, cleans it (missing values / outliers) and builds model inputs."""
 
    def sense(self, region, label, rng):
        img, sensor, weather, _, _ = make_sample(label, rng)
        sensor = sensor.copy(); sensor_raw = sensor.copy()
        fixed = 0
        for i in range(4):                          # simulate faulty IoT readings
            if rng.random() < 0.08:
                sensor_raw[i] = np.nan
        raw_dict = {n: (None if np.isnan(v) else round(float(v), 3)) for n, v in zip(SENSOR_NAMES, sensor_raw)}
        nan_mask = np.isnan(sensor_raw); fixed = int(nan_mask.sum())
        sensor_clean = np.where(nan_mask, np.array([0.2, 0.4, 0.05, 0.05]), sensor_raw)   # fill with normal baseline
        sensor_clean = np.clip(sensor_clean, 0, 1).astype("float32")
        return {"img": img, "sensor": sensor_clean, "weather": weather,
                "summary": {"region": region, "sources": ["Satellite imagery", "IoT sensors", "Weather feeds"],
                            "sensor_readings": raw_dict, "missing_values_fixed": fixed,
                            "rainfall": round(float(weather[-1, 0]), 2), "wind": round(float(weather[-1, 1]), 2)}}
 
 
class PredictionAgent:
    """Runs the trained DisasterNet model."""
 
    def __init__(self, path="disasternet.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError("disasternet.pt not found - run 'python train.py' first")
        self.model = DisasterNet(); self.model.load_state_dict(torch.load(path, map_location="cpu")); self.model.eval()
 
    def predict(self, data):
        with torch.no_grad():
            lg, risk = self.model(torch.tensor(data["img"])[None, None], torch.tensor(data["sensor"])[None],
                                  torch.tensor(data["weather"])[None])
        p = torch.softmax(lg, 1)[0]; k = int(p.argmax())
        return {"hazard": CLASSES[k], "confidence": round(float(p[k]) * 100, 1), "risk_score": round(float(risk[0]), 2),
                "probabilities": {c: round(float(v) * 100, 1) for c, v in zip(CLASSES, p)}}
 
 
class CoordinationAgent:
    """Turns predictions into alert level, priority and resource allocation."""
 
    def coordinate(self, pred):
        h, r = pred["hazard"], pred["risk_score"]
        if h == "Normal":
            return {"alert_level": "NO ALERT", "priority": 0, "resources": {}}
        level = "LOW" if r < .35 else "MEDIUM" if r < .6 else "HIGH" if r < .8 else "CRITICAL"
        qty = max(1, round(r * 10))
        return {"alert_level": level, "priority": round(r * 100),
                "resources": {name: max(1, qty - i * 2) for i, name in enumerate(RESOURCES[h])}}
 
 
class ResponseAgent:
    """Dispatches teams and prepares public notifications (simulated - no real SMS/FCM is sent)."""
 
    def respond(self, region, pred, plan):
        if plan["alert_level"] == "NO ALERT":
            return {"actions": ["Continue monitoring"], "notification": f"{region}: conditions normal."}
        acts = [f"Dispatch {n} x{q} to {region}" for n, q in plan["resources"].items()]
        msg = f"[{plan['alert_level']}] {pred['hazard']} warning for {region}. Follow official instructions and move to safe areas."
        return {"actions": acts, "notification": msg}
 
 
class DisasterMind:
    """Orchestrates Perception -> Prediction -> Coordination -> Response."""
 
    def __init__(self):
        self.perception, self.prediction = PerceptionAgent(), PredictionAgent()
        self.coordination, self.response = CoordinationAgent(), ResponseAgent()
        self.rng = np.random.default_rng()
 
    def run(self, region, hazard="auto"):
        label = int(self.rng.choice([0, 1, 2, 3, 4], p=[.3, .2, .2, .15, .15])) if hazard == "auto" else CLASSES.index(hazard)
        t0 = time.perf_counter()
        data = self.perception.sense(region, label, self.rng)
        pred = self.prediction.predict(data)
        plan = self.coordination.coordinate(pred)
        resp = self.response.respond(region, pred, plan)
        return {"region": region, "lat": REGIONS[region][0], "lon": REGIONS[region][1],
                "simulated_hazard": CLASSES[label], "perception": data["summary"], "prediction": pred,
                "coordination": plan, "response": resp, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "time": time.strftime("%H:%M:%S")}
