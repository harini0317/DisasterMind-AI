
import math
import statistics
import time
from datetime import datetime, timedelta, timezone
 
import requests
 
from agents import REGIONS, CoordinationAgent, ResponseAgent
 
IST = timezone(timedelta(hours=5, minutes=30))
CACHE_SECONDS = 600
PAST_HOURS = 72                      # hourly history requested (past_days=3)
ALERT_FROM = 0.25                    # below this the district is reported as Normal
_cache = {"t": 0, "data": None}
coord, resp = CoordinationAgent(), ResponseAgent()
 
 
def _get(url, params):
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    j = r.json()
    return j if isinstance(j, list) else [j]      # multi-location -> list, single -> dict
 
 
def _clean(xs):
    return [x for x in xs if x is not None]
 
 
def _clip(x):
    return max(0.0, min(1.0, x))
 
 
def _interp(x, pts):
    """Piecewise-linear score through (value, score) points."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]
 
 
def imd_rain_category(mm):
    for limit, name in [(2.5, "No/very light"), (15.6, "Light"), (64.5, "Moderate"),
                        (115.6, "Heavy"), (204.5, "Very heavy")]:
        if mm < limit:
            return name
    return "Extremely heavy"
 
 
def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))
 
 
# ---------------------------------------------------------------- fetchers (one call for all districts)
BATCH = 20                                          # districts per API call
 
 
def _batches(names):
    for i in range(0, len(names), BATCH):
        yield names[i:i + BATCH]
 
 
def fetch_weather(names):
    out = {}
    for chunk in _batches(names):
        lat = ",".join(str(REGIONS[n][0]) for n in chunk)
        lon = ",".join(str(REGIONS[n][1]) for n in chunk)
        res = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon, "past_days": 3, "forecast_days": 2, "timezone": "Asia/Kolkata",
            "hourly": "precipitation,wind_speed_10m,temperature_2m,relative_humidity_2m,surface_pressure"})
        out.update(zip(chunk, res))
    return out
 
 
def fetch_flood(names):
    out = {}
    for chunk in _batches(names):
        lat = ",".join(str(REGIONS[n][0]) for n in chunk)
        lon = ",".join(str(REGIONS[n][1]) for n in chunk)
        try:
            res = _get("https://flood-api.open-meteo.com/v1/flood", {
                "latitude": lat, "longitude": lon, "daily": "river_discharge",
                "past_days": 30, "forecast_days": 7})
            out.update(zip(chunk, res))
        except Exception:
            pass                                    # flood API down for this batch -> rain-only flood score
    return out
 
 
def fetch_quakes():
    start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=25, params={
            "format": "geojson", "latitude": 10.8, "longitude": 78.3, "maxradiuskm": 800,
            "minmagnitude": 3, "starttime": start, "orderby": "time", "limit": 200})
        r.raise_for_status()
        out = []
        for f in r.json()["features"]:
            lon, lat = f["geometry"]["coordinates"][:2]
            out.append((lat, lon, f["properties"]["mag"] or 0))
        return out
    except Exception:
        return []
 
 
# ---------------------------------------------------------------- scoring
def score_district(name, w, fl, quakes):
    h = w["hourly"]
    now = PAST_HOURS + datetime.now(IST).hour       # index of the current hour (data starts 3 days ago, 00:00)
    past24, past72, nxt = slice(now - 24, now), slice(now - 72, now), slice(now, now + 24)
 
    rain_past = sum(_clean(h["precipitation"][past24]))
    rain_72 = sum(_clean(h["precipitation"][past72]))
    rain_next = sum(_clean(h["precipitation"][nxt]))
    wind_max = max(_clean(h["wind_speed_10m"][nxt]) or [0])
    temp_max = max(_clean(h["temperature_2m"][nxt]) or [0])
    hum_min = min(_clean(h["relative_humidity_2m"][nxt]) or [100])
    p_now = (_clean(h["surface_pressure"][now:now + 1]) or [0])[0]
    p_min = min(_clean(h["surface_pressure"][nxt]) or [p_now])
    p_drop = max(0.0, p_now - p_min)
 
    # river discharge: forecast peak vs. 30-day median baseline
    ratio = 1.0
    if fl:
        d = fl["daily"]["river_discharge"]
        base, peak = _clean(d[:30]), _clean(d[30:])
        if base and peak:
            ratio = max(peak) / max(statistics.median(base), 0.1)
 
    # FLOOD: rain follows IMD categories (heavy 64.5, very heavy 115.6, extremely heavy 204.5 mm/24h).
    # River rise counts fully only when rain supports it or the rise is large (>= 3x baseline).
    rain_peak = max(rain_past, rain_next)
    rs = _interp(rain_peak, [(0, 0), (15.6, .1), (64.5, .4), (115.6, .7), (204.5, 1.0)])
    river = _clip((ratio - 1) / 2) * (1.0 if (rs >= 0.1 or ratio >= 3) else 0.3)
    flood = _clip(0.8 * rs + 0.2 * river)
 
    # CYCLONE: wind follows IMD classes (cyclonic storm 62, severe 89, very severe 118 km/h)
    ws = _interp(wind_max, [(0, 0), (30, .1), (62, .5), (88, .8), (117, 1.0)])
    cyclone = _clip(0.6 * ws + 0.25 * _clip(p_drop / 10) + 0.15 * _interp(rain_next, [(0, 0), (64.5, .4), (115.6, .7), (204.5, 1.0)]))
 
    # WILDFIRE: needs real extreme heat (36-42 C) AND dryness (RH 40-20 %) AND no rain in the last 72 h.
    # An ordinary hot Tamil Nadu day (about 35-37 C) stays below the alert level.
    heat = _clip((temp_max - 36) / 6)
    dry = _clip((40 - hum_min) / 20)
    dry_spell = 1.0 if rain_72 < 5 else 0.3
    wildfire = _clip((0.45 * heat + 0.40 * dry + 0.15 * _clip(wind_max / 40)) * dry_spell)
 
    # EARTHQUAKE: recent activity within 300 km (not a prediction)
    lat0, lon0 = REGIONS[name]
    quake, max_mag, near = 0.0, 0.0, 0
    for qlat, qlon, mag in quakes:
        dist = _haversine(lat0, lon0, qlat, qlon)
        if dist <= 300:
            near += 1
            max_mag = max(max_mag, mag)
            quake = max(quake, _clip((mag - 2.5) / 3.5) * (1 - dist / 300))
 
    risks = {"Flood": flood, "Cyclone": cyclone, "Earthquake": quake, "Wildfire": wildfire}
    summary = {
        "region": name,
        "sources": ["Open-Meteo weather", "GloFAS river discharge", "USGS seismic feed"],
        "sensor_readings": {"imd_rain_category": imd_rain_category(rain_peak),
                            "rain_past_24h_mm": round(rain_past, 1), "rain_past_72h_mm": round(rain_72, 1),
                            "rain_next_24h_mm": round(rain_next, 1), "river_discharge_ratio": round(ratio, 2),
                            "max_wind_kmh": round(wind_max, 1), "max_temp_c": round(temp_max, 1),
                            "min_humidity_pct": round(hum_min), "pressure_drop_hpa": round(p_drop, 1),
                            "quakes_7d_within_300km": near, "max_quake_mag": max_mag},
        "missing_values_fixed": 0,
        "rainfall": round(rain_past, 2), "wind": round(wind_max, 2)}
    return risks, summary
 
 
def assess(name, w, fl, quakes):
    t0 = time.perf_counter()
    risks, summary = score_district(name, w, fl, quakes)
    top = max(risks, key=risks.get)
    hazard = top if risks[top] >= ALERT_FROM else "Normal"
    pred = {"hazard": hazard,
            "confidence": None,                       # rule-based: no model confidence to report
            "risk_score": round(risks[top], 2),
            "probabilities": {k: round(v * 100, 1) for k, v in risks.items()}}   # per-hazard risk %
    plan = coord.coordinate(pred)
    rsp = resp.respond(name, pred, plan)
    return {"region": name, "lat": REGIONS[name][0], "lon": REGIONS[name][1],
            "simulated_hazard": "LIVE DATA", "perception": summary, "prediction": pred,
            "coordination": plan, "response": rsp,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "time": time.strftime("%H:%M:%S")}
 
 
def run_live_all(force=False):
    """Assess every district. Cached for 10 minutes to stay friendly to the free APIs."""
    if not force and _cache["data"] and time.time() - _cache["t"] < CACHE_SECONDS:
        return _cache["data"]
    names = list(REGIONS)
    weather, flood, quakes = fetch_weather(names), fetch_flood(names), fetch_quakes()
    out = [assess(n, weather[n], flood.get(n), quakes) for n in names]
    out.sort(key=lambda e: e["prediction"]["risk_score"], reverse=True)   # highest risk first
    _cache.update(t=time.time(), data=out)
    return out
 
 
if __name__ == "__main__":
    for e in run_live_all(force=True):
        p, c = e["prediction"], e["coordination"]
        print(f"{e['region']:<13} {p['hazard']:<10} risk={p['risk_score']:<5} alert={c['alert_level']}")
