"""Simulated multi-source dataset for DisasterMind AI.
Each sample has: satellite image patch (16x16), IoT sensor vector (4), weather sequence (6 x 4).
Classes: 0 Normal, 1 Flood, 2 Cyclone, 3 Earthquake, 4 Wildfire.
NOTE: data is synthetic (simulated) - replace with real satellite/IoT/weather data for production.
"""
import numpy as np

CLASSES = ["Normal", "Flood", "Cyclone", "Earthquake", "Wildfire"]
SENSOR_NAMES = ["water_level", "temperature", "seismic", "smoke_index"]
WEATHER_NAMES = ["rainfall", "wind_speed", "humidity", "pressure"]
IMG, T = 16, 6


def make_sample(label, rng, severity=None):
    s = rng.uniform(0.12, 1.0) if severity is None else severity
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    img = 0.3 + 0.22 * rng.standard_normal((IMG, IMG))
    sensor = np.array([0.2, 0.4, 0.05, 0.05]) + 0.12 * rng.standard_normal(4)
    t = np.linspace(0, 1, T)[:, None]
    weather = np.array([0.1, 0.2, 0.5, 0.6]) + 0.10 * rng.standard_normal((T, 4))

    if label == 1:      # Flood: water covers lower part of image, water level + rain rise
        r0 = rng.integers(5, 10)
        img[yy >= r0] += 0.5 * s
        sensor[0] += 0.6 * s
        weather[:, 0] += 0.7 * s * t[:, 0]
    elif label == 2:    # Cyclone: spiral cloud pattern, wind up, pressure down
        cy, cx = rng.uniform(6, 10, 2)
        r = np.hypot(yy - cy, xx - cx); th = np.arctan2(yy - cy, xx - cx)
        img += 0.45 * s * np.sin(3 * th + 0.6 * r)
        sensor[0] += 0.25 * s
        weather[:, 1] += 0.65 * s * t[:, 0]
        weather[:, 3] -= 0.35 * s * t[:, 0]
        weather[:, 0] += 0.35 * s * t[:, 0]
    elif label == 3:    # Earthquake: linear cracks, seismic spike
        for _ in range(rng.integers(1, 3)):
            c = rng.uniform(-4, 4); k = rng.uniform(0.6, 1.4)
            img[np.abs(xx - k * yy - c - 4) < 1.0] += 0.6 * s
        sensor[2] += 0.7 * s
    elif label == 4:    # Wildfire: hot spots, temperature + smoke up, humidity down
        for _ in range(rng.integers(3, 7)):
            y, x = rng.integers(0, IMG - 2, 2)
            img[y:y + 2, x:x + 2] += 0.7 * s
        sensor[1] += 0.45 * s
        sensor[3] += 0.6 * s
        weather[:, 2] -= 0.35 * s * t[:, 0]
        weather[:, 1] += 0.25 * s * t[:, 0]
    else:
        s = rng.uniform(0.0, 0.1)
        if rng.random() < 0.25:        # distractor: heavy rain / hot day without a real disaster
            sensor[rng.integers(0, 4)] += rng.uniform(0.15, 0.4)
            weather[:, rng.integers(0, 4)] += rng.uniform(0.1, 0.3)

    return (np.clip(img, 0, 1).astype("float32"),
            np.clip(sensor, 0, 1).astype("float32"),
            np.clip(weather, 0, 1).astype("float32"),
            label, float(s))


def make_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, n)
    rows = [make_sample(int(l), rng) for l in labels]
    img = np.stack([r[0] for r in rows])[:, None]
    sen = np.stack([r[1] for r in rows])
    wea = np.stack([r[2] for r in rows])
    y = np.array([r[3] for r in rows]); risk = np.array([r[4] for r in rows], dtype="float32")
    return img, sen, wea, y, risk
