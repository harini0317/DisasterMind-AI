"""Console demo (no browser needed): python run_demo.py"""
from agents import DisasterMind, REGIONS

dm = DisasterMind()
for region, hazard in [("Karur", "Flood"), ("Chennai", "Cyclone"), ("Nilgiris", "Wildfire"), ("Madurai", "Earthquake"), ("Salem", "Normal")]:
    r = dm.run(region, hazard)
    print("=" * 70)
    print(f"REGION: {region}   (simulated event: {hazard})")
    print(f"1. Perception  : sources={len(r['perception']['sources'])}, missing values fixed={r['perception']['missing_values_fixed']}")
    print(f"2. Prediction  : {r['prediction']['hazard']} ({r['prediction']['confidence']}% confidence), risk score {r['prediction']['risk_score']}")
    print(f"3. Coordination: alert={r['coordination']['alert_level']}, priority={r['coordination']['priority']}")
    for a in r["response"]["actions"]:
        print(f"4. Response    : {a}")
    print(f"   Notification: {r['response']['notification']}")
    print(f"   Pipeline latency: {r['latency_ms']} ms")
