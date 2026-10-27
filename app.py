"""Smart Agriculture Decision Support System - click-on-map version.
Click a point on the map -> climate + soil are fetched automatically ->
ML predicts yield, intercrops, rotation and nutrient recovery. No manual entry."""
import os, sys
import numpy as np, requests
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor

BASE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))  # works inside the .exe too
app = Flask(__name__, template_folder=os.path.join(BASE, 'templates'))

# name: (temp C, annual rain mm, soil pH, potential t/ha, family)  -- agronomic ranges, cite FAO/ICAR in report
CROPS = {
 "Rice": ((20,35),(1000,2500),(5.0,7.0),4.0,"cereal"),
 "Wheat": ((10,25),(400,900),(6.0,7.5),3.5,"cereal"),
 "Maize": ((18,32),(500,1000),(5.5,7.5),5.0,"cereal"),
 "Ragi": ((20,32),(500,1000),(5.0,7.5),2.0,"cereal"),
 "Groundnut": ((22,32),(500,1000),(5.5,7.0),1.8,"legume"),
 "Chickpea": ((15,28),(400,800),(6.0,8.0),1.5,"legume"),
 "Soybean": ((20,30),(600,1200),(6.0,7.0),2.2,"legume"),
 "Cotton": ((21,35),(600,1200),(6.0,8.0),2.0,"fibre"),
 "Tomato": ((18,30),(600,1100),(6.0,7.0),25.0,"vegetable"),
}
INTERCROP = {"cereal": ["Soybean","Groundnut","Chickpea"], "legume": ["Maize","Ragi"],
             "fibre": ["Soybean","Groundnut"], "vegetable": ["Groundnut","Chickpea"]}
ORGANIC = {"Compost": (10,5,8), "Farmyard manure": (5,2.5,5)}  # kg N,P2O5,K2O per tonne (typical values)
APPLIED_T_HA = 5

def fit(x, lo, hi):  # 1 at centre of range, 0 at/after edges
    mid, half = (lo+hi)/2, (hi-lo)/2
    return max(0.0, 1-abs(x-mid)/(half*1.6))

def train():
    rng, models = np.random.default_rng(7), {}
    for c,((t0,t1),(r0,r1),(p0,p1),pot,_) in CROPS.items():
        X = np.column_stack([rng.uniform(t0-12,t1+12,800), rng.uniform(r0*.3,r1*1.6,800), rng.uniform(4.5,8.8,800)])
        y = np.array([pot*(fit(t,t0,t1)*fit(r,r0,r1)*fit(p,p0,p1))**(1/3)*rng.uniform(.9,1.1) for t,r,p in X])
        models[c] = RandomForestRegressor(100, random_state=7).fit(X,y)
    return models
MODELS = train()

def environment(lat, lon):
    """Fetch last-year climate (Open-Meteo) and topsoil pH (SoilGrids). Falls back to defaults if offline."""
    env = {"temp":25.0,"rain":800.0,"ph":6.5,"source":"defaults (offline)"}
    try:
        w = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=20, params={
            "latitude":lat,"longitude":lon,"start_date":"2024-01-01","end_date":"2024-12-31",
            "daily":"temperature_2m_mean,precipitation_sum","timezone":"auto"}).json()["daily"]
        env["temp"] = round(float(np.nanmean(w["temperature_2m_mean"])),1)
        env["rain"] = round(float(np.nansum(w["precipitation_sum"])),0)
        env["source"] = "Open-Meteo climate"
        s = requests.get("https://rest.isric.org/soilgrids/v2.0/properties/query", timeout=25, params={
            "lat":lat,"lon":lon,"property":"phh2o","depth":"0-5cm","value":"mean"}).json()
        env["ph"] = round(s["properties"]["layers"][0]["depths"][0]["values"]["mean"]/10,1)
        env["source"] += " + SoilGrids soil"
    except Exception:
        pass
    return env

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/analyze")
def analyze():
    lat, lon = float(request.args["lat"]), float(request.args["lon"])
    env = environment(lat, lon); x = [[env["temp"], env["rain"], env["ph"]]]
    ranked = sorted(((c, float(m.predict(x)[0])) for c,m in MODELS.items()), key=lambda t:-t[1])
    yields = [{"crop":c,"yield_t_ha":round(y,2),"suitability":round(100*y/CROPS[c][3])} for c,y in ranked]
    best = yields[0]["crop"]; fam = CROPS[best][4]
    inter = [c for c in INTERCROP[fam] if c in MODELS][:2]
    nxt = [c for c in (["Soybean","Groundnut","Chickpea"] if fam!="legume" else ["Maize","Wheat","Rice"])
           if next(y["suitability"] for y in yields if y["crop"]==c) >= 35][:2] or ["Fallow + green manure"]
    nutr = {k: {"N":v[0]*APPLIED_T_HA,"P2O5":v[1]*APPLIED_T_HA,"K2O":v[2]*APPLIED_T_HA,
                "urea_saved_kg":round(v[0]*APPLIED_T_HA/0.46)} for k,v in ORGANIC.items()}
    return jsonify(location={"lat":lat,"lon":lon}, environment=env, yields=yields[:5], best=best,
                   intercrops=inter, rotation=[best]+nxt, nutrients=nutr, applied_t_ha=APPLIED_T_HA)

if __name__ == "__main__":
    app.run(debug=True)
