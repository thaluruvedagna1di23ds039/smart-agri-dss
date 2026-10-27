# Smart Agriculture Decision Support System (map-click version)
Run: `pip install -r requirements.txt` then `python app.py` and open http://127.0.0.1:5000
Click the map -> temperature, rainfall (Open-Meteo) and soil pH (SoilGrids) are fetched automatically.
Random Forest models predict crop yield; rules give intercrops, rotation and organic nutrient recovery.
Originality: all code is original. Cite OpenStreetMap, Leaflet (BSD-2), Open-Meteo (CC BY 4.0), SoilGrids (CC BY 4.0), scikit-learn (BSD).
Upgrade for the viva: replace the synthetic training data in train() with a real dataset (e.g. ICRISAT/Kaggle crop data) and cite it.
