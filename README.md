# Disaster Hotspot Prediction (Country-level)
Predict the most likely disaster type per country and explore a global hotspot map.

## How to run locally (Streamlit Cloud or similar)
1) Put your dataset at data/nasa_disaster_dataset.csv (or set RAW_CSV_URL env/secret to a raw CSV URL).
2) `pip install -r requirements.txt`
3) `streamlit run app.py`

## Notes
- Model trains quickly on the fly (OneHot + LogisticRegression).
- Map shows top predicted disaster type per country with confidence.
