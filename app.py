import os
import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pycountry

st.set_page_config(page_title="Disaster Hotspot Prediction", layout="wide")
st.title("Disaster Hotspot Prediction")
st.caption("Colab + GitHub + Streamlit (no local setup)")

# ----------------------------
# 1) Data loading
# ----------------------------
DEFAULT_LOCAL_PATH = "data/nasa_disaster_dataset.csv"
raw_url_hint = "https://raw.githubusercontent.com/<user>/<repo>/<branch>/data/nasa_disaster_dataset.csv"

st.sidebar.header("Data Source")
use_repo_file = st.sidebar.radio(
    "Choose input",
    ["Use file in this repo", "Use raw CSV URL"],
    index=0
)

raw_csv_url = st.sidebar.text_input("RAW_CSV_URL (optional)", value="")

def load_csv():
    if use_repo_file == "Use file in this repo" and os.path.exists(DEFAULT_LOCAL_PATH):
        return pd.read_csv(DEFAULT_LOCAL_PATH)
    if raw_csv_url.strip():
        r = requests.get(raw_csv_url.strip(), timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    if os.path.exists(DEFAULT_LOCAL_PATH):
        return pd.read_csv(DEFAULT_LOCAL_PATH)
    st.error("CSV not found. Upload it to data/nasa_disaster_dataset.csv or provide a RAW_CSV_URL like:\n" + raw_url_hint)
    st.stop()

df = load_csv()

# ----------------------------
# 2) Quick sanity checks
# ----------------------------
# Expected columns based on your dataset exploration
expected_cols = ["id", "country", "geolocation", "level", "adm1", "location", "disastertype", "continent"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.warning(f"Missing expected columns: {missing}. Proceeding with available ones.")
# Drop obvious index-ish columns if present
for c in ["Unnamed: 0", "index"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# Basic clean and keep relevant subset
keep_cols = [c for c in ["country", "adm1", "continent", "level", "disastertype"] if c in df.columns]
df = df[keep_cols].copy()

# Remove rows without target
df = df.dropna(subset=["disastertype"])
df["country"] = df["country"].fillna("Unknown")
if "adm1" in df.columns:
    df["adm1"] = df["adm1"].fillna("Unknown")
if "continent" in df.columns:
    df["continent"] = df["continent"].fillna("Unknown")
if "level" in df.columns:
    # level sometimes numeric-like; coerce to string so OHE treats uniformly
    df["level"] = df["level"].astype(str).fillna("Unknown")

st.subheader("Dataset Snapshot")
st.write(df.head(10))
st.write(f"Rows: {len(df):,}")

# ----------------------------
# 3) Train/test split & pipeline
# ----------------------------
target = "disastertype"
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].copy()
y = df[target].astype(str)

cat_cols = feature_cols  # all are categorical-like now
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="drop"
)

model = LogisticRegression(max_iter=1000, n_jobs=None)

clf = Pipeline(steps=[("prep", preprocess), ("clf", model)])

# Quick downsample option for speed on small instances
max_rows = st.sidebar.slider("Train on up to N rows (for speed)", min_value=2000, max_value=min(40000, len(df)), value=min(15000, len(df)), step=1000)
if len(df) > max_rows:
    df_sample = df.sample(n=max_rows, random_state=42)
    X = df_sample[feature_cols]
    y = df_sample[target].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with st.expander("Model Performance", expanded=True):
    st.write(f"Accuracy: {acc:.3f}")
    cr = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    st.text(cr)

# ----------------------------
# 4) Country-level prediction UI
# ----------------------------
st.subheader("Predict Most Likely Disaster Type for a Country/Region")

countries = sorted(df["country"].dropna().unique().tolist())
chosen_country = st.selectbox("Country", countries, index=0)

adm1_list = []
if "adm1" in df.columns:
    adm1_list = sorted(df.loc[df["country"] == chosen_country, "adm1"].dropna().unique().tolist())
adm1_list = ["Unknown"] + [a for a in adm1_list if a != "Unknown"]
chosen_adm1 = st.selectbox("Region / ADM1 (optional)", adm1_list, index=0) if "adm1" in df.columns else "Unknown"

continents = sorted(df["continent"].dropna().unique().tolist()) if "continent" in df.columns else ["Unknown"]
default_continent = df.loc[df["country"] == chosen_country, "continent"].mode()
if len(default_continent) == 0:
    default_continent = pd.Series(["Unknown"])
chosen_continent = st.selectbox("Continent (auto-picked if known)", continents, index=max(0, continents.index(default_continent.iloc[0]) if default_continent.iloc[0] in continents else 0)) if "continent" in df.columns else "Unknown"

level_values = sorted(df["level"].dropna().unique().tolist()) if "level" in df.columns else ["3"]
chosen_level = st.selectbox("Level (optional)", level_values, index=0) if "level" in df.columns else "3"

# Create a single-row DataFrame to predict
row = {}
row["country"] = chosen_country
if "adm1" in df.columns:
    row["adm1"] = chosen_adm1
if "continent" in df.columns:
    row["continent"] = chosen_continent
if "level" in df.columns:
    row["level"] = str(chosen_level)

X_one = pd.DataFrame([row], columns=feature_cols)
pred_label = clf.predict(X_one)[0]

proba_text = "Model does not expose probabilities."
if hasattr(clf.named_steps["clf"], "predict_proba"):
    proba = clf.predict_proba(X_one)[0]
    classes = clf.named_steps["clf"].classes_
    top_idx = proba.argsort()[::-1]
    top3 = [(classes[i], float(proba[i])) for i in top_idx[:3]]
    proba_text = ", ".join([f"{lbl}: {p:.2f}" for lbl, p in top3])

st.success(f"Predicted disaster type: {pred_label}")
st.write(f"Top probabilities: {proba_text}")

# ----------------------------
# 5) Global map of predicted hotspots
# ----------------------------
st.subheader("Global Hotspot Map (Predicted Top Disaster per Country)")

def iso3(country_name: str) -> str:
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except Exception:
        return None

countries_unique = sorted(df["country"].dropna().unique().tolist())

map_rows = []
for c in countries_unique:
    # Build a representative row for the country
    r = {"country": c}
    if "adm1" in df.columns:
        r["adm1"] = "Unknown"
    if "continent" in df.columns:
        cont_mode = df.loc[df["country"] == c, "continent"].mode()
        r["continent"] = cont_mode.iloc[0] if len(cont_mode) else "Unknown"
    if "level" in df.columns:
        r["level"] = "3"
    X_c = pd.DataFrame([r], columns=feature_cols)
    pred = clf.predict(X_c)[0]
    confidence = None
    if hasattr(clf.named_steps["clf"], "predict_proba"):
        proba = clf.predict_proba(X_c)[0]
        classes = clf.named_steps["clf"].classes_
        confidence = float(proba[classes.tolist().index(pred)])
    map_rows.append({"country": c, "predicted": pred, "confidence": confidence})

map_df = pd.DataFrame(map_rows)
map_df["iso_alpha"] = map_df["country"].apply(iso3)

# Drop rows without ISO code for mapping
map_df = map_df.dropna(subset=["iso_alpha"]).copy()

fig = px.choropleth(
    map_df,
    locations="iso_alpha",
    color="predicted",
    hover_name="country",
    hover_data={"confidence": ":.2f", "iso_alpha": False, "predicted": True},
    color_discrete_sequence=px.colors.qualitative.Set3,
    projection="natural earth"
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Download predicted country-level table"):
    st.dataframe(map_df.sort_values(["predicted", "confidence"], ascending=[True, False]))
    st.download_button(
        "Download CSV",
        data=map_df.to_csv(index=False),
        file_name="predicted_disaster_hotspots.csv",
        mime="text/csv"
    )
