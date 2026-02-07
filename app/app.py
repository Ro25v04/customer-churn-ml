import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

DATA_PATH = os.path.join("data", "processed", "Telco_processed.csv")
MODEL_PATH = os.path.join("models", "churn_model.pkl")
COLS_PATH  = os.path.join("models", "model_columns.pkl")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(model_path: str, cols_path: str):
    model = joblib.load(model_path)
    model_cols = joblib.load(cols_path)
    return model, model_cols

def make_features(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    # X must match training: drop target if present
    X = df.drop(columns=["Churn"], errors="ignore")

    # Encode categorical features
    X_enc = pd.get_dummies(X, drop_first=True)

    # Align columns to training columns (very important)
    X_enc = X_enc.reindex(columns=model_cols, fill_value=0)

    return X_enc

# -----------------------
# Load
# -----------------------
st.title("Customer Churn — Segments + Risk Dashboard")

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find dataset at {DATA_PATH}. Make sure it's in data/processed/")
    st.stop()

if not (os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH)):
    st.error("Model files not found. Run Day 5 saving step to create models/churn_model.pkl and models/model_columns.pkl")
    st.stop()

df = load_data(DATA_PATH)
model, model_cols = load_model(MODEL_PATH, COLS_PATH)

# -----------------------
# Predict
# -----------------------
X_enc = make_features(df, model_cols)
churn_prob = model.predict_proba(X_enc)[:, 1]
df_view = df.copy()
df_view["ChurnProbability"] = churn_prob

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

min_prob = st.sidebar.slider("Min churn probability", 0.0, 1.0, 0.5, 0.01)
segments = sorted(df_view["Segment"].unique()) if "Segment" in df_view.columns else []
selected_segments = st.sidebar.multiselect("Segments", options=segments, default=segments)

filtered = df_view.copy()
if "Segment" in filtered.columns:
    filtered = filtered[filtered["Segment"].isin(selected_segments)]
filtered = filtered[filtered["ChurnProbability"] >= min_prob]

# -----------------------
# Overview
# -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers", f"{len(df_view):,}")

with col2:
    churn_rate = df_view["Churn"].mean() if "Churn" in df_view.columns else None
    st.metric("Overall churn rate", f"{churn_rate:.2%}" if churn_rate is not None else "N/A")

with col3:
    st.metric("High-risk customers shown", f"{len(filtered):,}")

st.divider()

# -----------------------
# Charts
# -----------------------
left, right = st.columns(2)

with left:
    st.subheader("Churn rate by segment")
    if "Segment" in df_view.columns and "Churn" in df_view.columns:
        churn_by_seg = df_view.groupby("Segment")["Churn"].mean().sort_index()
        fig, ax = plt.subplots()
        ax.bar(churn_by_seg.index.astype(str), churn_by_seg.values)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Churn rate")
        st.pyplot(fig)
    else:
        st.info("Segment/Churn columns not found.")

with right:
    st.subheader("Avg churn probability by segment")
    if "Segment" in df_view.columns:
        prob_by_seg = df_view.groupby("Segment")["ChurnProbability"].mean().sort_index()
        fig, ax = plt.subplots()
        ax.bar(prob_by_seg.index.astype(str), prob_by_seg.values)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Avg churn probability")
        st.pyplot(fig)

st.divider()

# -----------------------
# Table
# -----------------------
st.subheader("Customer risk table")
show_cols = []
for c in ["Segment", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "ChurnProbability"]:
    if c in filtered.columns:
        show_cols.append(c)

st.dataframe(
    filtered.sort_values("ChurnProbability", ascending=False)[show_cols].head(200),
    use_container_width=True
)

st.divider()

# -----------------------
# Customer inspector
# -----------------------
st.subheader("Customer inspector (why is this customer high risk?)")
idx = st.number_input("Row index (from original dataframe)", min_value=0, max_value=len(df_view)-1, value=0, step=1)

row = df_view.iloc[int(idx)]
st.write("Selected customer:", row[show_cols] if show_cols else row)

# Explainability: show top positive coefficient features that are ON for this customer
st.caption("Approx explanation: features with positive coefficients that are active for this customer (Logistic Regression)")

try:
    coef = model.coef_[0]
    coef_series = pd.Series(coef, index=model_cols).sort_values(ascending=False)

    x_row = X_enc.iloc[int(idx)]
    active = x_row[x_row > 0].index

    # drivers = active features with positive coefficients
    drivers = coef_series.loc[coef_series.index.intersection(active)]
    drivers = drivers[drivers > 0].head(10)

    if len(drivers) == 0:
        st.info("No strong positive drivers found for this customer (or features not active).")
    else:
        st.dataframe(drivers.rename("Coefficient").to_frame(), use_container_width=True)
except Exception as e:
    st.warning(f"Could not compute drivers: {e}")
