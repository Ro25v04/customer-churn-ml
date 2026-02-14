import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------
# Page + Theme
# -----------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide", page_icon="📉")

# Small CSS polish (safe + simple)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; }
      .kpi-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .kpi-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
      .kpi-value { font-size: 1.6rem; font-weight: 700; margin: 0; }
      .kpi-sub { font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }
      .pill {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.15);
        margin-left: 8px;
      }
      .pill-low { background: rgba(0, 200, 0, 0.15); }
      .pill-med { background: rgba(255, 165, 0, 0.15); }
      .pill-high{ background: rgba(255, 0, 0, 0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Config
# -----------------------
DATA_PATH = os.path.join("data", "processed", "Telco_processed.csv")
MODEL_PATH = os.path.join("models", "churn_model.pkl")
COLS_PATH = os.path.join("models", "model_columns.pkl")

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
    # Drop non-features (target + identifiers)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # Encode categorical features
    X_enc = pd.get_dummies(X, drop_first=True)

    # Align columns to training columns
    X_enc = X_enc.reindex(columns=model_cols, fill_value=0)
    return X_enc


def risk_label(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


def risk_pill(p: float) -> str:
    lab = risk_label(p)
    cls = "pill-low" if lab == "Low" else (
        "pill-med" if lab == "Medium" else "pill-high")
    return f'<span class="pill {cls}">{lab}</span>'


# -----------------------
# Load
# -----------------------
st.markdown("## Customer Churn — Segments + Risk Dashboard")
st.caption(
    "Explore churn risk, segment behavior, and per-customer explanations.")

if not os.path.exists(DATA_PATH):
    st.error(
        f"Could not find dataset at {DATA_PATH}. Make sure it's in data/processed/")
    st.stop()

if not (os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH)):
    st.error("Model files not found. Create models/churn_model.pkl and models/model_columns.pkl (Day 5 saving step).")
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
# Sidebar
# -----------------------
st.sidebar.header("🔎 Filters")

min_prob = st.sidebar.slider("Min churn probability", 0.0, 1.0, 0.50, 0.01)

segments = sorted(df_view["Segment"].unique()
                  ) if "Segment" in df_view.columns else []
selected_segments = st.sidebar.multiselect(
    "Segments", options=segments, default=segments)

top_n = st.sidebar.slider("Rows to show (table)", 50, 500, 200, 50)

if st.sidebar.button("Reset filters"):
    st.rerun()

# Apply filters
filtered = df_view.copy()
if "Segment" in filtered.columns:
    filtered = filtered[filtered["Segment"].isin(selected_segments)]
filtered = filtered[filtered["ChurnProbability"] >= min_prob]

# -----------------------
# KPIs (pretty cards)
# -----------------------
k1, k2, k3, k4 = st.columns(4)

total_customers = len(df_view)
overall_churn_rate = df_view["Churn"].mean(
) if "Churn" in df_view.columns else None
high_risk_count = int((df_view["ChurnProbability"] >= 0.66).sum())
shown_count = len(filtered)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Customers</div>
          <p class="kpi-value">{total_customers:,}</p>
          <div class="kpi-sub">Total rows in dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    val = f"{overall_churn_rate:.2%}" if overall_churn_rate is not None else "N/A"
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Overall churn rate</div>
          <p class="kpi-value">{val}</p>
          <div class="kpi-sub">Based on Churn column</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">High-risk customers</div>
          <p class="kpi-value">{high_risk_count:,}</p>
          <div class="kpi-sub">ChurnProbability ≥ 0.66</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Shown after filters</div>
          <p class="kpi-value">{shown_count:,}</p>
          <div class="kpi-sub">Segment + probability filters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Overview", "🧾 Risk Table", "🔍 Customer Inspector"])

# -----------------------
# Tab 1: Charts
# -----------------------
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Churn rate by segment")
        if "Segment" in df_view.columns and "Churn" in df_view.columns:
            churn_by_seg = df_view.groupby(
                "Segment")["Churn"].mean().sort_index()
            fig, ax = plt.subplots()
            ax.bar(churn_by_seg.index.astype(str), churn_by_seg.values)
            ax.set_xlabel("Segment")
            ax.set_ylabel("Churn rate")
            ax.set_ylim(0, max(0.55, churn_by_seg.max() + 0.05))
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Segment/Churn columns not found.")

    with right:
        st.subheader("Avg churn probability by segment")
        if "Segment" in df_view.columns:
            prob_by_seg = df_view.groupby(
                "Segment")["ChurnProbability"].mean().sort_index()
            fig, ax = plt.subplots()
            ax.bar(prob_by_seg.index.astype(str), prob_by_seg.values)
            ax.set_xlabel("Segment")
            ax.set_ylabel("Avg churn probability")
            ax.set_ylim(0, max(0.75, prob_by_seg.max() + 0.05))
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Segment column not found.")

# -----------------------
# Tab 2: Table (with nicer formatting)
# -----------------------
with tab2:
    st.subheader("Customer risk table")

    show_cols = []
    for c in ["customerID", "Segment", "tenure", "MonthlyCharges", "TotalCharges", "Churn", "ChurnProbability"]:
        if c in filtered.columns:
            show_cols.append(c)

    table = filtered.sort_values("ChurnProbability", ascending=False)[
        show_cols].head(top_n).copy()

    if "ChurnProbability" in table.columns:
        table["Risk"] = table["ChurnProbability"].apply(risk_label)

    # Make table look nicer
    if "ChurnProbability" in table.columns:
        styled = (
            table.style
            .format({"ChurnProbability": "{:.3f}"})
            .background_gradient(subset=["ChurnProbability"], cmap="Reds")
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(table, use_container_width=True)

# -----------------------
# Tab 3: Inspector (customerID selection)
# -----------------------
with tab3:
    st.subheader("Customer inspector (why is this customer high risk?)")

    if "customerID" not in df_view.columns:
        st.info(
            "customerID column not found. Add it to Telco_processed.csv for UI selection.")
        st.stop()

    # Optionally limit dropdown to filtered customers (feels more useful)
    dropdown_source = filtered if len(
        filtered) > 0 and "customerID" in filtered.columns else df_view
    selected_id = st.selectbox(
        "Select customerID",
        options=dropdown_source["customerID"].astype(str).unique()
    )

    idx = df_view.index[df_view["customerID"].astype(
        str) == str(selected_id)][0]
    row = df_view.loc[idx]

    # Headline info
    p = float(row["ChurnProbability"])
    st.markdown(
        f"**ChurnProbability:** `{p:.3f}` {risk_pill(p)}",
        unsafe_allow_html=True
    )

    # Show selected customer
    show_cols_inspector = [c for c in ["customerID", "Segment", "tenure", "MonthlyCharges",
                                       "TotalCharges", "Churn", "ChurnProbability"] if c in df_view.columns]
    st.write("Selected customer:")
    st.dataframe(row[show_cols_inspector].to_frame(
        "value"), use_container_width=True)

    st.caption("Approx explanation: top positive model coefficients that are active for this customer (Logistic Regression).")

    try:
        coef = model.coef_[0]
        coef_series = pd.Series(
            coef, index=model_cols).sort_values(ascending=False)

        x_row = X_enc.loc[idx]
        active = x_row[x_row > 0].index

        drivers = coef_series.loc[coef_series.index.intersection(active)]
        drivers = drivers[drivers > 0].head(12)

        if len(drivers) == 0:
            st.info(
                "No strong positive drivers found for this customer (or features not active).")
        else:
            driver_df = drivers.rename("Coefficient").to_frame()
            st.dataframe(driver_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute drivers: {e}")
