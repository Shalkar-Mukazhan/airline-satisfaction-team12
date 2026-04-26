import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("pipeline_xgb.joblib")

pipe = load_model()

SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

NUM_COLS = ["Age", "Flight Distance"] + SERVICE_COLS + [
    "Total_delay", "Log_Flight_Distance", "Service_avg"
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class", "Age_group"]

# ── Sidebar: inputs ──────────────────────────────────────────────────
st.sidebar.header("Passenger Details")

st.sidebar.subheader("Demographics & Flight")
age = st.sidebar.slider("Age", 1, 85, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_dist = st.sidebar.slider("Flight Distance (km)", 50, 5000, 1000)
dep_delay = st.sidebar.slider("Departure Delay (min)", 0, 300, 0)
arr_delay = st.sidebar.slider("Arrival Delay (min)", 0, 300, 0)

st.sidebar.subheader("Service Ratings (1 = worst, 5 = best)")
ratings = {}
for col in SERVICE_COLS:
    ratings[col] = st.sidebar.slider(col, 1, 5, 3)

# ── Feature engineering (mirrors training pipeline) ──────────────────
total_delay = dep_delay + arr_delay
log_fd = np.log1p(flight_dist)
service_avg = np.mean(list(ratings.values()))

age_bins = [0, 25, 40, 60, 100]
age_labels = ["young", "adult", "middle", "senior"]
age_group = pd.cut([age], bins=age_bins, labels=age_labels)[0]

row = {
    "Age": age,
    "Flight Distance": flight_dist,
    **ratings,
    "Total_delay": total_delay,
    "Log_Flight_Distance": log_fd,
    "Service_avg": service_avg,
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": flight_class,
    "Age_group": str(age_group),
}

X_input = pd.DataFrame([row])[NUM_COLS + CAT_COLS]

# ── Main panel ───────────────────────────────────────────────────────
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("**Team 12** — MSc Machine Learning Group Project | XGBoost + SHAP")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    prob = pipe.predict_proba(X_input)[0][1]
    pred = int(pipe.predict(X_input)[0])

    if pred == 1:
        st.success(f"### 😊 Satisfied")
        st.metric("Confidence", f"{prob:.1%}")
    else:
        st.error(f"### 😞 Neutral / Dissatisfied")
        st.metric("Confidence", f"{1 - prob:.1%}")

    st.divider()
    st.markdown("**Top service drivers:**")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_ratings[:3]:
        st.markdown(f"- **{name}**: {val}/5")
    st.markdown("**Lowest rated:**")
    for name, val in sorted_ratings[-2:]:
        st.markdown(f"- **{name}**: {val}/5")

with col2:
    st.subheader("SHAP Explanation — Why this prediction?")

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feature_names = (
        NUM_COLS
        + list(pre.named_transformers_["cat"]
               .named_steps["enc"]
               .get_feature_names_out(CAT_COLS))
    )

    X_transformed = pre.transform(X_input)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    exp = shap.Explanation(
        values=sv,
        base_values=float(explainer.expected_value),
        data=X_transformed[0],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=15, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        "Each bar shows how much a feature pushed the prediction toward satisfied (red, right) "
        "or neutral/dissatisfied (blue, left). The base value is the average model output across all passengers."
    )

st.divider()
st.caption(
    "Model: XGBoost (F1 = 0.9643, ROC-AUC = 0.9952) trained on Airline Passenger Satisfaction dataset (Kaggle). "
    "Team 12 — Nazym Mailaubayeva & Shalkar Mukazhan."
)
