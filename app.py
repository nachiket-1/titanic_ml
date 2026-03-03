import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.preprocess import engineer_features, get_feature_columns

MODEL_PATH = "models/titanic_model.pkl"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to predict survival probability.")

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]

model = load_model()

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🧳 Passenger Details")

pclass   = st.sidebar.selectbox("Passenger Class", [1, 2, 3],
                                  format_func=lambda x: f"Class {x}")
sex      = st.sidebar.radio("Sex", ["male", "female"])
age      = st.sidebar.slider("Age", 1, 80, 28)
sibsp    = st.sidebar.slider("Siblings / Spouses Aboard", 0, 8, 0)
parch    = st.sidebar.slider("Parents / Children Aboard", 0, 6, 0)
fare     = st.sidebar.slider("Fare Paid (£)", 0, 500, 32)
embarked = st.sidebar.selectbox("Port of Embarkation",
                                  ["S", "C", "Q"],
                                  format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
title    = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

# ── Build Input DataFrame ─────────────────────────────────────────────────────
def build_input():
    family_size = sibsp + parch + 1
    is_alone    = int(family_size == 1)

    age_band_labels = ["Child","Young","Adult","MiddleAge","Senior"]
    age_bins = [0, 16, 32, 48, 64, 80]
    age_band = pd.cut([age], bins=age_bins, labels=age_band_labels)[0]

    fare_band_labels = ["Low","Mid","High","VeryHigh"]
    fare_bins = [0, 7.91, 14.45, 31.0, 512.33]
    fare_band = pd.cut([fare], bins=fare_bins, labels=fare_band_labels)[0]

    return pd.DataFrame([{
        "Age": age, "Fare": fare, "SibSp": sibsp, "Parch": parch,
        "FamilySize": family_size, "IsAlone": is_alone, "HasCabin": 0,
        "Pclass": pclass, "Sex": sex, "Embarked": embarked,
        "Title": title, "AgeBand": age_band, "FareBand": fare_band
    }])


# ── Predict ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Passenger Summary")
    summary = {
        "Class": f"Class {pclass}", "Sex": sex.capitalize(), "Age": age,
        "Siblings/Spouses": sibsp, "Parents/Children": parch,
        "Fare": f"£{fare}", "Embarked": embarked, "Title": title
    }
    for k, v in summary.items():
        st.write(f"**{k}:** {v}")

with col2:
    st.subheader("🎯 Prediction")
    input_df = build_input()

    pred       = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    survival_prob = pred_proba[1] * 100

    if pred == 1:
        st.success(f"✅ **SURVIVED** — {survival_prob:.1f}% confidence")
    else:
        st.error(f"❌ **DID NOT SURVIVE** — {100 - survival_prob:.1f}% confidence")

    # Probability bar
    st.markdown("**Survival Probability**")
    st.progress(int(survival_prob))
    st.caption(f"Survival: {survival_prob:.1f}%  |  Not Survived: {100 - survival_prob:.1f}%")

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Feature Importance")

try:
    rf_model    = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances   = rf_model.feature_importances_

    top_n  = 15
    idx    = np.argsort(importances)[::-1][:top_n]
    labels = [feature_names[i].replace("num__","").replace("cat__","") for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Most Important Features")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.info("Train the model first to see feature importances.")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | Titanic Dataset from Kaggle")