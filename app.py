# =============================================================================
# app.py — Streamlit Web Application for Car Evaluation Predictions
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Run     : streamlit run app.py
# Purpose : Interactive UI where a user selects car attributes and gets
#           a predicted acceptability class with confidence scores.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Evaluation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* ── Headers ── */
h1 { font-size: 2.4rem !important; font-weight: 700 !important;
     background: linear-gradient(90deg, #4CC9F0, #7209B7);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.6rem !important; font-weight: 600 !important; color: #c8d6e5 !important; }
h3 { font-size: 1.2rem !important; font-weight: 600 !important; color: #b8c2d4 !important; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(8px);
}

/* ── Prediction Badge ── */
.pred-badge {
    display: inline-block;
    padding: 14px 32px;
    border-radius: 50px;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.badge-unacc  { background: linear-gradient(135deg,#E63946,#C1121F); color: #fff; }
.badge-acc    { background: linear-gradient(135deg,#4361EE,#3A0CA3); color: #fff; }
.badge-good   { background: linear-gradient(135deg,#2DC653,#1B7A34); color: #fff; }
.badge-vgood  { background: linear-gradient(135deg,#F9844A,#F3722C); color: #fff; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Select boxes ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
FEATURE_COLS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

FEATURE_OPTIONS = {
    "buying":   ["low", "med", "high", "vhigh"],
    "maint":    ["low", "med", "high", "vhigh"],
    "doors":    ["2", "3", "4", "5more"],
    "persons":  ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety":   ["low", "med", "high"],
}

FEATURE_DISPLAY = {
    "buying":   "Buying Price 💰",
    "maint":    "Maintenance Cost 🔧",
    "doors":    "Number of Doors 🚪",
    "persons":  "Seating Capacity 👥",
    "lug_boot": "Luggage Boot Size 🧳",
    "safety":   "Safety Rating 🛡️",
}

CLASS_DISPLAY = {
    "unacc": ("Unacceptable", "badge-unacc", "❌"),
    "acc":   ("Acceptable",   "badge-acc",   "✅"),
    "good":  ("Good",         "badge-good",  "👍"),
    "vgood": ("Very Good",    "badge-vgood", "🌟"),
}

CLASS_COLORS = {
    "unacc": "#E63946",
    "acc":   "#4361EE",
    "good":  "#2DC653",
    "vgood": "#F9844A",
}


# ── Model Loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all saved models and encoders from disk (cached)."""
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree":       "decision_tree.pkl",
        "Random Forest":       "random_forest.pkl",
    }
    # Load XGBoost if available
    if os.path.exists(os.path.join(MODELS_DIR, "xgboost.pkl")):
        model_files["XGBoost"] = "xgboost.pkl"

    for name, fname in model_files.items():
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            models[name] = joblib.load(fpath)

    ordinal_enc = joblib.load(os.path.join(MODELS_DIR, "ordinal_encoder.pkl"))
    label_enc   = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return models, ordinal_enc, label_enc


# ── Prediction Logic ──────────────────────────────────────────────────────────

def predict_car(model, ordinal_enc, label_enc, user_input: dict):
    """
    Encode user input and return predicted class + probabilities.

    Parameters:
        model       : Trained sklearn model.
        ordinal_enc : Fitted OrdinalEncoder.
        label_enc   : Fitted LabelEncoder.
        user_input  : {feature: value} dict from sidebar.

    Returns:
        pred_label  : Predicted class string (e.g., 'acc').
        proba       : Class probability array.
    """
    # Build DataFrame with the correct column order
    row = pd.DataFrame([user_input], columns=FEATURE_COLS)
    # Encode features
    X_enc = ordinal_enc.transform(row)
    # Predict
    pred_idx   = model.predict(X_enc)[0]
    pred_label = label_enc.classes_[pred_idx]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_enc)[0]
    else:
        # Fallback: one-hot probability
        proba = np.zeros(len(label_enc.classes_))
        proba[pred_idx] = 1.0

    return pred_label, proba


# ── Probability Bar Chart ─────────────────────────────────────────────────────

def plot_proba_chart(proba, class_names):
    """Draw a horizontal probability bar chart using matplotlib."""
    colors = [CLASS_COLORS.get(c, "#888") for c in class_names]
    display_names = [CLASS_DISPLAY[c][0] for c in class_names]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bars = ax.barh(display_names, proba * 100, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.55)

    for bar, val in zip(bars, proba * 100):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10,
                fontweight="bold", color="white")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#c8d6e5")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🚗 Car Evaluation System")
    st.markdown("**Predict car acceptability using Machine Learning** | *CodTech Internship Project*")
    st.markdown("---")

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        models, ordinal_enc, label_enc = load_models()
        class_names = list(label_enc.classes_)
    except FileNotFoundError:
        st.error(
            "⚠️ **Models not found!**  \n"
            "Please run `python main.py` first to train and save all models."
        )
        st.stop()

    if not models:
        st.error("No models could be loaded. Please run `python main.py` to train models.")
        st.stop()

    # ── Sidebar — Feature Inputs ──────────────────────────────────────────────
    st.sidebar.markdown("## 🔧 Car Attributes")
    st.sidebar.markdown("Select the car's features below:")
    st.sidebar.markdown("---")

    user_input = {}
    for col, display_name in FEATURE_DISPLAY.items():
        user_input[col] = st.sidebar.selectbox(
            label=display_name,
            options=FEATURE_OPTIONS[col],
            key=f"select_{col}",
        )

    st.sidebar.markdown("---")

    # ── Model Selection ───────────────────────────────────────────────────────
    selected_model_name = st.sidebar.selectbox(
        "🤖 Select Model",
        options=list(models.keys()),
        index=list(models.keys()).index("Random Forest")
            if "Random Forest" in models else 0,
        key="model_select"
    )
    selected_model = models[selected_model_name]

    # ── Predict Button ────────────────────────────────────────────────────────
    predict_btn = st.sidebar.button("🔍 Predict Now", use_container_width=True)

    # ── Intern Info ────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**👤 Intern Information**")
    st.sidebar.markdown("**Name:** Mayank")
    st.sidebar.markdown("**ID:** CTTS148")
    st.sidebar.markdown("**Duration:** 4 Weeks")
    st.sidebar.markdown("**Degree:** B.Tech AIML (5th Sem)")
    st.sidebar.markdown("**Company:** CodTech IT Solutions")

    # ── Layout: 2 columns ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("### 📋 Car Feature Summary")

        # Display selected attributes as a styled table
        feature_df = pd.DataFrame({
            "Feature": list(FEATURE_DISPLAY.values()),
            "Selected Value": [user_input[c].upper() for c in FEATURE_COLS],
        })
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature":        st.column_config.TextColumn("Feature", width="medium"),
                "Selected Value": st.column_config.TextColumn("Value",   width="small"),
            }
        )

        # ── Prediction Result ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        if predict_btn:
            pred_label, proba = predict_car(
                selected_model, ordinal_enc, label_enc, user_input
            )
            display_name, badge_class, icon = CLASS_DISPLAY[pred_label]
            confidence = proba.max() * 100

            st.markdown(
                f'<div class="pred-badge {badge_class}">'
                f'{icon}  {display_name}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Model:** {selected_model_name}  &nbsp;|&nbsp;  **Confidence:** {confidence:.1f}%")

            # Probability bar chart
            st.markdown("#### 📊 Class Probability Distribution")
            fig = plot_proba_chart(proba, class_names)
            st.pyplot(fig, use_container_width=True)

        else:
            st.info("👈 Configure the car attributes in the sidebar and click **Predict Now**.")

    with col2:
        st.markdown("### 📚 About the Dataset")
        st.markdown("""
The **UCI Car Evaluation Dataset** is a classic classification benchmark:
- **1,728** car instances
- **6 input features** (all categorical/ordinal)
- **4 target classes**: Unacceptable, Acceptable, Good, Very Good

| Feature | Description |
|---------|-------------|
| Buying | Purchase price |
| Maint | Maintenance cost |
| Doors | No. of doors |
| Persons | Seating capacity |
| Lug Boot | Luggage boot size |
| Safety | Safety rating |
        """)

        st.markdown("---")
        st.markdown("### 🤖 Models Overview")

        # Class legend
        for cls, (label, badge, icon) in CLASS_DISPLAY.items():
            color = CLASS_COLORS[cls]
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">{icon} {label}</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 🏆 Model Performance")
        st.markdown("*(Approximate scores on test set)*")

        # Static performance table (approximate; actual values printed in terminal after main.py)
        perf_data = {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "Accuracy": ["~85%", "~97%", "~97%+"],
            "Recommended": ["Baseline", "Good", "✅ Best"],
        }
        st.dataframe(
            pd.DataFrame(perf_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
1. Select car attributes in the sidebar
2. Choose a model
3. Click **Predict Now**
4. The model encodes your inputs and returns the predicted class + probabilities
        """)


if __name__ == "__main__":
    main()
