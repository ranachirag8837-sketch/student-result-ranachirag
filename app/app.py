import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from pathlib import Path

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction AI",
    page_icon="🎓",
    layout="wide"
)

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
IMAGE_PATH = BASE_DIR / "assets" / "header_banner.png"

LINEAR_MODEL_PATH = MODEL_DIR / "linear_model.pkl"
HYBRID_MODEL_PATH = MODEL_DIR / "hybrid_logistic_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# =============================
# Load Models
# =============================
@st.cache_resource
def load_models():
    try:
        linear = joblib.load(LINEAR_MODEL_PATH)
        hybrid = joblib.load(HYBRID_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return linear, hybrid, scaler
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

linear_model, hybrid_model, scaler = load_models()

# =============================
# UI Layout
# =============================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_column_width=True)

    st.markdown("## 🎓 Student Result Prediction")
    st.markdown("Hybrid ML Model (Pass/Fail + Marks Prediction)")

    # 🔐 Safe Inputs
    sh = st.number_input("📘 Daily Study Hours", 0.0, 12.0, 4.0)
    at = st.number_input("📊 Attendance Percentage (%)", 0.0, 100.0, 40.0)

    predict = st.button("🚀 Generate Prediction")

# =============================
# Prediction Logic
# =============================
if predict:

    if None in (linear_model, hybrid_model, scaler):
        st.error("Model not loaded properly.")
        st.stop()

    try:
        # Safety Clamp
        sh_val = np.clip(float(sh), 0, 12)
        at_val = np.clip(float(at), 0, 100)

        # ---------- MODEL PIPELINE ----------
        X_scaled = scaler.transform([[sh_val, at_val]])
        linear_score = linear_model.predict(X_scaled)[0]
        hybrid_input = np.column_stack((X_scaled, [linear_score]))

        pass_prob = hybrid_model.predict_proba(hybrid_input)[0][1]

        # 🔐 Safety Clamp for Probability
        pass_prob = np.clip(pass_prob, 0, 1)

        # ---------- MARKS ----------
        raw_marks = linear_score if linear_score > 1 else linear_score * 100
        raw_marks = np.clip(raw_marks, 0, 100)

        expected_marks = (sh_val * 10) + (at_val * 0.5)
        marks = 0.6 * raw_marks + 0.4 * expected_marks
        marks = np.clip(marks, 35, 90)

        # ---------- EFFORT PENALTY ----------
        if sh_val < 3 or at_val < 40:
            pass_prob = min(pass_prob, 0.30)
            marks = min(marks, 45)

        # ---------- PASS / FAIL ----------
        result_word = "PASS" if pass_prob >= 0.5 else "FAIL"

        # ---------- STATUS ----------
        if marks >= 85:
            status_text = "Excellent"
            status_color = "#22C55E"
            advice = "Fantastic! Your preparation is solid."
        elif marks >= 65:
            status_text = "Good"
            status_color = "#FACC15"
            advice = "You're on the right track. Keep it up!"
        else:
            status_text = "Needs Improvement"
            status_color = "#EF4444"
            advice = "Immediate attention required."

        safe_percent = int(pass_prob * 100)

        # =============================
        # Result Card
        # =============================
        with col2:

            st.markdown("### 📊 Prediction Result")
            st.metric("Pass Probability", f"{safe_percent}%")
            st.metric("Estimated Marks", f"{marks:.1f}/100")
            st.markdown(f"## {result_word}")

            # Progress Bar
            st.progress(safe_percent / 100)

            # Status
            st.markdown(f"**Status:** {status_text}")
            st.info(advice)

            # =============================
            # Performance Chart
            # =============================
            st.write("## 📈 Performance Benchmarking")

            hours_range = np.arange(1, 11)
            marks_trend = np.array([25, 32, 40, 48, 55, 65, 75, 82, 90, 96])

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(hours_range, marks_trend, linewidth=3, marker='o',
                    label='Average Growth')

            ax.scatter(sh_val, marks, s=200,
                       label='Your Prediction')

            ax.annotate(f"{marks:.1f}",
                        (sh_val, marks),
                        xytext=(sh_val, marks + 5),
                        ha='center')

            ax.set_xlabel("Study Hours")
            ax.set_ylabel("Expected Marks")
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Hybrid Predictor AI • Internship Project 2026")
