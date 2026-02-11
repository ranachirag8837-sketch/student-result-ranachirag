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
        scaler_obj = joblib.load(SCALER_PATH)
        return linear, hybrid, scaler_obj
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None


linear_model, hybrid_model, scaler = load_models()


# =============================
# CSS Styling (UNCHANGED DESIGN)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 45px;
    width: 100%;
    font-weight: bold;
    border: none;
}
.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =============================
# UI Layout
# =============================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_container_width=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='opacity:0.8;'>Hybrid ML Model (Scientifically Aligned)</p>",
        unsafe_allow_html=True
    )

    sh = st.number_input("📘 Daily Study Hours", 0.0, 12.0, 4.0, step=0.5)
    at = st.number_input("📊 Attendance Percentage (%)", 0.0, 100.0, 40.0, step=1.0)

    predict = st.button("🚀 Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================
# Prediction Logic (ALIGNED)
# =============================
if predict and linear_model is not None and scaler is not None:
    try:
        sh_val = np.clip(float(sh), 0, 12)
        at_val = np.clip(float(at), 0, 100)

        # ---------- MODEL PIPELINE ----------
        X_scaled = scaler.transform([[sh_val, at_val]])
        linear_score = float(linear_model.predict(X_scaled)[0])

        # Convert to marks if model outputs 0–1
        if linear_score <= 1:
            model_marks = linear_score * 100
        else:
            model_marks = linear_score

        model_marks = np.clip(model_marks, 0, 100)

        # ---------- SCIENTIFIC EXPECTATION MODEL ----------
        # Study weight = 60%, Attendance = 40%
        expected_marks = (sh_val / 12) * 60 + (at_val / 100) * 40

        # Blend ML + Academic Formula
        marks = 0.3 * model_marks + 0.7 * expected_marks
        marks = np.clip(marks, 0, 100)

        # ---------- PASS LOGIC (BASED ON MARKS) ----------
        pass_threshold = 35
        result_word = "PASS" if marks >= pass_threshold else "FAIL"

        # ---------- PROBABILITY ALIGNED WITH MARKS ----------
        # Smooth probability curve based on marks
        pass_prob = 1 / (1 + np.exp(-(marks - pass_threshold)/5))
        pass_prob = np.clip(pass_prob, 0, 1)

        # ---------- STATUS ----------
        if marks >= 85:
            status_text = "Excellent"
            status_color = "#22C55E"
            advice = "Outstanding academic performance!"
        elif marks >= 65:
            status_text = "Good"
            status_color = "#FACC15"
            advice = "Strong performance. Keep improving!"
        elif marks >= 35:
            status_text = "Needs Improvement"
            status_color = "#F97316"
            advice = "You passed, but more effort is needed."
        else:
            status_text = "At Risk"
            status_color = "#EF4444"
            advice = "High risk of failure. Immediate attention required."

        # PASS color independent
        result_color = "#22C55E" if result_word == "PASS" else "#EF4444"

        # =============================
        # Result Card
        # =============================
        with col2:

            components.html(f"""
            <div style="
                background:rgba(255,255,255,0.15);
                padding:30px;
                border-radius:25px;
                text-align:center;
                color:white;
                font-family:sans-serif;
            ">
                <h2>Prediction Result</h2>
                <p>Pass Probability: <b>{pass_prob * 100:.1f}%</b></p>
                <p>Estimated Marks: <b>{marks:.1f}%</b></p>
                <h1 style="color:{result_color}; font-size:70px;">
                    {result_word}
                </h1>
            </div>
            """, height=300)

            st.markdown(f"### Current Standing: {status_text}")
            st.info(advice)

            # =============================
            # Performance Chart
            # =============================
            st.write("## 📈 Performance Benchmarking")

            hours_range = np.arange(1, 11)
            marks_trend = (hours_range / 12) * 60 + (at_val / 100) * 40

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hours_range, marks_trend, marker='o', linewidth=3)
            ax.scatter(sh_val, marks, s=200)

            ax.set_xlabel("Study Hours")
            ax.set_ylabel("Expected Marks (%)")
            ax.set_ylim(0, 100)
            ax.grid(True)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Hybrid Predictor AI • Scientifically Aligned Version • 2026")

