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
    return (
        joblib.load(LINEAR_MODEL_PATH),
        joblib.load(HYBRID_MODEL_PATH),
        joblib.load(SCALER_PATH),
    )

linear_model, hybrid_model, scaler = load_models()

# =============================
# CSS
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
}
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 45px;
    text-align: center;
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# UI Layout
# =============================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_column_width=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)

    sh = st.text_input("📘 Daily Study Hours", "6")
    at = st.text_input("📊 Attendance Percentage (%)", "75")
    predict = st.button("🚀 Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction
# =============================
if predict:
    sh_val = float(sh)
    at_val = float(at)

    X_scaled = scaler.transform([[sh_val, at_val]])
    linear_score = linear_model.predict(X_scaled)[0]
    hybrid_input = np.column_stack((X_scaled, [linear_score]))

    pass_prob = hybrid_model.predict_proba(hybrid_input)[0][1]

    marks = np.clip((sh_val * 10 + at_val * 0.5), 35, 90)

    result_word = "PASS" if pass_prob >= 0.5 else "FAIL"

    if marks >= 85:
        status_text = "Excellent"
        status_color = "#22C55E"
        advice = "Fantastic! Your preparation is solid."
    elif marks >= 65:
        status_text = "Good"
        status_color = "#FACC15"
        advice = "You're on the right track."
    else:
        status_text = "Needs Improvement"
        status_color = "#EF4444"
        advice = "Immediate attention required."

    with col2:
        # =============================
        # RESULT CARD
        # =============================
        components.html(f"""
        <div style="
            background:rgba(255,255,255,0.15);
            padding:30px;
            border-radius:25px;
            text-align:center;
        ">
            <h2>Prediction Result</h2>
            <p>Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p>Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{status_color}; font-size:80px;">{result_word}</h1>
        </div>
        """, height=300)

        # =============================
        # ADVANCED ANALYTICS
        # =============================
        components.html(f"""
        <div style="
            margin-top:20px;
            background:linear-gradient(135deg,#6a11cb,#2575fc);
            border-radius:30px;
            padding:40px;
            color:white;
        ">
            <h1>Advanced Analytics</h1>

            <!-- MAIN PROGRESS BAR -->
            <div style="
                background:rgba(255,255,255,0.2);
                border-radius:15px;
                height:28px;
                margin-top:15px;
                overflow:hidden;
            ">
                <div style="
                    width:{int(pass_prob*100)}%;
                    background:{status_color};
                    height:100%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-weight:bold;
                    transition:width 1s ease-in-out;
                ">
                    {int(pass_prob*100)}%
                </div>
            </div>

            <h3 style="margin-top:30px;">📌 Potential Scorecard</h3>

            <!-- THEORY -->
            <p>📘 Theoretic Proficiency</p>
            <div style="background:rgba(255,255,255,0.15); height:22px; border-radius:10px;">
                <div style="
                    width:{int(pass_prob*85)}%;
                    background:#60a5fa;
                    height:100%;
                    border-radius:10px;
                    text-align:center;
                    font-size:13px;
                    font-weight:bold;
                ">
                    {int(pass_prob*85)}%
                </div>
            </div>

            <!-- PRACTICAL -->
            <p style="margin-top:15px;">💻 Application Skills</p>
            <div style="background:rgba(255,255,255,0.15); height:22px; border-radius:10px;">
                <div style="
                    width:{int(pass_prob*90)}%;
                    background:#22c55e;
                    height:100%;
                    border-radius:10px;
                    text-align:center;
                    font-size:13px;
                    font-weight:bold;
                ">
                    {int(pass_prob*90)}%
                </div>
            </div>

            <div style="
                margin-top:30px;
                background:rgba(0,0,0,0.25);
                padding:20px;
                border-left:6px solid {status_color};
                border-radius:15px;
            ">
                <b>AI Recommendation:</b><br>{advice}
            </div>
        </div>
        """, height=600)

        # =============================
        # PERFORMANCE CHART
        # =============================
        st.write("## 📈 Performance Benchmarking")

        hours_range = np.arange(1, 11)
        marks_trend = [25, 32, 40, 48, 55, 65, 75, 82, 90, 96]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('#4B0082')
        fig.patch.set_facecolor('#000000')

        ax.plot(hours_range, marks_trend, linewidth=3, marker='o', label='Average Growth')
        ax.scatter(sh_val, marks, s=200, label='Your Prediction')

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Expected Marks")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)

# =============================
# Footer
# =============================
st.markdown("<hr><center style='opacity:0.4;'>Hybrid Predictor AI • Internship Project 2026</center>",
            unsafe_allow_html=True)
