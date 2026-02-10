
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
        return (
            joblib.load(LINEAR_MODEL_PATH),
            joblib.load(HYBRID_MODEL_PATH),
            joblib.load(SCALER_PATH),
        )
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


linear_model, hybrid_model, scaler = load_models()

# =============================
# CSS (Custom Styling)
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
.stTextInput input {
    background-color: white !important;
    color: White !important;
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
        st.image(str(IMAGE_PATH), use_column_width=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.8;'>Hybrid ML Model (Pass/Fail + Marks Prediction)</p>", unsafe_allow_html=True)

    sh = st.text_input("📘 Daily Study Hours", "4")
    at = st.text_input("📊 Attendance Percentage (%)", "40")
    predict = st.button("🚀 Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction & Visualization Logic
# =============================
if predict and linear_model is not None:
    try:
        sh_val = float(sh)
        at_val = float(at)

        # ---------- MODEL PIPELINE ----------
        X_scaled = scaler.transform([[sh_val, at_val]])
        linear_score = linear_model.predict(X_scaled)[0]
        hybrid_input = np.column_stack((X_scaled, [linear_score]))

        pass_prob = hybrid_model.predict_proba(hybrid_input)[0][1]

        # ---------- MARKS ----------
        marks = np.clip(
            linear_score if linear_score > 1 else linear_score * 100,
            0, 100
        )

        # ---------- REALISM CLAMP ----------
        expected_marks = (sh_val * 10) + (at_val * 0.5)
        marks = 0.6 * marks + 0.4 * expected_marks
        marks = np.clip(marks, 0, 100)

        marks = np.clip(marks, 35, 90)

        # ---------- EFFORT PENALTY ----------
        if sh_val < 3 or at_val < 40:
            pass_prob = min(pass_prob, 0.30)
            marks = min(marks, 45)

        # ---------- PASS / FAIL ----------
        result_word = "PASS" if pass_prob >= 0.5 else "FAIL"

        # ---------- STATUS (BASED ON MARKS) ----------
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

        result_big_text = result_word


        # 2. Result Card Display
        with col2:
            st.write("")
            components.html(f"""
            <div style="
                background:rgba(255,255,255,0.15);
                padding:30px;
                border-radius:25px;
                text-align:center;
                color:white;
                font-family: sans-serif;
                border: 1px solid rgba(255,255,255,0.1);">
                <h2 style="margin:0;">Prediction Result</h2>
                <p style="font-size:18px; margin:10px 0;">Pass Probability: <b>{pass_prob * 100:.1f}%</b></p>
                <p style="font-size:18px; margin:0;">Estimated Marks: <b>{marks:.1f}/100</b></p>
                <h1 style="color:{status_color}; font-size:80px; margin:15px 0; font-weight:900; letter-spacing:2px;">
                    {result_word}
                </h1>
            </div>
            """, height=320)

            # 3. Advanced Analytics Card
            components.html(f"""
            <div style="
                margin-top:20px;
                background:linear-gradient(135deg,#6a11cb,#2575fc);
                border-radius:30px;
                padding:40px;
                color:white;
                font-family: sans-serif;
            ">
                <h1 style="margin:0; font-size:26px;">Advanced Analytics</h1>
                <div style="border-left: 10px solid {status_color}; padding: 20px; background: rgba(255,255,255,0.1);">
                <h1 style="color:{status_color};">{result_big_text}</h1>
                <p>Current Standing: <b>{status_text}</b></p>
                </div>

                <div style="background:rgba(255,255,255,0.2); border-radius:15px; height:12px; margin-bottom:35px; margin-top:10px;">
                    <div style="width:{int(pass_prob * 100)}%; background:{status_color}; height:100%; border-radius:15px; box-shadow: 0 0 15px {status_color};"></div>
                </div>

                <h2 style="font-size:20px;">📌 Potential Scorecard</h2>
                <hr style="opacity: 0.2; margin-bottom:20px;">

                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>📘 <b>Theoretic Proficiency:</b> Concept clear, improve speed.</span>
                        <span>{int(pass_prob * 85)}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:8px;">
                        <div style="width:{int(pass_prob * 85)}%; background:#60a5fa; height:100%; border-radius:10px;"></div>
                    </div>
                </div>

                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>💻 <b>Application Skills:</b>  Good logic, practice projects. </span>
                        <span>{int(pass_prob * 90)}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:8px;">
                        <div style="width:{int(pass_prob * 90)}%; background:#22c55e; height:100%; border-radius:10px;"></div>
                    </div>
                </div>

                <div style="margin-top:30px; padding:20px; background:rgba(0,0,0,0.2); border-left:8px solid {status_color}; border-radius:15px;">
                    <b style="font-size:18px;">AI Recommendation:</b><br>
                    <span>{advice}</span>
                </div>
            </div>
            """, height=580)

            # 4. Performance Chart
            st.write("## 📈 Performance Benchmarking")
            hours_range = np.arange(1, 11)
            marks_trend = np.array([25, 32, 40, 48, 55, 65, 75, 82, 90, 96])

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#4B0082')

            ax.plot(hours_range, marks_trend, color='#60a5fa', linewidth=3, marker='o', markerfacecolor='white',
                    label='Average Growth')
            ax.scatter(sh_val, marks, color=status_color, s=250, zorder=5, label='Your Prediction', edgecolor='white')

            ax.annotate(f"{marks:.1f}", (sh_val, marks), xytext=(sh_val, marks + 5), color='white', fontweight='bold',
                        ha='center')

            ax.set_xlabel("Study Hours", color='white')
            ax.set_ylabel("Expected Marks", color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.1)
            ax.legend(facecolor='#4B0082', labelcolor='white')

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =============================
# Footer
# =============================
st.markdown("<br><hr><center style='opacity:0.4;'>Hybrid Predictor AI • Internship Project 2026</center>",
            unsafe_allow_html=True)



