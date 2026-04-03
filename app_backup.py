# ============================================================
# app.py — AI-Based Heart Risk Prediction System (Enhanced)
# ============================================================
# Hybrid Streamlit web application with:
#   1. ECG Signal Analysis (CNN-LSTM)
#   2. Clinical Risk Prediction (Random Forest)
#   3. Hybrid Fusion Logic
#   4. ★ PDF Report with embedded charts
#   5. ★ Real-time ECG animation (hospital monitor style)
#   6. ★ Patient vs Population comparison
#   7. ★ Feature importance chart
#   8. ★ Multi-language support (English / Hindi / Telugu)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import datetime

matplotlib.use("Agg")

from utils import (
    load_ecg_model,
    read_uploaded_file,
    preprocess_signal,
    predict_ecg,
    parse_manual_input,
    generate_report,
    generate_hybrid_report,
    generate_pdf_report,
    load_clinical_model,
    predict_clinical,
    compute_fusion_risk,
    get_patient_comparison,
    get_feature_importance,
    get_text,
    TRANSLATIONS,
    EXPECTED_LENGTH,
)


# ────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Heart Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* Hero */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem; border-radius: 20px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);
    }
    .hero-header::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(220,38,38,0.08) 0%, transparent 60%);
        animation: pulse-glow 4s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    .hero-header h1 { color: #fff; font-size: 2.2rem; font-weight: 800; margin: 0 0 0.5rem 0; position: relative; z-index: 1; }
    .hero-subtitle { color: rgba(255,255,255,0.7); font-size: 0.95rem; position: relative; z-index: 1; }
    .hero-badges { display: flex; justify-content: center; gap: 0.7rem; margin-top: 1rem; position: relative; z-index: 1; flex-wrap: wrap; }
    .hero-badge { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.75rem; font-weight: 500; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.3rem;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card h3 { color: rgba(255,255,255,0.55); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.4rem; }
    .metric-card .metric-value { font-size: 1.4rem; font-weight: 700; color: #fff; }

    /* Result boxes */
    .result-normal { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); border: 1px solid #10b981; border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 4px 24px rgba(16,185,129,0.2); animation: fadeInUp 0.6s ease; }
    .result-abnormal { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border: 1px solid #ef4444; border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 4px 24px rgba(239,68,68,0.2); animation: fadeInUp 0.6s ease; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .result-label { font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; }
    .result-confidence { font-size: 1rem; color: rgba(255,255,255,0.8); }

    /* Section headers */
    .section-header { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); border: 1px solid rgba(99,102,241,0.2); border-radius: 14px; padding: 1rem 1.5rem; margin: 1.5rem 0 1rem 0; }
    .section-header h2 { color: #c7d2fe; font-size: 1.1rem; font-weight: 600; margin: 0; }

    /* Risk cards */
    .risk-card-low { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); border: 2px solid #10b981; border-radius: 20px; padding: 2.2rem; text-align: center; box-shadow: 0 8px 32px rgba(16,185,129,0.25); animation: fadeInUp 0.6s ease; }
    .risk-card-medium { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); border: 2px solid #f59e0b; border-radius: 20px; padding: 2.2rem; text-align: center; box-shadow: 0 8px 32px rgba(245,158,11,0.25); animation: fadeInUp 0.6s ease; }
    .risk-card-high { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border: 2px solid #ef4444; border-radius: 20px; padding: 2.2rem; text-align: center; box-shadow: 0 8px 32px rgba(239,68,68,0.25); animation: fadeInUp 0.6s ease; }
    .risk-title { font-size: 1.8rem; font-weight: 800; color: #fff; margin-bottom: 0.3rem; letter-spacing: 2px; }
    .risk-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.75); }

    /* Boxes */
    .recommendation-box { background: linear-gradient(135deg, #1e3a5f 0%, #1a365d 100%); border: 1px solid rgba(59,130,246,0.3); border-radius: 14px; padding: 1.5rem; color: #bfdbfe; font-size: 0.95rem; line-height: 1.7; margin: 1rem 0; }
    .info-box { background: linear-gradient(135deg, #1e3a5f 0%, #1a365d 100%); border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; padding: 1.2rem; color: #bfdbfe; font-size: 0.88rem; line-height: 1.6; }

    /* Comparison rows */
    .comp-row { display: flex; align-items: center; gap: 1rem; padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 0.4rem; }
    .comp-normal { background: rgba(16,185,129,0.1); border-left: 3px solid #10b981; }
    .comp-warning { background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444; }
    .comp-label { color: rgba(255,255,255,0.7); font-size: 0.82rem; min-width: 120px; }
    .comp-value { color: #fff; font-weight: 600; font-size: 0.95rem; }
    .comp-avg { color: rgba(255,255,255,0.5); font-size: 0.78rem; }
    .comp-badge { padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .badge-ok { background: rgba(16,185,129,0.2); color: #10b981; }
    .badge-warn { background: rgba(239,68,68,0.2); color: #ef4444; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%); }
    section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 { color: #c7d2fe; }
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li { color: rgba(255,255,255,0.75); }

    /* Disclaimer */
    .disclaimer { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 1rem; color: #fcd34d; font-size: 0.8rem; text-align: center; margin-top: 2rem; }

    /* ECG Animation */
    .ecg-monitor {
        background: #0a0a0a; border: 2px solid #1a3a2a; border-radius: 12px;
        padding: 1rem; position: relative; overflow: hidden;
    }
    .ecg-monitor::before {
        content: '● LIVE'; position: absolute; top: 8px; right: 12px;
        color: #10b981; font-size: 0.65rem; font-weight: 700;
        animation: blink 1.2s infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* Hide defaults */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "lang" not in st.session_state:
    st.session_state.lang = "English"


# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────

with st.sidebar:
    # Language selector
    st.markdown("### 🌐 Language / भाषा / భాష")
    lang = st.selectbox(
        "Select Language",
        options=list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.lang),
        key="lang_select",
        label_visibility="collapsed",
    )
    st.session_state.lang = lang
    T = lambda key: get_text(lang, key)

    st.markdown("---")
    st.markdown(f"# {T('sidebar_title')}")
    st.markdown("---")

    st.markdown(
        """
        ### 🔬 Hybrid AI Approach

        **1. ECG Signal Analysis**
        - CNN-LSTM deep learning model
        - MIT-BIH Arrhythmia Database

        **2. Clinical Risk Assessment**
        - Random Forest classifier
        - Heart Disease dataset (1025+ records)

        ---

        ### ⚙️ Fusion Logic

        | ECG | Clinical | Result |
        |-----|----------|--------|
        | ⚠️  | ⚠️       | 🔴 HIGH |
        | ⚠️  | ✅       | 🟠 MEDIUM |
        | ✅  | ⚠️       | 🟠 MEDIUM |
        | ✅  | ✅       | 🟢 LOW |

        **Composite** = 40% ECG + 60% Clinical

        ---

        ### 📊 Models

        | Model | Type |
        |-------|------|
        | ECG | CNN-LSTM |
        | Clinical | Random Forest |

        ---

        ### ✨ Enhanced Features
        - 📄 PDF Report with charts
        - 🖥️ ECG Monitor animation
        - 👤 Patient vs Population
        - 📊 Feature Importance
        - 🌐 Multi-language (EN/HI/TE)

        ---
        *Final Year AI Healthcare — 2026*
        """
    )

    st.markdown("---")
    st.markdown(f'<div class="disclaimer">{T("disclaimer")}</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_MODEL_PATH = os.path.join(BASE_DIR, "ecg_model .h5")

T = lambda key: get_text(st.session_state.lang, key)


@st.cache_resource(show_spinner=False)
def get_ecg_model():
    return load_ecg_model(ECG_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_clinical_model():
    return load_clinical_model()


with st.spinner("🔄 Loading AI Models..."):
    ecg_model = get_ecg_model()
    clinical_model, clinical_scaler = get_clinical_model()


# ────────────────────────────────────────────────────────────
# HERO HEADER
# ────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero-header">
        <h1>{T('app_title')}</h1>
        <p class="hero-subtitle">{T('app_subtitle')}</p>
        <div class="hero-badges">
            <span class="hero-badge">🧠 CNN-LSTM</span>
            <span class="hero-badge">🌲 Random Forest</span>
            <span class="hero-badge">📊 MIT-BIH</span>
            <span class="hero-badge">❤️ Heart Disease</span>
            <span class="hero-badge">🔬 Hybrid Fusion</span>
            <span class="hero-badge">🌐 Multi-Language</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Top metric cards
col1, col2, col3, col4, col5 = st.columns(5)
cards = [("ECG Model", "CNN-LSTM"), ("Clinical Model", "RF"), ("ECG Input", "200 pts"), ("Clinical Features", "13"), ("Risk Levels", "3")]
for col, (label, value) in zip([col1, col2, col3, col4, col5], cards):
    with col:
        st.markdown(f'<div class="metric-card"><h3>{label}</h3><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SECTION 1: ECG UPLOAD
# ════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-header"><h2>{T("module1_title")}</h2></div>', unsafe_allow_html=True)
st.markdown(f'<div class="info-box">{T("ecg_instructions")}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab_upload, tab_manual = st.tabs(["📁 File Upload", "⌨️ Manual Input"])

signal = None
input_method = None

with tab_upload:
    uploaded_file = st.file_uploader("Choose ECG file", type=["csv", "npy"], key="file_uploader")
    if uploaded_file is not None:
        try:
            signal = read_uploaded_file(uploaded_file)
            input_method = "file"
            st.success(f"✅ **{uploaded_file.name}** loaded ({signal.shape[0]} samples)")
        except ValueError as e:
            st.error(f"❌ {e}")

with tab_manual:
    st.markdown("Enter **200 ECG values** (comma or space separated):")
    manual_text = st.text_area("ECG Values", height=120, placeholder="0.123, -0.456, 0.789, ...", key="manual_input")
    if st.button("📥 Load", key="load_manual"):
        if manual_text.strip():
            try:
                signal = parse_manual_input(manual_text)
                input_method = "manual"
                st.success(f"✅ Parsed {signal.shape[0]} samples")
            except ValueError as e:
                st.error(f"❌ {e}")
        else:
            st.warning("⚠️ Enter ECG values first.")


# ── ECG Visualization with Hospital Monitor Animation ──
if signal is not None:
    st.markdown(f'<div class="section-header"><h2>{T("ecg_viz_title")}</h2></div>', unsafe_allow_html=True)

    # ★ Hospital-style ECG monitor with scan line animation
    st.markdown('<div class="ecg-monitor">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    x_axis = np.arange(len(signal))

    # ECG grid (hospital monitor style)
    for y_line in np.arange(np.floor(np.min(signal)), np.ceil(np.max(signal)) + 0.5, 0.5):
        ax.axhline(y=y_line, color="#0d3320", linewidth=0.3, alpha=0.5)
    for x_line in np.arange(0, len(signal), 10):
        ax.axvline(x=x_line, color="#0d3320", linewidth=0.3, alpha=0.5)

    # Major grid
    for y_line in np.arange(np.floor(np.min(signal)), np.ceil(np.max(signal)) + 1, 1.0):
        ax.axhline(y=y_line, color="#0d4428", linewidth=0.6, alpha=0.6)
    for x_line in np.arange(0, len(signal), 50):
        ax.axvline(x=x_line, color="#0d4428", linewidth=0.6, alpha=0.6)

    # ECG trace — bright green like hospital monitor
    ax.plot(x_axis, signal, color="#00ff41", linewidth=1.8, alpha=0.95)

    # Glow effect (thicker, transparent)
    ax.plot(x_axis, signal, color="#00ff41", linewidth=4, alpha=0.15)

    ax.set_xlabel("Sample Index", color="#2d6a3f", fontsize=10, fontweight="500")
    ax.set_ylabel("Amplitude", color="#2d6a3f", fontsize=10, fontweight="500")
    ax.set_title("ECG Monitor — Live Signal", color="#00ff41", fontsize=13, fontweight="700", pad=12)
    ax.tick_params(colors="#2d6a3f", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#0d3320")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # Signal stats
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.metric("📐 Samples", f"{len(signal)}")
    with sc2: st.metric("📈 Max", f"{np.max(signal):.4f}")
    with sc3: st.metric("📉 Min", f"{np.min(signal):.4f}")
    with sc4: st.metric("📊 Mean", f"{np.mean(signal):.4f}")


# ════════════════════════════════════════════════════════════
# SECTION 2: CLINICAL INPUT FORM
# ════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-header"><h2>{T("module2_title")}</h2></div>', unsafe_allow_html=True)
st.markdown(f'<div class="info-box">{T("clinical_instructions")}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: age = st.number_input(T("age"), min_value=1, max_value=120, value=50, step=1)
with c2: sex = st.selectbox(T("sex"), options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
with c3: cp = st.selectbox(T("cp"), options=[("Typical Angina (0)", 0), ("Atypical Angina (1)", 1), ("Non-anginal Pain (2)", 2), ("Asymptomatic (3)", 3)], format_func=lambda x: x[0])

c4, c5, c6 = st.columns(3)
with c4: trestbps = st.number_input(T("trestbps"), min_value=60, max_value=250, value=130, step=1)
with c5: chol = st.number_input(T("chol"), min_value=100, max_value=600, value=240, step=1)
with c6: fbs = st.selectbox(T("fbs"), options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])

c7, c8, c9 = st.columns(3)
with c7: restecg = st.selectbox(T("restecg"), options=[("Normal (0)", 0), ("ST-T Abnormality (1)", 1), ("LV Hypertrophy (2)", 2)], format_func=lambda x: x[0])
with c8: thalach = st.number_input(T("thalach"), min_value=60, max_value=220, value=150, step=1)
with c9: exang = st.selectbox(T("exang"), options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])

c10, c11, c12 = st.columns(3)
with c10: oldpeak = st.number_input(T("oldpeak"), min_value=0.0, max_value=7.0, value=1.0, step=0.1)
with c11: slope = st.selectbox(T("slope"), options=[("Downsloping (0)", 0), ("Flat (1)", 1), ("Upsloping (2)", 2)], format_func=lambda x: x[0])
with c12: ca = st.selectbox(T("ca"), options=[0, 1, 2, 3, 4])

c13, _, _ = st.columns(3)
with c13: thal = st.selectbox(T("thal"), options=[("Normal (0)", 0), ("Fixed Defect (1)", 1), ("Reversable Defect (2)", 2), ("Not Described (3)", 3)], format_func=lambda x: x[0])

clinical_inputs = {
    "age": age, "sex": sex[1], "cp": cp[1], "trestbps": trestbps,
    "chol": chol, "fbs": fbs[1], "restecg": restecg[1], "thalach": thalach,
    "exang": exang[1], "oldpeak": oldpeak, "slope": slope[1], "ca": ca, "thal": thal[1],
}


# ════════════════════════════════════════════════════════════
# SECTION 3: RUN PREDICTION
# ════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-header"><h2>{T("predict_title")}</h2></div>', unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1: run_hybrid = st.button(T("run_hybrid"), type="primary", use_container_width=True, key="run_hybrid_btn")
with p2: run_clinical_only = st.button(T("run_clinical"), use_container_width=True, key="run_clinical_btn")


# ── Gauge Chart Helper ──
def draw_gauge_chart(score_percent, title="Composite Risk Score"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

    theta_bg = np.linspace(np.pi, 0, 200); r = 1.0
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg), color="#334155", linewidth=18, solid_capstyle="round")

    score_frac = score_percent / 100.0
    theta_sc = np.linspace(np.pi, np.pi - score_frac * np.pi, 200)
    color = "#10b981" if score_percent < 35 else "#f59e0b" if score_percent < 65 else "#ef4444"
    ax.plot(r * np.cos(theta_sc), r * np.sin(theta_sc), color=color, linewidth=18, solid_capstyle="round")

    na = np.pi - score_frac * np.pi
    ax.annotate("", xy=(0.75 * np.cos(na), 0.75 * np.sin(na)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#e2e8f0", lw=2.5))
    ax.plot(0, 0, "o", color="#e2e8f0", markersize=8, zorder=5)

    ax.text(0, -0.15, f"{score_percent}%", ha="center", va="center", fontsize=28, fontweight="bold", color="#fff")
    ax.text(0, -0.38, title, ha="center", va="center", fontsize=11, color="#94a3b8", fontweight="500")
    ax.text(-1.05, -0.08, "Low", ha="center", fontsize=9, color="#10b981", fontweight="600")
    ax.text(0, 1.1, "Medium", ha="center", fontsize=9, color="#f59e0b", fontweight="600")
    ax.text(1.05, -0.08, "High", ha="center", fontsize=9, color="#ef4444", fontweight="600")

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.5, 1.3); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    return fig


# ── Feature Importance Chart ──
def draw_feature_importance_chart(clinical_model):
    importances = get_feature_importance(clinical_model)
    labels = [x[0] for x in importances]
    values = [x[1] for x in importances]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

    colors = []
    for i, v in enumerate(values):
        if i < 3: colors.append("#6366f1")
        elif i < 7: colors.append("#8b5cf6")
        else: colors.append("#a78bfa")

    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85, height=0.6, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#e2e8f0", fontsize=9, fontweight="600")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color="#c7d2fe")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", color="#94a3b8", fontsize=10)
    ax.set_title("Random Forest — Feature Importance Ranking", color="#e2e8f0", fontsize=13, fontweight="700", pad=12)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="x", alpha=0.15, color="#475569", linestyle="--")
    plt.tight_layout()
    return fig


# ── Patient Comparison Chart ──
def draw_comparison_chart(comparisons):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

    labels = [c["feature"] for c in comparisons]
    patient_vals = [c["value"] for c in comparisons]
    pop_means = [c["mean"] for c in comparisons]
    normal_lows = [c["normal_low"] for c in comparisons]
    normal_highs = [c["normal_high"] for c in comparisons]

    x = np.arange(len(labels))
    width = 0.3

    # Normal range band
    for i in range(len(labels)):
        ax.fill_between([i - 0.4, i + 0.4], normal_lows[i], normal_highs[i], alpha=0.08, color="#22d3ee")

    bars1 = ax.bar(x - width / 2, patient_vals, width, label="Your Value", color="#6366f1", alpha=0.9, edgecolor="none", zorder=3)
    bars2 = ax.bar(x + width / 2, pop_means, width, label="Population Avg", color="#475569", alpha=0.7, edgecolor="none", zorder=3)

    # Status dots on patient bars
    for i, c in enumerate(comparisons):
        dot_color = "#10b981" if c["in_range"] else "#ef4444"
        ax.plot(i - width / 2, patient_vals[i] + (max(patient_vals) * 0.03), "o", color=dot_color, markersize=6, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color="#c7d2fe")
    ax.set_ylabel("Value", color="#94a3b8", fontsize=10)
    ax.set_title("Patient Values vs Population Statistics", color="#e2e8f0", fontsize=13, fontweight="700", pad=12)
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper right")
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="y", alpha=0.15, color="#475569", linestyle="--")
    plt.tight_layout()
    return fig


# ────────────────────────────────────────────────────
# FULL HYBRID ANALYSIS
# ────────────────────────────────────────────────────

if run_hybrid:
    if signal is None:
        st.error("❌ Upload ECG data first for hybrid analysis.")
    else:
        try:
            progress = st.progress(0); status = st.empty()
            for i in range(100):
                time.sleep(0.012); progress.progress(i + 1)
                if i < 20: status.text("🔄 Preprocessing ECG...")
                elif i < 45: status.text("🧠 ECG model inference...")
                elif i < 70: status.text("🩺 Clinical analysis...")
                elif i < 90: status.text("🔬 Computing fusion...")
                else: status.text("📊 Generating results...")

            processed = preprocess_signal(signal)
            ecg_result = predict_ecg(ecg_model, processed)
            clinical_result = predict_clinical(clinical_model, clinical_scaler, clinical_inputs)
            fusion_result = compute_fusion_risk(ecg_result, clinical_result)

            progress.empty(); status.empty()

            st.session_state["last_ecg_result"] = ecg_result
            st.session_state["last_clinical_result"] = clinical_result
            st.session_state["last_fusion_result"] = fusion_result
            st.session_state["last_clinical_inputs"] = clinical_inputs
            st.session_state["last_signal"] = signal

            st.session_state.prediction_history.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "Hybrid", "ecg_label": "Abnormal" if ecg_result["label"] == 1 else "Normal",
                "clinical_risk": f"{clinical_result['risk_percent']}%",
                "final_risk": fusion_result["risk_level"], "composite": f"{fusion_result['composite_percent']}%",
            })

            # ── RESULTS DASHBOARD ──
            st.markdown(f'<div class="section-header"><h2>{T("results_title")}</h2></div>', unsafe_allow_html=True)

            # Risk card
            risk_class = f"risk-card-{fusion_result['risk_level'].lower()}"
            st.markdown(
                f'<div class="{risk_class}"><div class="risk-title">{fusion_result["risk_emoji"]} {fusion_result["risk_level"]} RISK</div>'
                f'<div class="risk-subtitle">Composite Score: {fusion_result["composite_percent"]}%</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Side-by-side ECG + Clinical
            r1, r2 = st.columns(2)
            with r1:
                cls = "result-normal" if ecg_result["label"] == 0 else "result-abnormal"
                st.markdown(f'<div class="{cls}"><div class="result-label">{ecg_result["class_name"]}</div><div class="result-confidence">Confidence: {ecg_result["confidence"]}%</div></div>', unsafe_allow_html=True)
            with r2:
                cls = "result-normal" if clinical_result["label"] == 0 else "result-abnormal"
                st.markdown(f'<div class="{cls}"><div class="result-label">{clinical_result["class_name"]}</div><div class="result-confidence">Risk: {clinical_result["risk_percent"]}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge + Details
            gc, dc = st.columns([1, 1])
            with gc:
                st.pyplot(draw_gauge_chart(fusion_result["composite_percent"]))
            with dc:
                st.markdown("#### 📋 Detailed Breakdown")
                st.metric("🧠 ECG Score", f"{ecg_result['probability']:.4f}")
                st.metric("🩺 Clinical Risk", f"{clinical_result['risk_percent']}%")
                st.metric("🔬 Composite", f"{fusion_result['composite_percent']}%")

            # ★ PATIENT vs POPULATION COMPARISON
            st.markdown(f'<div class="section-header"><h2>{T("comparison_title")}</h2></div>', unsafe_allow_html=True)

            comparisons = get_patient_comparison(clinical_inputs)

            # Visual comparison chart
            comp_fig = draw_comparison_chart(comparisons)
            st.pyplot(comp_fig)
            plt.close(comp_fig)

            # Detailed comparison rows
            for c in comparisons:
                row_cls = "comp-normal" if c["in_range"] else "comp-warning"
                badge_cls = "badge-ok" if c["in_range"] else "badge-warn"
                st.markdown(
                    f'<div class="comp-row {row_cls}">'
                    f'<span class="comp-label">{c["feature"]}</span>'
                    f'<span class="comp-value">{c["value"]} {c["unit"]}</span>'
                    f'<span class="comp-avg">Avg: {c["mean"]} {c["unit"]} • Range: {c["normal_low"]}-{c["normal_high"]}</span>'
                    f'<span class="comp-avg">Percentile: {c["percentile"]}th</span>'
                    f'<span class="comp-badge {badge_cls}">{c["status"]}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ★ FEATURE IMPORTANCE
            st.markdown(f'<div class="section-header"><h2>{T("feature_title")}</h2></div>', unsafe_allow_html=True)
            fi_fig = draw_feature_importance_chart(clinical_model)
            st.pyplot(fi_fig)
            plt.close(fi_fig)

            # Recommendation
            st.markdown(
                f'<div class="recommendation-box"><b>{T("recommendation")}</b><br>{fusion_result["recommendation"]}</div>',
                unsafe_allow_html=True,
            )

        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")


# ────────────────────────────────────────────────────
# CLINICAL-ONLY ANALYSIS
# ────────────────────────────────────────────────────

if run_clinical_only:
    try:
        progress = st.progress(0); status = st.empty()
        for i in range(100):
            time.sleep(0.008); progress.progress(i + 1)
            if i < 50: status.text("🩺 Analyzing...")
            else: status.text("📊 Results...")

        clinical_result = predict_clinical(clinical_model, clinical_scaler, clinical_inputs)
        ecg_dummy = {"probability": 0.0, "label": 0, "class_name": "N/A", "confidence": 0}
        fusion_result = compute_fusion_risk(ecg_dummy, clinical_result)

        progress.empty(); status.empty()

        st.session_state["last_clinical_result"] = clinical_result
        st.session_state["last_fusion_result"] = fusion_result
        st.session_state["last_clinical_inputs"] = clinical_inputs

        st.session_state.prediction_history.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Clinical Only", "ecg_label": "N/A",
            "clinical_risk": f"{clinical_result['risk_percent']}%",
            "final_risk": fusion_result["risk_level"], "composite": f"{fusion_result['composite_percent']}%",
        })

        st.markdown('<div class="section-header"><h2>📊 Clinical Analysis Results</h2></div>', unsafe_allow_html=True)

        cls = "result-normal" if clinical_result["label"] == 0 else "result-abnormal"
        st.markdown(f'<div class="{cls}"><div class="result-label">{clinical_result["class_name"]}</div><div class="result-confidence">Risk: {clinical_result["risk_percent"]}%</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        gc, dc = st.columns([1, 1])
        with gc: st.pyplot(draw_gauge_chart(clinical_result["risk_percent"], "Clinical Risk Score"))
        with dc:
            st.markdown("#### 📋 Details")
            st.metric("🩺 Risk", f"{clinical_result['risk_percent']}%")
            st.metric("🏷️ Class", "Heart Disease Risk" if clinical_result["label"] == 1 else "Low Risk")
            st.info("💡 Upload ECG for **Full Hybrid Analysis**.")

        # Comparison
        st.markdown(f'<div class="section-header"><h2>{T("comparison_title")}</h2></div>', unsafe_allow_html=True)
        comparisons = get_patient_comparison(clinical_inputs)
        st.pyplot(draw_comparison_chart(comparisons))
        for c in comparisons:
            row_cls = "comp-normal" if c["in_range"] else "comp-warning"
            badge_cls = "badge-ok" if c["in_range"] else "badge-warn"
            st.markdown(
                f'<div class="comp-row {row_cls}"><span class="comp-label">{c["feature"]}</span>'
                f'<span class="comp-value">{c["value"]} {c["unit"]}</span>'
                f'<span class="comp-avg">Avg: {c["mean"]} • Range: {c["normal_low"]}-{c["normal_high"]}</span>'
                f'<span class="comp-badge {badge_cls}">{c["status"]}</span></div>',
                unsafe_allow_html=True,
            )

        # Feature importance
        st.markdown(f'<div class="section-header"><h2>{T("feature_title")}</h2></div>', unsafe_allow_html=True)
        st.pyplot(draw_feature_importance_chart(clinical_model))

        st.markdown(f'<div class="recommendation-box"><b>{T("recommendation")}</b><br>{fusion_result["recommendation"]}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ {e}")


# ════════════════════════════════════════════════════════════
# SECTION 4: DOWNLOAD REPORT (PDF + TXT)
# ════════════════════════════════════════════════════════════

if "last_fusion_result" in st.session_state:
    st.markdown(f'<div class="section-header"><h2>{T("download_title")}</h2></div>', unsafe_allow_html=True)

    ecg_res = st.session_state.get("last_ecg_result", None)
    clin_res = st.session_state.get("last_clinical_result")
    fusion_res = st.session_state.get("last_fusion_result")
    clin_inp = st.session_state.get("last_clinical_inputs", {})
    sig = st.session_state.get("last_signal", None)

    dl1, dl2 = st.columns(2)

    with dl1:
        # ★ PDF Report
        try:
            pdf_bytes = generate_pdf_report(
                ecg_result=ecg_res, clinical_result=clin_res,
                fusion_result=fusion_res, clinical_inputs=clin_inp,
                signal=sig, feature_importances=get_feature_importance(clinical_model),
            )
            st.download_button(
                label=T("download_pdf"), data=pdf_bytes,
                file_name="heart_risk_report.pdf", mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation error: {e}. Text report available.")

    with dl2:
        report_text = generate_hybrid_report(
            ecg_result=ecg_res, clinical_result=clin_res,
            fusion_result=fusion_res, clinical_inputs=clin_inp, signal=sig,
        )
        st.download_button(
            label=T("download_txt"), data=report_text,
            file_name="heart_risk_report.txt", mime="text/plain",
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════
# SECTION 5: PREDICTION HISTORY
# ════════════════════════════════════════════════════════════

if st.session_state.prediction_history:
    st.markdown(f'<div class="section-header"><h2>{T("history_title")}</h2></div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df.columns = ["Time", "Mode", "ECG Result", "Clinical Risk", "Final Risk", "Composite"]
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.prediction_history = []
        st.rerun()


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">{T("disclaimer")}</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; padding:2rem 0 1rem; color:rgba(255,255,255,0.3); font-size:0.78rem;">'
    'AI-Based Heart Risk Prediction System • CNN-LSTM + Random Forest • TensorFlow & Scikit-learn & Streamlit<br>'
    '© 2026 — Final Year AI Healthcare Project</div>',
    unsafe_allow_html=True,
)
