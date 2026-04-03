# ============================================================
# utils.py — Preprocessing & Prediction Helper Functions
# ============================================================
# Extended module providing utilities for:
#   • ECG signal analysis  (CNN-LSTM model)
#   • Clinical risk prediction  (Random Forest model)
#   • Hybrid fusion logic
#   • Report generation (PDF + text)
#   • Patient comparison vs population statistics
#   • Feature importance analysis
# ============================================================

import os
import io
import pickle
import datetime
import tempfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from tensorflow.keras.models import load_model


# ────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────

EXPECTED_LENGTH = 200      # ECG model input length
ECG_THRESHOLD = 0.5        # Decision boundary for ECG classification
CLINICAL_THRESHOLD = 0.5   # Decision boundary for clinical risk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ────────────────────────────────────────────────────────────
# MULTI-LANGUAGE SUPPORT
# ────────────────────────────────────────────────────────────

TRANSLATIONS = {
    "English": {
        "app_title": "🫀 AI-Based Heart Risk Prediction System",
        "app_subtitle": "Hybrid Deep Learning + Machine Learning • ECG Signal + Clinical Data Analysis",
        "sidebar_title": "🫀 About This System",
        "module1_title": "📤 Module 1 — ECG Signal Upload",
        "module2_title": "🏥 Module 2 — Clinical Data Input",
        "predict_title": "🚀 Run Hybrid Prediction",
        "results_title": "📊 Results Dashboard",
        "download_title": "📥 Download Report",
        "history_title": "📜 Prediction History",
        "comparison_title": "👤 Patient vs Population Comparison",
        "feature_title": "📊 Feature Importance Analysis",
        "ecg_viz_title": "📊 ECG Signal Visualization",
        "run_hybrid": "🔬 Run Full Hybrid Analysis",
        "run_clinical": "🩺 Run Clinical-Only Analysis",
        "ecg_instructions": "📋 <b>Instructions:</b>  Upload a <code>.csv</code> or <code>.npy</code> file containing exactly <b>200 ECG signal samples</b> (one heartbeat segment), or enter values manually.",
        "clinical_instructions": "🩺 <b>Patient Information:</b>  Enter the patient's clinical data below. All fields are required for accurate risk prediction.",
        "risk_high": "HIGH RISK",
        "risk_medium": "MEDIUM RISK",
        "risk_low": "LOW RISK",
        "recommendation": "💡 Recommendation:",
        "disclaimer": "⚠️ <b>Medical Disclaimer:</b> This application is developed for educational and research purposes only. It is NOT intended for clinical use and should NOT be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.",
        "age": "🎂 Age (years)",
        "sex": "⚧ Sex",
        "cp": "💢 Chest Pain Type",
        "trestbps": "🩸 Resting Blood Pressure (mm Hg)",
        "chol": "🧪 Serum Cholesterol (mg/dl)",
        "fbs": "🍬 Fasting Blood Sugar > 120 mg/dl",
        "restecg": "📈 Resting ECG Results",
        "thalach": "💓 Max Heart Rate Achieved (bpm)",
        "exang": "🏃 Exercise Induced Angina",
        "oldpeak": "📉 ST Depression (oldpeak)",
        "slope": "📐 ST Slope",
        "ca": "🫀 Major Vessels (0-4)",
        "thal": "🧬 Thalassemia",
        "patient_value": "Your Value",
        "population_avg": "Population Avg",
        "normal_range": "Normal Range",
        "status": "Status",
        "download_pdf": "📄 Download PDF Report",
        "download_txt": "📝 Download Text Report",
    },
    "हिंदी": {
        "app_title": "🫀 एआई-आधारित हृदय जोखिम भविष्यवाणी प्रणाली",
        "app_subtitle": "हाइब्रिड डीप लर्निंग + मशीन लर्निंग • ईसीजी सिग्नल + क्लिनिकल डेटा विश्लेषण",
        "sidebar_title": "🫀 इस प्रणाली के बारे में",
        "module1_title": "📤 मॉड्यूल 1 — ईसीजी सिग्नल अपलोड",
        "module2_title": "🏥 मॉड्यूल 2 — क्लिनिकल डेटा इनपुट",
        "predict_title": "🚀 हाइब्रिड भविष्यवाणी चलाएं",
        "results_title": "📊 परिणाम डैशबोर्ड",
        "download_title": "📥 रिपोर्ट डाउनलोड करें",
        "history_title": "📜 भविष्यवाणी इतिहास",
        "comparison_title": "👤 रोगी बनाम जनसंख्या तुलना",
        "feature_title": "📊 फीचर महत्व विश्लेषण",
        "ecg_viz_title": "📊 ईसीजी सिग्नल विज़ुअलाइज़ेशन",
        "run_hybrid": "🔬 पूर्ण हाइब्रिड विश्लेषण चलाएं",
        "run_clinical": "🩺 केवल क्लिनिकल विश्लेषण चलाएं",
        "ecg_instructions": "📋 <b>निर्देश:</b> ठीक <b>200 ईसीजी सिग्नल सैंपल</b> वाली <code>.csv</code> या <code>.npy</code> फ़ाइल अपलोड करें, या मैन्युअल रूप से मान दर्ज करें।",
        "clinical_instructions": "🩺 <b>रोगी जानकारी:</b> सटीक जोखिम भविष्यवाणी के लिए नीचे रोगी का क्लिनिकल डेटा दर्ज करें।",
        "risk_high": "उच्च जोखिम",
        "risk_medium": "मध्यम जोखिम",
        "risk_low": "कम जोखिम",
        "recommendation": "💡 सिफारिश:",
        "disclaimer": "⚠️ <b>चिकित्सा अस्वीकरण:</b> यह एप्लिकेशन केवल शैक्षिक और अनुसंधान उद्देश्यों के लिए है। यह पेशेवर चिकित्सा निदान का विकल्प नहीं है।",
        "age": "🎂 आयु (वर्ष)",
        "sex": "⚧ लिंग",
        "cp": "💢 छाती में दर्द का प्रकार",
        "trestbps": "🩸 आराम रक्तचाप (mm Hg)",
        "chol": "🧪 सीरम कोलेस्ट्रॉल (mg/dl)",
        "fbs": "🍬 उपवास रक्त शर्करा > 120 mg/dl",
        "restecg": "📈 आराम ईसीजी परिणाम",
        "thalach": "💓 अधिकतम हृदय गति (bpm)",
        "exang": "🏃 व्यायाम प्रेरित एनजाइना",
        "oldpeak": "📉 ST अवसाद (oldpeak)",
        "slope": "📐 ST ढलान",
        "ca": "🫀 प्रमुख वाहिकाएं (0-4)",
        "thal": "🧬 थैलेसीमिया",
        "patient_value": "आपका मान",
        "population_avg": "जनसंख्या औसत",
        "normal_range": "सामान्य सीमा",
        "status": "स्थिति",
        "download_pdf": "📄 PDF रिपोर्ट डाउनलोड करें",
        "download_txt": "📝 टेक्स्ट रिपोर्ट डाउनलोड करें",
    },
    "తెలుగు": {
        "app_title": "🫀 AI-ఆధారిత గుండె ప్రమాద అంచనా వ్యవస్థ",
        "app_subtitle": "హైబ్రిడ్ డీప్ లెర్నింగ్ + మెషిన్ లెర్నింగ్ • ECG సిగ్నల్ + క్లినికల్ డేటా విశ్లేషణ",
        "sidebar_title": "🫀 ఈ వ్యవస్థ గురించి",
        "module1_title": "📤 మాడ్యూల్ 1 — ECG సిగ్నల్ అప్‌లోడ్",
        "module2_title": "🏥 మాడ్యూల్ 2 — క్లినికల్ డేటా ఇన్‌పుట్",
        "predict_title": "🚀 హైబ్రిడ్ అంచనా అమలు",
        "results_title": "📊 ఫలితాల డాష్‌బోర్డ్",
        "download_title": "📥 నివేదిక డౌన్‌లోడ్",
        "history_title": "📜 అంచనా చరిత్ర",
        "comparison_title": "👤 రోగి vs జనాభా పోలిక",
        "feature_title": "📊 ఫీచర్ ప్రాముఖ్యత విశ్లేషణ",
        "ecg_viz_title": "📊 ECG సిగ్నల్ విజువలైజేషన్",
        "run_hybrid": "🔬 పూర్తి హైబ్రిడ్ విశ్లేషణ",
        "run_clinical": "🩺 క్లినికల్ మాత్రమే విశ్లేషణ",
        "ecg_instructions": "📋 <b>సూచనలు:</b> ఖచ్చితంగా <b>200 ECG సిగ్నల్ నమూనాలు</b> ఉన్న ఫైల్‌ను అప్‌లోడ్ చేయండి.",
        "clinical_instructions": "🩺 <b>రోగి సమాచారం:</b> ఖచ్చితమైన ప్రమాద అంచనా కోసం క్లినికల్ డేటా నమోదు చేయండి.",
        "risk_high": "అధిక ప్రమాదం",
        "risk_medium": "మధ్యస్థ ప్రమాదం",
        "risk_low": "తక్కువ ప్రమాదం",
        "recommendation": "💡 సిఫారసు:",
        "disclaimer": "⚠️ <b>వైద్య నిరాకరణ:</b> ఈ అప్లికేషన్ విద్యా మరియు పరిశోధన ప్రయోజనాల కోసం మాత్రమే.",
        "age": "🎂 వయస్సు (సంవత్సరాలు)",
        "sex": "⚧ లింగం",
        "cp": "💢 ఛాతీ నొప్పి రకం",
        "trestbps": "🩸 విశ్రాంతి రక్తపోటు (mm Hg)",
        "chol": "🧪 సీరం కొలెస్ట్రాల్ (mg/dl)",
        "fbs": "🍬 ఉపవాస రక్త చక్కెర > 120 mg/dl",
        "restecg": "📈 విశ్రాంతి ECG ఫలితాలు",
        "thalach": "💓 గరిష్ట హృదయ స్పందన (bpm)",
        "exang": "🏃 వ్యాయామ ప్రేరిత ఆంజైనా",
        "oldpeak": "📉 ST డిప్రెషన్ (oldpeak)",
        "slope": "📐 ST వాలు",
        "ca": "🫀 ప్రధాన నాళాలు (0-4)",
        "thal": "🧬 తాలసేమియా",
        "patient_value": "మీ విలువ",
        "population_avg": "జనాభా సగటు",
        "normal_range": "సాధారణ పరిధి",
        "status": "స్థితి",
        "download_pdf": "📄 PDF నివేదిక డౌన్‌లోడ్",
        "download_txt": "📝 టెక్స్ట్ నివేదిక డౌన్‌లోడ్",
    },
}


def get_text(lang: str, key: str) -> str:
    """Get translated text for the given language and key."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, TRANSLATIONS["English"].get(key, key))


# ────────────────────────────────────────────────────────────
# POPULATION STATISTICS (from Heart Disease dataset)
# ────────────────────────────────────────────────────────────

POPULATION_STATS = {
    "age":      {"mean": 54.4, "std": 9.1,  "min": 29, "max": 77,  "normal_low": 30, "normal_high": 70,  "unit": "years",  "label": "Age"},
    "trestbps": {"mean": 131.6, "std": 17.5, "min": 94, "max": 200, "normal_low": 90, "normal_high": 140, "unit": "mm Hg",  "label": "Resting BP"},
    "chol":     {"mean": 246.0, "std": 51.6, "min": 126,"max": 564, "normal_low": 125,"normal_high": 200, "unit": "mg/dl",  "label": "Cholesterol"},
    "thalach":  {"mean": 149.6, "std": 22.9, "min": 71, "max": 202, "normal_low": 100,"normal_high": 190, "unit": "bpm",    "label": "Max Heart Rate"},
    "oldpeak":  {"mean": 1.07,  "std": 1.18, "min": 0,  "max": 6.2, "normal_low": 0,  "normal_high": 2.0, "unit": "",       "label": "ST Depression"},
}


def get_patient_comparison(clinical_inputs: dict) -> list:
    """
    Compare patient values against population statistics.

    Returns a list of dicts with comparison data for each continuous feature.
    """
    comparisons = []
    for feat, stats in POPULATION_STATS.items():
        val = clinical_inputs.get(feat, 0)
        in_range = stats["normal_low"] <= val <= stats["normal_high"]

        # Percentile approximation using z-score
        z = (val - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
        # Approximate percentile from z-score (simplified)
        from math import erf, sqrt
        percentile = 50 * (1 + erf(z / sqrt(2)))

        comparisons.append({
            "feature": stats["label"],
            "value": val,
            "unit": stats["unit"],
            "mean": stats["mean"],
            "normal_low": stats["normal_low"],
            "normal_high": stats["normal_high"],
            "in_range": in_range,
            "percentile": round(percentile, 1),
            "status": "✅ Normal" if in_range else "⚠️ Outside Range",
        })
    return comparisons


# ────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG", "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "ST Slope", "ca": "Major Vessels",
    "thal": "Thalassemia",
}


def get_feature_importance(model) -> list:
    """
    Extract feature importance from the Random Forest model.

    Returns sorted list of (feature_label, importance) tuples.
    """
    with open(os.path.join(MODELS_DIR, "clinical_features.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    importances = model.feature_importances_
    result = []
    for name, imp in zip(feature_names, importances):
        result.append((FEATURE_LABELS.get(name, name), round(imp, 4)))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ════════════════════════════════════════════════════════════
#  SECTION 1:  ECG MODEL UTILITIES
# ════════════════════════════════════════════════════════════

def load_ecg_model(model_path: str):
    """Load the pre-trained ECG classification model from an .h5 file."""
    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def read_uploaded_file(uploaded_file):
    """Read an uploaded file (.csv or .npy) and return a 1-D numpy array."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=None)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        signal = df.values.flatten().astype(np.float64)

    elif filename.endswith(".npy"):
        signal = np.load(uploaded_file).astype(np.float64)
        signal = signal.flatten()

    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. Please upload a .csv or .npy file."
        )

    if signal.size == 0:
        raise ValueError("The uploaded file contains no data.")

    return signal


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Preprocess raw ECG signal: validate, z-score, reshape to (1,200,1)."""
    if signal.shape[0] != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected exactly {EXPECTED_LENGTH} data points, "
            f"but received {signal.shape[0]}."
        )

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0:
        raise ValueError("Signal has zero standard deviation.")
    normalized = (signal - mean_val) / std_val
    processed = normalized.reshape(1, EXPECTED_LENGTH, 1)
    return processed


def predict_ecg(model, processed_signal: np.ndarray) -> dict:
    """Run inference on a preprocessed ECG signal."""
    # Use direct call instead of .predict() to prevent Streamlit thread deadlocks
    prediction = model(processed_signal, training=False).numpy()
    probability = float(prediction[0][0])
    label = 1 if probability >= ECG_THRESHOLD else 0

    if label == 1:
        class_name = "⚠️ Abnormal Heartbeat (Potential Risk)"
        confidence = probability * 100
    else:
        class_name = "✅ Normal Heartbeat"
        confidence = (1 - probability) * 100

    return {
        "probability": probability,
        "label": label,
        "class_name": class_name,
        "confidence": round(confidence, 2),
    }


def parse_manual_input(text: str) -> np.ndarray:
    """Parse comma/space separated numbers into a 1-D numpy array."""
    text = text.strip().replace(",", " ")
    parts = text.split()
    if len(parts) == 0:
        raise ValueError("No values detected.")

    try:
        signal = np.array([float(v) for v in parts])
    except ValueError:
        raise ValueError("Could not parse all values as numbers.")
    return signal


# ════════════════════════════════════════════════════════════
#  SECTION 2:  CLINICAL MODEL UTILITIES
# ════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(BASE_DIR, "models")
CLINICAL_MODEL_PATH = os.path.join(MODELS_DIR, "clinical_rf_model.pkl")
CLINICAL_SCALER_PATH = os.path.join(MODELS_DIR, "clinical_scaler.pkl")
CLINICAL_FEATURES_PATH = os.path.join(MODELS_DIR, "clinical_features.pkl")


def load_clinical_model():
    """Load the trained clinical Random Forest model and its scaler."""
    if not os.path.exists(CLINICAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Clinical model not found at '{CLINICAL_MODEL_PATH}'. "
            "Run `python clinical_model.py` first."
        )

    with open(CLINICAL_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(CLINICAL_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_clinical(model, scaler, clinical_data: dict) -> dict:
    """Predict heart disease risk from clinical features."""
    with open(CLINICAL_FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    features = np.array([[clinical_data[feat] for feat in feature_names]])
    features_scaled = scaler.transform(features)
    probabilities = model.predict_proba(features_scaled)[0]
    risk_prob = float(probabilities[1])
    label = 1 if risk_prob >= CLINICAL_THRESHOLD else 0

    if label == 1:
        class_name = "⚠️ Elevated Heart Disease Risk"
    else:
        class_name = "✅ Low Heart Disease Risk"

    return {
        "probability": risk_prob,
        "label": label,
        "risk_percent": round(risk_prob * 100, 2),
        "class_name": class_name,
    }


# ════════════════════════════════════════════════════════════
#  SECTION 3:  HYBRID FUSION LOGIC
# ════════════════════════════════════════════════════════════

def compute_fusion_risk(ecg_result: dict, clinical_result: dict) -> dict:
    """Combine ECG and clinical predictions into a final risk assessment."""
    ecg_abnormal = ecg_result["label"] == 1
    clinical_abnormal = clinical_result["label"] == 1

    ecg_prob = ecg_result["probability"]
    clinical_prob = clinical_result["probability"]
    composite = 0.4 * ecg_prob + 0.6 * clinical_prob
    composite_percent = round(composite * 100, 2)

    if ecg_abnormal and clinical_abnormal:
        risk_level = "HIGH"
        risk_color = "#ef4444"
        risk_emoji = "🔴"
        recommendation = (
            "Both ECG analysis and clinical assessment indicate elevated risk. "
            "Immediate medical consultation is strongly recommended. "
            "Please consult a cardiologist for comprehensive evaluation."
        )
    elif ecg_abnormal or clinical_abnormal:
        risk_level = "MEDIUM"
        risk_color = "#f59e0b"
        risk_emoji = "🟠"
        recommendation = (
            "One of the two assessments shows elevated risk. "
            "Scheduling a follow-up with your healthcare provider is recommended. "
            "Additional tests may be needed for accurate diagnosis."
        )
    else:
        risk_level = "LOW"
        risk_color = "#10b981"
        risk_emoji = "🟢"
        recommendation = (
            "Both ECG and clinical assessments are within normal ranges. "
            "Continue maintaining a healthy lifestyle with regular check-ups. "
            "Monitor any new symptoms and consult a doctor if concerns arise."
        )

    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "composite_score": composite,
        "composite_percent": composite_percent,
        "ecg_abnormal": ecg_abnormal,
        "clinical_abnormal": clinical_abnormal,
        "recommendation": recommendation,
    }


# ════════════════════════════════════════════════════════════
#  SECTION 4:  REPORT GENERATION (TEXT)
# ════════════════════════════════════════════════════════════

def generate_report(result: dict, signal: np.ndarray) -> str:
    """Generate a text report for ECG-only prediction (legacy)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "  AI-BASED HEART ATTACK DETECTION SYSTEM",
        "  PREDICTION REPORT",
        "=" * 60, "",
        f"  Date & Time   : {now}",
        f"  Signal Length  : {len(signal)} samples", "",
        "-" * 60, "  PREDICTION RESULT", "-" * 60, "",
        f"  Classification : {result['class_name']}",
        f"  Confidence     : {result['confidence']}%",
        f"  Raw Probability: {result['probability']:.4f}", "",
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_hybrid_report(
    ecg_result, clinical_result, fusion_result,
    clinical_inputs, signal=None,
) -> str:
    """Generate comprehensive hybrid text report."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 65,
        "   AI-BASED HEART RISK PREDICTION SYSTEM",
        "   COMPREHENSIVE PREDICTION REPORT",
        "=" * 65, "",
        f"   Report Generated : {now}", "",
    ]

    # ECG
    lines += ["-" * 65, "   1. ECG SIGNAL ANALYSIS", "-" * 65, ""]
    if ecg_result is not None:
        lines += [
            f"   Classification   : {ecg_result['class_name']}",
            f"   Confidence       : {ecg_result['confidence']}%",
            f"   Raw Score        : {ecg_result['probability']:.4f}", "",
        ]
        if signal is not None:
            lines += [
                f"   Signal Length    : {len(signal)} samples",
                f"   Min Value       : {np.min(signal):.6f}",
                f"   Max Value       : {np.max(signal):.6f}",
                f"   Mean Value      : {np.mean(signal):.6f}", "",
            ]
    else:
        lines += ["   ECG data was not provided.", ""]

    # Clinical
    lines += ["-" * 65, "   2. CLINICAL DATA ASSESSMENT", "-" * 65, ""]
    lines += [
        "   Patient Information:",
        f"     Age               : {clinical_inputs.get('age', 'N/A')} years",
        f"     Sex               : {'Male' if clinical_inputs.get('sex', 0) == 1 else 'Female'}",
        f"     Chest Pain Type   : {clinical_inputs.get('cp', 'N/A')}",
        f"     Resting BP        : {clinical_inputs.get('trestbps', 'N/A')} mm Hg",
        f"     Cholesterol       : {clinical_inputs.get('chol', 'N/A')} mg/dl",
        f"     Fasting BS > 120  : {'Yes' if clinical_inputs.get('fbs', 0) == 1 else 'No'}",
        f"     Max Heart Rate    : {clinical_inputs.get('thalach', 'N/A')} bpm",
        f"     Exercise Angina   : {'Yes' if clinical_inputs.get('exang', 0) == 1 else 'No'}",
        f"     ST Depression     : {clinical_inputs.get('oldpeak', 'N/A')}",
        "",
        f"     Heart Disease Risk : {clinical_result['risk_percent']}%",
        f"     Classification     : {clinical_result['class_name']}", "",
    ]

    # Fusion
    lines += ["-" * 65, "   3. HYBRID RISK ASSESSMENT", "-" * 65, ""]
    lines += [
        f"   Final Risk Level     : {fusion_result['risk_level']}",
        f"   Composite Score      : {fusion_result['composite_percent']}%",
        f"   ECG Status           : {'Abnormal' if fusion_result['ecg_abnormal'] else 'Normal'}",
        f"   Clinical Status      : {'Elevated' if fusion_result['clinical_abnormal'] else 'Normal'}",
        "", f"   {fusion_result['recommendation']}", "",
    ]

    # Patient comparison
    comparisons = get_patient_comparison(clinical_inputs)
    lines += ["-" * 65, "   4. PATIENT vs POPULATION COMPARISON", "-" * 65, ""]
    for c in comparisons:
        lines.append(
            f"   {c['feature']:18s}: {c['value']:>7} {c['unit']:6s} "
            f"(Avg: {c['mean']}, Range: {c['normal_low']}-{c['normal_high']}) "
            f"[{c['status']}]"
        )
    lines += [""]

    # Disclaimer
    lines += [
        "=" * 65, "   DISCLAIMER", "=" * 65, "",
        "   For educational and research purposes only.",
        "   NOT a substitute for professional medical diagnosis.", "",
        "=" * 65,
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  SECTION 5:  PDF REPORT GENERATION
# ════════════════════════════════════════════════════════════

def _save_chart_to_temp(fig) -> str:
    """Save a matplotlib figure to a temp PNG file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return tmp.name


def _create_ecg_chart_for_pdf(signal):
    """Create a white-background ECG chart suitable for PDF embedding."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    x = np.arange(len(signal))
    ax.plot(x, signal, color="#2563eb", linewidth=1.2)
    ax.fill_between(x, signal, alpha=0.08, color="#2563eb")
    ax.set_xlabel("Sample Index", fontsize=8, color="#475569")
    ax.set_ylabel("Amplitude", fontsize=8, color="#475569")
    ax.set_title("ECG Signal Waveform", fontsize=10, fontweight="bold", color="#1e293b")
    ax.grid(True, alpha=0.3, color="#cbd5e1", linestyle="--")
    ax.tick_params(labelsize=7, colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    plt.tight_layout()
    return fig


def _create_gauge_for_pdf(score_percent):
    """Create a white-background gauge chart for PDF embedding."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    theta_bg = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg), color="#e2e8f0", linewidth=14, solid_capstyle="round")

    score_frac = score_percent / 100.0
    theta_sc = np.linspace(np.pi, np.pi - score_frac * np.pi, 200)
    if score_percent < 35:
        color = "#10b981"
    elif score_percent < 65:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    ax.plot(r * np.cos(theta_sc), r * np.sin(theta_sc), color=color, linewidth=14, solid_capstyle="round")

    needle_angle = np.pi - score_frac * np.pi
    ax.annotate("", xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2))
    ax.plot(0, 0, "o", color="#334155", markersize=6, zorder=5)

    ax.text(0, -0.15, f"{score_percent}%", ha="center", va="center", fontsize=20, fontweight="bold", color="#1e293b")
    ax.text(0, -0.35, "Composite Risk Score", ha="center", fontsize=8, color="#64748b")
    ax.text(-1.05, -0.08, "Low", ha="center", fontsize=7, color="#10b981", fontweight="bold")
    ax.text(0, 1.08, "Medium", ha="center", fontsize=7, color="#f59e0b", fontweight="bold")
    ax.text(1.05, -0.08, "High", ha="center", fontsize=7, color="#ef4444", fontweight="bold")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.5, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


def _create_comparison_chart_for_pdf(comparisons):
    """Create a horizontal bar chart comparing patient to population averages."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    labels = [c["feature"] for c in comparisons]
    patient_vals = [c["value"] for c in comparisons]
    pop_means = [c["mean"] for c in comparisons]

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height / 2, patient_vals, bar_height, label="Patient", color="#3b82f6", alpha=0.85)
    bars2 = ax.barh(y_pos + bar_height / 2, pop_means, bar_height, label="Population Avg", color="#94a3b8", alpha=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Value", fontsize=8, color="#475569")
    ax.set_title("Patient vs Population Comparison", fontsize=10, fontweight="bold", color="#1e293b")
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7, colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.grid(axis="x", alpha=0.3, color="#cbd5e1", linestyle="--")
    plt.tight_layout()
    return fig


def generate_pdf_report(
    ecg_result, clinical_result, fusion_result,
    clinical_inputs, signal=None, feature_importances=None,
) -> bytes:
    """
    Generate a professional PDF report with embedded charts.

    Returns PDF bytes ready for download.
    """
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, "AI-Based Heart Risk Prediction System", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(226, 232, 240)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | For Educational Purposes Only",
                      align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 12, "Heart Risk Prediction Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Report Date: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Risk Level Banner ──
    risk = fusion_result["risk_level"]
    if risk == "HIGH":
        pdf.set_fill_color(239, 68, 68)
    elif risk == "MEDIUM":
        pdf.set_fill_color(245, 158, 11)
    else:
        pdf.set_fill_color(16, 185, 129)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, f"  FINAL RISK LEVEL:  {risk}  |  Composite Score: {fusion_result['composite_percent']}%",
             fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Gauge Chart ──
    gauge_fig = _create_gauge_for_pdf(fusion_result["composite_percent"])
    gauge_path = _save_chart_to_temp(gauge_fig)
    pdf.image(gauge_path, x=55, w=100)
    os.unlink(gauge_path)
    pdf.ln(4)

    # ── ECG Results ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "1. ECG Signal Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if ecg_result is not None:
        ecg_label = "Abnormal" if ecg_result["label"] == 1 else "Normal"
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(50, 7, "Classification:", new_x="RIGHT")
        pdf.set_font("Helvetica", "B", 10)
        if ecg_result["label"] == 1:
            pdf.set_text_color(239, 68, 68)
        else:
            pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 7, ecg_label, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(50, 7, "Confidence:", new_x="RIGHT")
        pdf.cell(0, 7, f"{ecg_result['confidence']}%", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(50, 7, "Raw Score:", new_x="RIGHT")
        pdf.cell(0, 7, f"{ecg_result['probability']:.4f}", new_x="LMARGIN", new_y="NEXT")

        # ECG Chart
        if signal is not None:
            pdf.ln(4)
            ecg_fig = _create_ecg_chart_for_pdf(signal)
            ecg_path = _save_chart_to_temp(ecg_fig)
            pdf.image(ecg_path, x=15, w=180)
            os.unlink(ecg_path)
            pdf.ln(4)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 7, "ECG data was not provided for this assessment.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Clinical Results ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "2. Clinical Data Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Patient info table
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(60, 7, "Parameter", border=1, fill=True)
    pdf.cell(40, 7, "Value", border=1, fill=True)
    pdf.cell(60, 7, "Parameter", border=1, fill=True)
    pdf.cell(30, 7, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(71, 85, 105)

    rows = [
        ("Age", f"{clinical_inputs.get('age', 'N/A')} years", "Sex", "Male" if clinical_inputs.get("sex", 0) == 1 else "Female"),
        ("Chest Pain Type", str(clinical_inputs.get("cp", "N/A")), "Resting BP", f"{clinical_inputs.get('trestbps', 'N/A')} mm Hg"),
        ("Cholesterol", f"{clinical_inputs.get('chol', 'N/A')} mg/dl", "Fasting BS > 120", "Yes" if clinical_inputs.get("fbs", 0) == 1 else "No"),
        ("Max Heart Rate", f"{clinical_inputs.get('thalach', 'N/A')} bpm", "Exercise Angina", "Yes" if clinical_inputs.get("exang", 0) == 1 else "No"),
        ("ST Depression", str(clinical_inputs.get("oldpeak", "N/A")), "Major Vessels", str(clinical_inputs.get("ca", "N/A"))),
    ]

    for r in rows:
        pdf.cell(60, 6, r[0], border=1)
        pdf.cell(40, 6, r[1], border=1)
        pdf.cell(60, 6, r[2], border=1)
        pdf.cell(30, 6, r[3], border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(50, 7, "Clinical Risk:", new_x="RIGHT")
    if clinical_result["label"] == 1:
        pdf.set_text_color(239, 68, 68)
    else:
        pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 7, f"{clinical_result['risk_percent']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Patient vs Population Comparison ──
    comparisons = get_patient_comparison(clinical_inputs)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "3. Patient vs Population Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    comp_fig = _create_comparison_chart_for_pdf(comparisons)
    comp_path = _save_chart_to_temp(comp_fig)
    pdf.image(comp_path, x=15, w=180)
    os.unlink(comp_path)
    pdf.ln(4)

    # Comparison table
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 41, 59)
    for h, w in [("Feature", 35), ("Your Value", 25), ("Pop. Avg", 25), ("Normal Range", 35), ("Percentile", 25), ("Status", 30)]:
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(71, 85, 105)
    for c in comparisons:
        pdf.cell(35, 6, c["feature"], border=1)
        pdf.cell(25, 6, f"{c['value']} {c['unit']}", border=1)
        pdf.cell(25, 6, f"{c['mean']} {c['unit']}", border=1)
        pdf.cell(35, 6, f"{c['normal_low']} - {c['normal_high']} {c['unit']}", border=1)
        pdf.cell(25, 6, f"{c['percentile']}th", border=1)
        pdf.cell(30, 6, "Normal" if c["in_range"] else "Outside Range", border=1)
        pdf.ln()
    pdf.ln(4)

    # ── Recommendation ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "4. Recommendation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 6, fusion_result["recommendation"])
    pdf.ln(8)

    # ── Disclaimer ──
    pdf.set_fill_color(254, 249, 195)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 6, "  MEDICAL DISCLAIMER", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 53, 15)
    pdf.multi_cell(0, 5,
        "This system is for educational and research purposes only. "
        "It is NOT intended for clinical use. Always consult a qualified "
        "healthcare provider for medical advice, diagnosis, or treatment."
    )

    return bytes(pdf.output())
