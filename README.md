# 🫀 AI-Based Heart Risk Prediction System

> A **hybrid AI healthcare web application** that combines ECG signal analysis (CNN-LSTM) with clinical risk prediction (Random Forest) for comprehensive heart risk assessment.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.2+-green?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)

---

## 📋 Overview

This project provides a **real-world inspired AI healthcare system** that analyzes both **ECG signals** and **clinical patient data** to produce a comprehensive heart risk assessment. It combines two AI models using a hybrid fusion approach.

### Key Features

- 📤 **ECG Upload** — Upload `.csv` / `.npy` files or enter values manually
- 🩺 **Clinical Input** — 13-feature patient data form (age, BP, cholesterol, etc.)
- 🧠 **Dual AI Models** — CNN-LSTM (ECG) + Random Forest (Clinical)
- 🔬 **Hybrid Fusion** — Weighted composite risk scoring
- 📊 **Gauge Chart** — Visual risk score indicator
- 📈 **Signal Visualization** — Interactive ECG waveform display
- 📜 **Prediction History** — Track past predictions in-session
- 📥 **Download Reports** — Exportable comprehensive prediction reports
- 🎨 **Premium UI** — Medical-themed dark interface with animations

---

## 🏗️ Project Structure

```
Heart Attack prediction/
├── app.py                    # Main Streamlit hybrid web application
├── utils.py                  # ECG + Clinical preprocessing & prediction helpers
├── clinical_model.py         # Clinical model training script
├── ecg_model .h5             # Pre-trained CNN-LSTM ECG model
├── ecg_model .py             # ECG model training code (reference)
├── requirements.txt          # Python dependencies
├── generate_sample_data.py   # Generate test ECG data
├── models/                   # Trained clinical model artifacts
│   ├── clinical_rf_model.pkl
│   ├── clinical_scaler.pkl
│   └── clinical_features.pkl
├── data/                     # Datasets
│   ├── heart.csv             # Heart Disease clinical dataset
│   └── mit-bih-.../          # MIT-BIH Arrhythmia Database (reference)
└── Sample Data/              # Test ECG files (.csv, .npy)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "/Users/kowshik/Documents/Heart Attack prediction"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the clinical model:**
   ```bash
   python clinical_model.py
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser:** Navigate to `http://localhost:8501`

---

## 🔬 System Architecture

### Module 1: ECG Signal Analysis (CNN-LSTM)

| Property        | Value                              |
|-----------------|-------------------------------------|
| Architecture    | CNN-LSTM (Conv1D + LSTM + Dense)   |
| Input Shape     | (200, 1)                           |
| Output          | Sigmoid (Binary Classification)    |
| Threshold       | 0.5                                |
| Training Data   | MIT-BIH Arrhythmia Database        |
| Normalization   | Z-score (mean=0, std=1)            |

### Module 2: Clinical Risk Prediction (Random Forest)

| Property        | Value                              |
|-----------------|-------------------------------------|
| Algorithm       | Random Forest (200 trees)          |
| Features        | 13 clinical parameters             |
| Output          | Probability (0–1)                  |
| Training Data   | Heart Disease Dataset (1025 records)|
| Accuracy        | ~99%                               |
| Scaling         | StandardScaler                     |

### Module 3: Hybrid Fusion Logic

| ECG Result | Clinical Result | Final Risk |
|------------|-----------------|------------|
| ⚠️ Abnormal | ⚠️ High Risk   | 🔴 HIGH    |
| ⚠️ Abnormal | ✅ Low Risk     | 🟠 MEDIUM  |
| ✅ Normal   | ⚠️ High Risk   | 🟠 MEDIUM  |
| ✅ Normal   | ✅ Low Risk     | 🟢 LOW     |

**Composite Score** = 0.4 × ECG probability + 0.6 × Clinical probability

---

## 🩺 Clinical Input Parameters

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | Age | Patient age | 1–120 years |
| 2 | Sex | Biological sex | Male / Female |
| 3 | Chest Pain Type | CP type | 0–3 |
| 4 | Resting BP | Blood pressure (mm Hg) | 60–250 |
| 5 | Cholesterol | Serum cholesterol (mg/dl) | 100–600 |
| 6 | Fasting Blood Sugar | FBS > 120 mg/dl | Yes / No |
| 7 | Resting ECG | ECG results | 0–2 |
| 8 | Max Heart Rate | During exercise (bpm) | 60–220 |
| 9 | Exercise Angina | Exercise-induced | Yes / No |
| 10 | ST Depression | Oldpeak value | 0–7 |
| 11 | ST Slope | Peak exercise ST | 0–2 |
| 12 | Major Vessels | Colored by fluoroscopy | 0–4 |
| 13 | Thalassemia | Thal type | 0–3 |

---

## 📊 How to Use

1. **Upload ECG** — Go to Module 1 and upload a `.csv` or `.npy` file with 200 ECG values
2. **Enter Clinical Data** — Fill in all 13 fields in Module 2
3. **Run Analysis** — Click "Run Full Hybrid Analysis" (or "Clinical-Only")
4. **View Results** — See risk level, gauge chart, confidence distribution
5. **Download Report** — Export the comprehensive prediction report
6. **Track History** — View all predictions made in the session

---

## ⚠️ Disclaimer

This application is developed for **educational and research purposes only**. It is **NOT** intended for clinical use and should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.

---

## 📝 License

This project is created for academic purposes as a final-year AI Healthcare project.

---

*Built with ❤️ using TensorFlow, Scikit-learn, and Streamlit*
