# 🫀 AI-Based Heart Risk Prediction System

> A **hybrid AI healthcare web application** that combines ECG signal analysis (CNN-LSTM), clinical risk prediction (Random Forest), and a **Large Language Model (Gemini AI)** for comprehensive, human-readable heart risk assessment.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-green?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.5_Flash-purple?logo=google)

---

## 📋 Overview

This project provides a **real-world inspired AI healthcare system** that analyzes both **ECG signals** and **clinical patient data** to produce a highly accurate risk assessment. It combines two mathematical AI models using a hybrid fusion approach, and utilizes an embedded **Agentic AI Assistant** to interpret the results for the patient.

### Key Features

- 📤 **ECG Upload** — Upload `.csv` / `.npy` files or enter values manually
- 🩺 **Clinical Input** — 13-feature patient data form (age, BP, cholesterol, etc.)
- 🧠 **Dual Machine Learning Models** — CNN-LSTM (ECG) + Random Forest (Clinical)
- 🔬 **Hybrid Fusion** — Weighted composite risk scoring
- 👨‍⚕️ **Virtual Cardiologist (Agentic AI)** — Embedded Google Gemini AI chatbot that explains risk output utilizing **Tool Calling / External APIs**.
- 📊 **Gauge Chart & Visualizations** — Visual risk score indicator and interactive ECG waveform
- 📜 **Prediction History** — Track past predictions in-session
- 📥 **Download Reports** — Exportable comprehensive prediction reports (PDF & TXT)
- 🎨 **Premium UI** — Medical-themed dark interface with fast rendering

---

## 🏗️ Project Structure

```text
Heart Attack prediction/
├── app.py                    # Main Streamlit hybrid web application
├── medical_agent.py          # AI Agent logic (Gemini API & Tool Calling)
├── utils.py                  # ECG + Clinical preprocessing, prediction, formatting helpers
├── clinical_model.py         # Clinical model training script
├── ecg_model .h5             # Pre-trained CNN-LSTM ECG model
├── ecg_model .py             # ECG model training code (reference)
├── requirements.txt          # Python dependencies
├── .env                      # API keys and secrets (User created)
├── models/                   # Trained clinical model artifacts
│   ├── clinical_rf_model.pkl
│   ├── clinical_scaler.pkl
│   └── clinical_features.pkl
└── data/                     # Datasets
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A free **Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "/Users/kowshik/Documents/Heart Attack prediction"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=AI...your_api_key_here...
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser:** Navigate to `http://localhost:8501`

---

## 🔬 System Architecture

The application is built on a 4-stage processing pipeline:

### Module 1: ECG Signal Analysis (CNN-LSTM)
- **Architecture**: 1D Convolutional Neural Network + LSTM (Long Short-Term Memory)
- **Input Object**: 200 array data points 
- **Output**: Binary Classification (Normal vs. Abnormal)

### Module 2: Clinical Risk Prediction (Random Forest)
- **Algorithm**: Random Forest Classifier
- **Features**: 13 standard clinical parameters from the Cleveland Heart Disease Dataset
- **Output**: Probability risk percentage (0-100%)

### Module 3: Hybrid Fusion Logic
A weighted composite mathematical risk score determining the overall "Risk Level":
**Composite Score** = 40% × (ECG probability) + 60% × (Clinical probability)

### Module 4: Agentic AI Assistant (Gemini) & External API Tool Calling
Once the mathematics are processed, the system triggers the `medical_agent.py` module.
- **Technology**: Google Gemini `gemini-2.5-flash` model.
- **External Integration**: Uses the `google-generativeai` wrapper to communicate with Google's Cloud API endpoint.
- **Agentic Workflow**: The agent uses **Tool Calling** (Function Calling). We provide it with specific Python tools (such as checking AHA medical guidelines for cholesterol, or calculating BMI). The LLM autonomously decides *when* to execute these tools to properly answer user queries regarding their predicted heart risk.

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

1. **Upload ECG** — Upload a `.csv` or `.npy` file containing exactly 200 ECG values.
2. **Enter Clinical Data** — Complete the 13 clinical parameter fields.
3. **Run Analysis** — Click "Run Full Hybrid Analysis".
4. **Consult Dr. AI** — Scroll to the bottom of the page to chat with the integrated Virtual Cardiologist to understand *why* you received your specific score.
5. **Download Report** — Export the statistical breakdown as a PDF.

---

## ⚠️ Disclaimer

This application is developed for **educational and research purposes only**. It is **NOT** intended for clinical use and should not be used as a substitute for professional medical diagnosis, advice, or treatment. The AI model responses may occasionally be inaccurate. Always consult a qualified healthcare provider.

---

## 📝 License

Created for academic purposes as a final-year AI Healthcare project.

---

*Built with ❤️ using TensorFlow, Scikit-learn, Google Gemini AI, and Streamlit.*
