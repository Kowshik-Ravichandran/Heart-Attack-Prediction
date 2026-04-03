import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API key if it exists
if API_KEY:
    genai.configure(api_key=API_KEY)


# ────────────────────────────────────────────────────────────
# AGENT TOOLS (EXTERNAL KNOWLEDGE & UTILITIES)
# ────────────────────────────────────────────────────────────

def get_medical_guidelines(topic: str) -> str:
    """
    Fetches general medical guidelines for a given topic (e.g., 'blood pressure', 'cholesterol').
    Use this tool if the user asks what the normal ranges are for a given metric.
    """
    topic_lower = topic.lower()
    if "cholesterol" in topic_lower:
        return "The American Heart Association recommends keeping Total Cholesterol under 200 mg/dL. 200-239 is borderline high, and 240+ is considered high. High cholesterol leads to plaque buildup (atherosclerosis) increasing risk for heart attacks."
    elif "blood pressure" in topic_lower or "trestbps" in topic_lower:
        return "Normal resting blood pressure is generally <120 / <80 mmHg. Elevated is 120-129. Hypertension Stage 1 is 130-139. Hypertension Stage 2 is 140+ based on AHA guidelines. High blood pressure forces the heart to work harder, severely increasing heart attack risk."
    elif "ecg" in topic_lower or "st depression" in topic_lower or "abnormal" in topic_lower:
        return "ECG abnormalities such as ST-T wave changes, ST depression (oldpeak), or LV Hypertrophy can indicate that the heart muscle isn't getting enough oxygen or is thickening due to high blood pressure."
    elif "max heart rate" in topic_lower or "thalach" in topic_lower:
        return "A healthy max heart rate during exercise is typically calculated as 220 minus your age. A lower achieved max heart rate can sometimes indicate poor cardiac condition or the use of beta-blockers."
    else:
        return "No specific guidelines available for that exact topic. Remind the user to consult a board-certified cardiologist."

def check_bmi_risk(weight_kg: float, height_cm: float) -> str:
    """
    Calculates the Body Mass Index (BMI) and checks if it poses a heart risk.
    """
    if height_cm <= 0:
        return "Invalid height."
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Healthy weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return f"The calculated BMI is {bmi:.1f} ({category}). An overweight or obese BMI is an independent risk factor for heart disease."


# ────────────────────────────────────────────────────────────
# AGENT INITIALIZATION
# ────────────────────────────────────────────────────────────

def create_agent_chat():
    """
    Initializes the Gemini agent with specialized Medical knowledge
    and enables automatic tool calling.
    """
    if not API_KEY:
        return None

    # This is the "Persona" or System Prompt for our agent.
    system_instruction = """You are Dr. AI, an expert Virtual Cardiologist embedded in a Machine Learning Heart Risk Prediction app.
Your role is to explain the patient's machine-learning risk scores to them in simple, empathetic, and clear language.

When the patient starts a new chat, you will be given their recent ML predictions (CNN-LSTM ECG Result, Random Forest Clinical Risk).
- If their risk is HIGH, explain WHY based on their specific features (e.g., "Your blood pressure of 180 is very high..."). 
- Keep your answers VERY short, concise, and easy to read. Max 3-4 sentences unless explaining something complex.
- You have access to tools (`get_medical_guidelines`, `check_bmi_risk`). Call these tools if the user asks about specific guidelines (like "Is my cholesterol bad?").
- You are speaking directly to the patient ("You", "Your risk"). 
- IMPORTANT: Always include a brief disclaimer that you are an AI and they should consult a human doctor for medical advice.
"""

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        tools=[get_medical_guidelines, check_bmi_risk]
    )

    # Enable automatic tool calling so the AI decides when to run the Python functions above
    chat = model.start_chat(enable_automatic_function_calling=True)
    return chat
