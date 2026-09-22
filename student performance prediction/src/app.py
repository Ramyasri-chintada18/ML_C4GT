from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Reliable model path loading using pathlib
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "student_performance_model.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = Path("models") / "student_performance_model.pkl"

model = joblib.load(MODEL_PATH)

# Page header
st.title("🎓 Student Performance Prediction")
st.write("Predict a student's Performance Index based on academic and lifestyle factors.")

# Input fields
hours = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=24,
    value=5,
    step=1
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)

activity = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=24,
    value=7,
    step=1
)

practice_papers = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)

# Predict button
if st.button("Predict Performance"):
    # Encode categorical feature: Yes -> 1, No -> 0
    activity_value = 1 if activity == "Yes" else 0

    # Model input order: [hours, previous_scores, activity_value, sleep_hours, practice_papers]
    feature_names = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
    input_data = pd.DataFrame(
        [[hours, previous_scores, activity_value, sleep_hours, practice_papers]],
        columns=feature_names
    )

    # Make raw prediction
    prediction = model.predict(input_data)[0]

    # Enforce performance index bounds between 10 and 100
    prediction = max(10, min(100, prediction))

    # Display prediction result
    st.success(f"Predicted Performance Index: {prediction:.2f}")