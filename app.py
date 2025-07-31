import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide")

# Load the trained model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_adaboost_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

st.title("Cardiovascular Disease Prediction")

# Layout: left for inputs, spacer for gap, right for prediction
left, gap, right = st.columns([3, 0.3, 2])

with left:
    st.markdown("### Enter Patient Information")
    with st.container():
        height = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", 0, 100, 0)
        fruit_consumption = st.number_input("Fruit Consumption (servings/day)", 0, 100, 30)
        green_vegetables_consumption = st.number_input("Green Vegetables (servings/day)", 0, 100, 30)
        fried_potato_consumption = st.number_input("Fried Potato (servings/week)", 0, 100, 0)
        age = st.number_input("Age", 18, 120, 50)
        checkup = st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
                               format_func=lambda x: ["Never", "5+ years ago", "Within past 5 years",
                                                      "Within past 2 years", "Within past year"][x])
        general_health = st.selectbox("General Health", [0, 1, 2, 3, 4],
                                      format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])
        exercise = st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        skin_cancer = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        other_cancer = st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        depression = st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        arthritis = st.selectbox("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        smoking_history = st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        gender = st.selectbox("Gender", ["Female", "Male"])

with right:
    st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold;'>Prediction Result</div>", unsafe_allow_html=True)

    female = 1 if gender == "Female" else 0
    male = 1 if gender == "Male" else 0

    if bmi < 18.5:
        bmi_category = 0
    elif 18.5 <= bmi < 25:
        bmi_category = 1
    elif 25 <= bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 0

    if st.button("Predict", use_container_width=True):
        input_features = [
            'height', 'weight', 'bmi', 'alcohol_consumption', 'fruit_consumption',
            'green_vegetables_consumption', 'fried_potato_consumption', 'age',
            'checkup', 'general_health', 'exercise', 'skin_cancer', 'other_cancer',
            'depression', 'arthritis', 'diabetes', 'smoking_history', 'female', 'male',
            'bmi_category'
        ]
      
        user_input = pd.DataFrame([[height, weight, bmi, alcohol_consumption, fruit_consumption,
                                    green_vegetables_consumption, fried_potato_consumption, age,
                                    checkup, general_health, exercise, skin_cancer, other_cancer,
                                    depression, arthritis, diabetes, smoking_history, female, male,
                                    bmi_category]],
                                  columns=input_features)

        # Scale the input
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction_proba = model.predict_proba(user_input_scaled)[:, 1]
        st.write(
            f"<div style='text-align: center; font-size: 18px; font-weight: bold;'>Probability of Heart Disease: {prediction_proba[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        if prediction_proba[0] > 0.5:
            st.warning("Higher likelihood of heart disease.")
        else:
            st.info("Lower likelihood of heart disease.")
