import streamlit as st
import pandas as pd
import pickle

# Load the trained AdaBoost model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_adaboost_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Define feature names in the exact order used during training
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
    'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption',
    'Age', 'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded',
    'Skin_Cancer_Encoded', 'Other_Cancer_Encoded', 'Depression_Encoded',
    'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded',
    'Female', 'Male', 'BMI_Category'
]

st.title("Cardiovascular Disease Prediction")
st.write("Enter the patient's information below to predict the likelihood of heart disease.")

# Collect user input
height = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
alcohol_consumption = st.number_input("Alcohol Consumption (drinks per week)", min_value=0, max_value=100, value=0)
fruit_consumption = st.number_input("Fruit Consumption (servings per day)", min_value=0, max_value=100, value=30)
green_vegetables_consumption = st.number_input("Green Vegetables Consumption (servings per day)", min_value=0, max_value=100, value=30)
fried_potato_consumption = st.number_input("Fried Potato Consumption (servings per week)", min_value=0, max_value=100, value=0)
age = st.number_input("Age", min_value=18, max_value=120, value=50)
checkup = st.selectbox("Last Checkup", options=[0, 1, 2, 3, 4],
                       format_func=lambda x: ["Within past year", "Within past 2 years", "Within past 5 years", "5+ years ago", "Never"][x])
general_health = st.selectbox("General Health", options=[0, 1, 2, 3, 4],
                              format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x])
exercise = st.selectbox("Exercise", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
skin_cancer = st.selectbox("Skin Cancer", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
other_cancer = st.selectbox("Other Cancer", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
depression = st.selectbox("Depression", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
arthritis = st.selectbox("Arthritis", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
smoking_history = st.selectbox("Smoking History", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
gender = st.selectbox("Gender", options=["Female", "Male"])

# Encode gender
female = 1 if gender == "Female" else 0
male = 1 if gender == "Male" else 0

# Calculate BMI category
if bmi < 18.5:
    bmi_category = 0
elif 18.5 <= bmi < 25:
    bmi_category = 1
elif 25 <= bmi < 30:
    bmi_category = 2
else:
    bmi_category = 3

# Predict button
if st.button("Predict"):
    user_input = pd.DataFrame([[
        height, weight, bmi, alcohol_consumption, fruit_consumption,
        green_vegetables_consumption, fried_potato_consumption, age,
        checkup, general_health, exercise, skin_cancer, other_cancer,
        depression, arthritis, diabetes, smoking_history, female, male,
        bmi_category
    ]], columns=feature_names)

    prediction_proba = model.predict_proba(user_input)[:, 1]

    st.subheader("Prediction Result")
    st.metric(label="Probability of Heart Disease", value=f"{prediction_proba[0]*100:.2f}%")

    if prediction_proba[0] > 0.5:
        st.error("High likelihood of heart disease. Consider a medical consultation.")
    else:
        st.success("Low likelihood of heart disease.")
