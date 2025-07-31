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
    st.error("Model file 'best_adaboost_model.pkl' not found.")
    st.stop()

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found.")
    st.stop()

st.title("🫀 Cardiovascular Disease Prediction")

# Define feature names
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

# Feature importances
try:
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
except AttributeError:
    st.error("Model missing feature importance info.")
    st.stop()

important_features = feature_importances_df[feature_importances_df['Importance'] > 0]['Feature'].tolist()

# Input collection
st.markdown("### 📝 Enter Patient Information")
user_inputs = {}

if 'Age' in important_features:
    user_inputs['Age'] = st.slider("Age", 18, 100, 45)
if 'General_Health_Encoded' in important_features:
    user_inputs['General_Health_Encoded'] = st.radio("General Health", [0, 1, 2, 3, 4],
        format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])
if 'Male' in important_features:
    gender = st.radio("Gender", ["Female", "Male"])
    user_inputs['Male'] = 1 if gender == "Male" else 0
if 'Arthritis_Encoded' in important_features:
    user_inputs['Arthritis_Encoded'] = st.radio("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Diabetes_Encoded' in important_features:
    user_inputs['Diabetes_Encoded'] = st.radio("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Smoking_History_Encoded' in important_features:
    user_inputs['Smoking_History_Encoded'] = st.radio("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Skin_Cancer_Encoded' in important_features:
    user_inputs['Skin_Cancer_Encoded'] = st.radio("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Checkup_Encoded' in important_features:
    user_inputs['Checkup_Encoded'] = st.radio("Last Checkup", [0, 1, 2, 3, 4],
        format_func=lambda x: ["Never", "5+ years ago", "Within past 5 years", "Within past 2 years", "Within past year"][x])
if 'Depression_Encoded' in important_features:
    user_inputs['Depression_Encoded'] = st.radio("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Height_(cm)' in important_features:
    user_inputs['Height_(cm)'] = st.number_input("Height (cm)", 50, 300, 170)
if 'Weight_(kg)' in important_features:
    user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", 10.0, 500.0, 70.0)
if 'BMI' in important_features:
    user_inputs['BMI'] = st.number_input("BMI", 10.0, 100.0, 25.0)
if 'Alcohol_Consumption' in important_features:
    user_inputs['Alcohol_Consumption'] = st.slider("Alcohol (drinks/week)", 0, 50, 0)
if 'Fruit_Consumption' in important_features:
    user_inputs['Fruit_Consumption'] = st.slider("Fruit (servings/day)", 0, 20, 2)
if 'Green_Vegetables_Consumption' in important_features:
    user_inputs['Green_Vegetables_Consumption'] = st.slider("Green Veg (servings/day)", 0, 20, 3)
if 'FriedPotato_Consumption' in important_features:
    user_inputs['FriedPotato_Consumption'] = st.slider("Fried Potato (servings/week)", 0, 20, 0)
if 'Exercise_Encoded' in important_features:
    user_inputs['Exercise_Encoded'] = st.radio("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])
if 'Other_Cancer_Encoded' in important_features:
    user_inputs['Other_Cancer_Encoded'] = st.radio("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])

# Derived features
user_inputs['Female'] = 0 if user_inputs.get('Male') == 1 else 1
bmi_val = user_inputs.get('BMI', 25.0)
if bmi_val < 18.5:
    user_inputs['BMI_Category_Encoded'] = 0
elif 18.5 <= bmi_val < 25:
    user_inputs['BMI_Category_Encoded'] = 1
elif 25 <= bmi_val < 30:
    user_inputs['BMI_Category_Encoded'] = 2
else:
    user_inputs['BMI_Category_Encoded'] = 0

# Fill missing features
user_input_row = {feature: 0 for feature in feature_names}
for feature in user_inputs:
    user_input_row[feature] = user_inputs[feature]

user_input_df = pd.DataFrame([user_input_row], columns=feature_names)

# Prediction
st.markdown("### 🧪 Prediction Result")
if st.button("🔍 Predict"):
    user_input_scaled = scaler.transform(user_input_df)
    prediction_proba = model.predict_proba(user_input_scaled)[:, 1]
    st.markdown(
        f"<div style='font-size: 18px; font-weight: bold;'>Heart Disease Risk: {prediction_proba[0]*100:.2f}%</div>",
        unsafe_allow_html=True
    )
    if prediction_proba[0] > 0.5:
        st.warning("⚠️ Higher likelihood of heart disease.")
    else:
        st.success("✅ Lower likelihood of heart disease.")

# Dashboard link
st.markdown("---")
st.markdown("### 📊 [Go to Data Insights Dashboard](YOUR_DASHBOARD_LINK_HERE)")

