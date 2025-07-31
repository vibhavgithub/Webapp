import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Cardiovascular Disease Prediction", page_icon="❤️")

# Custom CSS styling
st.markdown("""
<style>
.scrollable-container {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 15px;
    border: 1px solid #b3e5fc;
    padding: 15px;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
}
.centered-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 20px;
    border: 1px solid #b3e5fc;
    border-radius: 8px;
    background-color: #ffffff;
    margin-top: 20px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
}
.stButton>button {
    background-color: #00bcd4;
    color: white;
    padding: 12px 28px;
    font-size: 18px;
    margin: 15px 2px;
    cursor: pointer;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #00838f;
}
.prediction-text {
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
    color: #00796b;
}
</style>
""", unsafe_allow_html=True)

# Load model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

# Load scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found.")
    st.stop()

# Title
st.title("Cardiovascular Disease Prediction")

# Feature names and importances
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

importances = model.feature_importances_
feature_importances_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

important_features = feature_importances_df[feature_importances_df['Importance'] > 0]['Feature'].tolist()

# Layout: Inputs in col1 & col2, result in col3
col1, col2, col3 = st.columns([1.5, 1.5, 2])
user_inputs = {}

# Split fields
with col1:
    st.markdown("### Patient Info (Part 1)")
    with st.container():
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        if 'Age' in important_features:
            user_inputs['Age'] = st.number_input("Age", 18, 120, 50)
        if 'General_Health_Encoded' in important_features:
            user_inputs['General_Health_Encoded'] = st.selectbox("General Health", [0, 1, 2, 3, 4],
                format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])
        if 'Male' in important_features:
            user_inputs['Male'] = st.selectbox("Gender", ["Female", "Male"])
        if 'Height_(cm)' in important_features:
            user_inputs['Height_(cm)'] = st.number_input("Height (cm)", 50, 300, 170)
        if 'Weight_(kg)' in important_features:
            user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", 10.0, 500.0, 70.0, step=0.1)
        if 'BMI' in important_features:
            user_inputs['BMI'] = st.number_input("BMI", 10.0, 100.0, 25.0, step=0.1)
        if 'Alcohol_Consumption' in important_features:
            user_inputs['Alcohol_Consumption'] = st.number_input("Alcohol (drinks/week)", 0, 100, 0)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Patient Info (Part 2)")
    with st.container():
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        if 'Fruit_Consumption' in important_features:
            user_inputs['Fruit_Consumption'] = st.number_input("Fruit (servings/day)", 0, 100, 30)
        if 'Green_Vegetables_Consumption' in important_features:
            user_inputs['Green_Vegetables_Consumption'] = st.number_input("Green Veg (servings/day)", 0, 100, 30)
        if 'FriedPotato_Consumption' in important_features:
            user_inputs['FriedPotato_Consumption'] = st.number_input("Fried Potato (servings/week)", 0, 100, 0)
        if 'Exercise_Encoded' in important_features:
            user_inputs['Exercise_Encoded'] = st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Arthritis_Encoded' in important_features:
            user_inputs['Arthritis_Encoded'] = st.selectbox("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Diabetes_Encoded' in important_features:
            user_inputs['Diabetes_Encoded'] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Smoking_History_Encoded' in important_features:
            user_inputs['Smoking_History_Encoded'] = st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Skin_Cancer_Encoded' in important_features:
            user_inputs['Skin_Cancer_Encoded'] = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Other_Cancer_Encoded' in important_features:
            user_inputs['Other_Cancer_Encoded'] = st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Checkup_Encoded' in important_features:
            user_inputs['Checkup_Encoded'] = st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
                format_func=lambda x: ["Never", "5+ years ago", "Within past 5 years", "Within past 2 years", "Within past year"][x])
        if 'Depression_Encoded' in important_features:
            user_inputs['Depression_Encoded'] = st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        st.markdown('</div>', unsafe_allow_html=True)

# Prediction Panel
with col3:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 22px; font-weight: bold;'>Prediction Result</div>", unsafe_allow_html=True)

    user_input_data = {feature: 0 for feature in feature_names}
    for feature in user_inputs:
        user_input_data[feature] = user_inputs[feature]

    user_input_data['Female'] = 1 if user_inputs.get('Male') == 'Female' else 0
    user_input_data['Male'] = 1 if user_inputs.get('Male') == 'Male' else 0

    bmi_val = user_inputs.get('BMI', 25.0)
    if bmi_val < 18.5:
        user_input_data['BMI_Category_Encoded'] = 0
    elif 18.5 <= bmi_val < 25:
        user_input_data['BMI_Category_Encoded'] = 1
    elif 25 <= bmi_val < 30:
        user_input_data['BMI_Category_Encoded'] = 2
    else:
        user_input_data['BMI_Category_Encoded'] = 0

    if st.button("Predict", use_container_width=True):
        user_input_df = pd.DataFrame([user_input_data], columns=feature_names)
        user_input_scaled = scaler.transform(user_input_df)
        prediction_proba = model.predict_proba(user_input_scaled)[:, 1]

        st.markdown(
            f"<div class='prediction-text'>Probability of Heart Disease: {prediction_proba[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        if prediction_proba[0] > 0.5:
            st.warning("Higher likelihood of heart disease.")
        else:
            st.info("Lower likelihood of heart disease.")

    st.markdown("---")
    st.markdown("[Data Insights Dashboard Link](YOUR_DASHBOARD_LINK_HERE)")
    st.markdown('</div>', unsafe_allow_html=True)
