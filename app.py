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

# Page Title
st.title("Cardiovascular Disease Prediction")

# Define feature names - ensuring they match the training data
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

# Get feature importances
try:
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
except AttributeError:
    st.error("Could not get feature importances from the loaded model.")
    st.stop()

important_features = feature_importances_df[feature_importances_df['Importance'] > 0]['Feature'].tolist()

# Layout containers
layout = st.container()
with layout:
    col1, col_spacer, col2 = st.columns([3, 0.5, 2], gap="large")

    # LEFT PANEL - INPUT FORM
    with col1:
        st.markdown("### Enter Patient Information")
        # Scrollable container
        st.markdown("""
            <style>
            .scroll-box {
                max-height: 80vh;
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
            user_inputs = {}

            # Dynamically generate inputs based on feature importance
            if 'Age' in important_features:
                user_inputs['Age'] = st.number_input("Age", 18, 120, 50)
            if 'General_Health_Encoded' in important_features:
                user_inputs['General_Health_Encoded'] = st.selectbox("General Health", [0, 1, 2, 3, 4],
                    format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])
            if 'Male' in important_features:
                user_inputs['Male'] = st.selectbox("Gender", ["Female", "Male"], format_func=lambda x: x)
            if 'Arthritis_Encoded' in important_features:
                user_inputs['Arthritis_Encoded'] = st.selectbox("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Diabetes_Encoded' in important_features:
                user_inputs['Diabetes_Encoded'] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Smoking_History_Encoded' in important_features:
                user_inputs['Smoking_History_Encoded'] = st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Skin_Cancer_Encoded' in important_features:
                user_inputs['Skin_Cancer_Encoded'] = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Checkup_Encoded' in important_features:
                user_inputs['Checkup_Encoded'] = st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
                    format_func=lambda x: ["Never", "5+ years ago", "Within past 5 years", "Within past 2 years", "Within past year"][x])
            if 'Depression_Encoded' in important_features:
                user_inputs['Depression_Encoded'] = st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Height_(cm)' in important_features:
                user_inputs['Height_(cm)'] = st.number_input("Height (cm)", 50, 300, 170)
            if 'Weight_(kg)' in important_features:
                user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", 10.0, 500.0, 70.0, step=0.1)
            if 'BMI' in important_features:
                user_inputs['BMI'] = st.number_input("BMI", 10.0, 100.0, 25.0, step=0.1)
            if 'Alcohol_Consumption' in important_features:
                user_inputs['Alcohol_Consumption'] = st.number_input("Alcohol Consumption (drinks/week)", 0, 100, 0)
            if 'Fruit_Consumption' in important_features:
                user_inputs['Fruit_Consumption'] = st.number_input("Fruit Consumption (servings/day)", 0, 100, 30)
            if 'Green_Vegetables_Consumption' in important_features:
                user_inputs['Green_Vegetables_Consumption'] = st.number_input("Green Vegetables (servings/day)", 0, 100, 30)
            if 'FriedPotato_Consumption' in important_features:
                user_inputs['FriedPotato_Consumption'] = st.number_input("Fried Potato (servings/week)", 0, 100, 0)
            if 'Exercise_Encoded' in important_features:
                user_inputs['Exercise_Encoded'] = st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            if 'Other_Cancer_Encoded' in important_features:
                user_inputs['Other_Cancer_Encoded'] = st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT PANEL - PREDICTION OUTPUT
    with col2:
        st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold;'>Prediction Result</div>", unsafe_allow_html=True)

        # Prepare inputs
        input_data = {}
        for feature in important_features:
            if feature in user_inputs:
                input_data[feature] = user_inputs[feature]
            elif feature == 'Female':
                input_data['Female'] = 1 if user_inputs.get('Male') == 'Female' else 0
            elif feature == 'Male':
                input_data['Male'] = 1 if user_inputs.get('Male') == 'Male' else 0
            elif feature == 'BMI_Category_Encoded':
                bmi_val = user_inputs.get('BMI', 25.0)
                if bmi_val < 18.5:
                    input_data['BMI_Category_Encoded'] = 0
                elif bmi_val < 25:
                    input_data['BMI_Category_Encoded'] = 1
                elif bmi_val < 30:
                    input_data['BMI_Category_Encoded'] = 2
                else:
                    input_data['BMI_Category_Encoded'] = 0

        if st.button("Predict", use_container_width=True):
            user_input_row = {f: 0 for f in feature_names}
            user_input_row.update(input_data)

            user_input_df = pd.DataFrame([user_input_row], columns=feature_names)
            user_input_scaled = scaler.transform(user_input_df)
            prediction_proba = model.predict_proba(user_input_scaled)[:, 1]

            st.write(
                f"<div style='text-align: center; font-size: 18px; font-weight: bold;'>Probability of Heart Disease: {prediction_proba[0]*100:.2f}%</div>",
                unsafe_allow_html=True
            )
            if prediction_proba[0] > 0.5:
                st.warning("Higher likelihood of heart disease.")
            else:
                st.info("Lower likelihood of heart disease.")

        st.markdown("---")
        st.markdown("#### 🔍 [Data Insights Dashboard](YOUR_DASHBOARD_LINK_HERE)")
