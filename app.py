import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("best_adaboost_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Get feature names from model
feature_names = list(model.feature_names_in_)

# Separate features into important ones and derived ones
input_features = [
    'height', 'weight', 'bmi', 'alcohol_consumption', 'fruit_consumption',
    'green_vegetables_consumption', 'fried_potato_consumption', 'age',
    'checkup', 'general_health', 'exercise', 'skin_cancer', 'other_cancer',
    'depression', 'arthritis', 'diabetes', 'smoking_history', 'female', 'male',
    'bmi_category'
]

# Sidebar (Left panel)
st.sidebar.title("Enter User Data")

user_inputs = {}

user_inputs['Height'] = st.sidebar.slider("Height (in cm)", 140, 210, 170)
user_inputs['Weight'] = st.sidebar.slider("Weight (in kg)", 40, 140, 70)
user_inputs['BMI'] = round(user_inputs['Weight'] / ((user_inputs['Height'] / 100) ** 2), 1)

st.sidebar.markdown(f"**BMI: {user_inputs['BMI']}**")

if user_inputs['BMI'] < 18.5:
    user_inputs['BMI Category'] = 0  # Underweight
elif user_inputs['BMI'] <= 24.9:
    user_inputs['BMI Category'] = 1  # Normal
elif user_inputs['BMI'] <= 29.9:
    user_inputs['BMI Category'] = 2  # Overweight
else:
    user_inputs['BMI Category'] = 3  # Obese

user_inputs['Alcohol Consumption'] = st.sidebar.slider("Alcohol Consumption (days/week)", 0, 7, 1)
user_inputs['Fruit Consumption'] = st.sidebar.slider("Fruit Consumption (days/week)", 0, 7, 3)
user_inputs['Green Vegetables Consumption'] = st.sidebar.slider("Green Vegetables Consumption (days/week)", 0, 7, 3)
user_inputs['Fried Potato Consumption'] = st.sidebar.slider("Fried Potato Consumption (days/week)", 0, 7, 2)
user_inputs['Age'] = st.sidebar.slider("Age", 18, 100, 30)
user_inputs['Checkup'] = st.sidebar.selectbox("Last Routine Checkup", [0, 1, 2, 3])
user_inputs['General Health'] = st.sidebar.slider("General Health (1=Poor, 5=Excellent)", 1, 5, 3)
user_inputs['Exercise'] = st.sidebar.selectbox("Did Physical Exercise Recently?", [0, 1])
user_inputs['Skin Cancer'] = st.sidebar.selectbox("History of Skin Cancer", [0, 1])
user_inputs['Other Cancer'] = st.sidebar.selectbox("History of Other Cancer", [0, 1])
user_inputs['Depression'] = st.sidebar.selectbox("Diagnosed with Depression", [0, 1])
user_inputs['Arthritis'] = st.sidebar.selectbox("Diagnosed with Arthritis", [0, 1])
user_inputs['Diabetes'] = st.sidebar.selectbox("Diagnosed with Diabetes", [0, 1])
user_inputs['Smoking History'] = st.sidebar.selectbox("Smoking History", [0, 1])
user_inputs['Male'] = st.sidebar.selectbox("Sex", ["Male", "Female"])

# Layout: Right panel center with gap
col1, col2, col3 = st.columns([1, 2.5, 1])  # Leave space on left/right for centering

with col2:
    st.title("Heart Disease Prediction")

    if st.button("Predict", use_container_width=True):
        # Prepare single row input
        input_data = {
            'height': user_inputs['Height'],
            'weight': user_inputs['Weight'],
            'bmi': user_inputs['BMI'],
            'bmi_category': user_inputs['BMI Category'],
            'alcohol_consumption': user_inputs['Alcohol Consumption'],
            'fruit_consumption': user_inputs['Fruit Consumption'],
            'green_vegetables_consumption': user_inputs['Green Vegetables Consumption'],
            'fried_potato_consumption': user_inputs['Fried Potato Consumption'],
            'age': user_inputs['Age'],
            'checkup': user_inputs['Checkup'],
            'general_health': user_inputs['General Health'],
            'exercise': user_inputs['Exercise'],
            'skin_cancer': user_inputs['Skin Cancer'],
            'other_cancer': user_inputs['Other Cancer'],
            'depression': user_inputs['Depression'],
            'arthritis': user_inputs['Arthritis'],
            'diabetes': user_inputs['Diabetes'],
            'smoking_history': user_inputs['Smoking History'],
            'female': 1 if user_inputs['Male'] == "Female" else 0,
            'male': 1 if user_inputs['Male'] == "Male" else 0
        }

        # Ensure all required features are present
        user_input_row = {feature: input_data.get(feature, 0) for feature in feature_names}

        # Convert to DataFrame
        user_input_df = pd.DataFrame([user_input_row])

        # Scale
        user_input_scaled = scaler.transform(user_input_df.values)

        # Predict
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)

        # Display Result (Top of Right Panel)
        if prediction[0] == 0:
            st.success("No Heart Disease Risk Detected ✅")
        else:
            st.error("Potential Heart Disease Risk Detected ⚠️")

        st.subheader("Prediction Probability")
        st.write(f"Healthy: {round(prediction_proba[0][0]*100, 2)}%")
        st.write(f"Risk: {round(prediction_proba[0][1]*100, 2)}%")

    st.markdown("---")

    # Dashboard link (Bottom of Right Panel)
    st.markdown("📊 [Click here to view the Data Insight Dashboard](https://your-dashboard-link.com)")
