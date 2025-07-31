import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🫀 Cardiovascular Disease Prediction</h1>", unsafe_allow_html=True)

# Load model and scaler
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found.")
    st.stop()

# Feature names
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

# Feature importance
try:
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
except AttributeError:
    st.error("Could not read feature importances.")
    st.stop()

important_features = feature_importances_df[feature_importances_df['Importance'] > 0]['Feature'].tolist()

st.markdown("### 📝 Enter Patient Information")

user_inputs = {}

input_container = st.container()
with input_container:
    cols = st.columns(3)  # Compact layout
    index = 0

    def next_col():
        nonlocal index
        col = cols[index % len(cols)]
        index += 1
        return col

    for feature in important_features:
        if feature == 'Age':
            with next_col():
                user_inputs['Age'] = st.number_input("Age", 18, 120, 50, key='age')
        elif feature == 'General_Health_Encoded':
            with next_col():
                user_inputs['General_Health_Encoded'] = st.selectbox("General Health", [0, 1, 2, 3, 4],
                    format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x], key='gh')
        elif feature == 'Male':
            with next_col():
                user_inputs['Male'] = st.selectbox("Gender", ["Female", "Male"], key='gender')
        elif feature == 'Arthritis_Encoded':
            with next_col():
                user_inputs['Arthritis_Encoded'] = st.selectbox("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='arth')
        elif feature == 'Diabetes_Encoded':
            with next_col():
                user_inputs['Diabetes_Encoded'] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='diab')
        elif feature == 'Smoking_History_Encoded':
            with next_col():
                user_inputs['Smoking_History_Encoded'] = st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='smk')
        elif feature == 'Skin_Cancer_Encoded':
            with next_col():
                user_inputs['Skin_Cancer_Encoded'] = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='sc')
        elif feature == 'Checkup_Encoded':
            with next_col():
                user_inputs['Checkup_Encoded'] = st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
                    format_func=lambda x: ["Never", "5+ years ago", "Within 5 years", "Within 2 years", "Within 1 year"][x], key='chk')
        elif feature == 'Depression_Encoded':
            with next_col():
                user_inputs['Depression_Encoded'] = st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='dep')
        elif feature == 'Height_(cm)':
            with next_col():
                user_inputs['Height_(cm)'] = st.number_input("Height (cm)", 50, 300, 170, key='ht')
        elif feature == 'Weight_(kg)':
            with next_col():
                user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", 10.0, 500.0, 70.0, step=0.1, key='wt')
        elif feature == 'BMI':
            with next_col():
                user_inputs['BMI'] = st.number_input("BMI", 10.0, 100.0, 25.0, step=0.1, key='bmi')
        elif feature == 'Alcohol_Consumption':
            with next_col():
                user_inputs['Alcohol_Consumption'] = st.number_input("Alcohol (drinks/week)", 0, 100, 0, key='alc')
        elif feature == 'Fruit_Consumption':
            with next_col():
                user_inputs['Fruit_Consumption'] = st.number_input("Fruits/day", 0, 100, 30, key='fru')
        elif feature == 'Green_Vegetables_Consumption':
            with next_col():
                user_inputs['Green_Vegetables_Consumption'] = st.number_input("Green Veg/day", 0, 100, 30, key='veg')
        elif feature == 'FriedPotato_Consumption':
            with next_col():
                user_inputs['FriedPotato_Consumption'] = st.number_input("Fried Potato/week", 0, 100, 0, key='fp')
        elif feature == 'Exercise_Encoded':
            with next_col():
                user_inputs['Exercise_Encoded'] = st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='exe')
        elif feature == 'Other_Cancer_Encoded':
            with next_col():
                user_inputs['Other_Cancer_Encoded'] = st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x], key='oc')

# --- Prediction Section ---
st.markdown("### 🧪 Prediction Result")

# Prepare input row
input_data = {}
for feature in important_features:
    if feature in user_inputs:
        input_data[feature] = user_inputs[feature]
    else:
        if feature == 'Female':
            input_data['Female'] = 1 if user_inputs.get('Male', 'Male') == 'Female' else 0
        elif feature == 'Male':
            input_data['Male'] = 1 if user_inputs.get('Male', 'Male') == 'Male' else 0
        elif feature == 'BMI_Category_Encoded':
            bmi_val = user_inputs.get('BMI', 25.0)
            if bmi_val < 18.5:
                input_data['BMI_Category_Encoded'] = 0
            elif 18.5 <= bmi_val < 25:
                input_data['BMI_Category_Encoded'] = 1
            elif 25 <= bmi_val < 30:
                input_data['BMI_Category_Encoded'] = 2
            else:
                input_data['BMI_Category_Encoded'] = 0
        else:
            input_data[feature] = 0

if st.button("Predict", use_container_width=True):
    user_input_row = {feature: 0 for feature in feature_names}
    for feature in input_data:
        user_input_row[feature] = input_data[feature]

    user_input_df = pd.DataFrame([user_input_row], columns=feature_names)
    user_input_scaled = scaler.transform(user_input_df)
    prediction_proba = model.predict_proba(user_input_scaled)[:, 1]
    percentage = prediction_proba[0] * 100

    st.markdown(f"<h4 style='text-align: center;'>Heart Disease Risk: {percentage:.2f}%</h4>", unsafe_allow_html=True)
    if percentage > 50:
        st.warning("⚠️ Higher likelihood of heart disease.")
    else:
        st.success("✅ Lower likelihood of heart disease.")

# --- Dashboard Section ---
st.markdown("### 📊 Data Insight Dashboard")
st.markdown("[Go to Dashboard ➡️](YOUR_DASHBOARD_LINK_HERE)")
