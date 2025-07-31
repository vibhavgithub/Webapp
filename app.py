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

feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

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

# Layout: 2 for inputs, 1 for prediction
col1, col2, col3 = st.columns([1.2, 1.2, 1])

# Store user input
user_inputs = {}

# Distribute inputs into two columns
input_fields = []

# Prepare list of input widgets based on importance
if 'Age' in important_features:
    input_fields.append(('Age', st.number_input("Age", 18, 120, 50)))
if 'General_Health_Encoded' in important_features:
    input_fields.append(('General_Health_Encoded', st.selectbox("General Health", [0, 1, 2, 3, 4],
        format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x])))
if 'Male' in important_features:
    input_fields.append(('Male', st.selectbox("Gender", ["Female", "Male"])))
if 'Arthritis_Encoded' in important_features:
    input_fields.append(('Arthritis_Encoded', st.selectbox("Arthritis", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Diabetes_Encoded' in important_features:
    input_fields.append(('Diabetes_Encoded', st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Smoking_History_Encoded' in important_features:
    input_fields.append(('Smoking_History_Encoded', st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Skin_Cancer_Encoded' in important_features:
    input_fields.append(('Skin_Cancer_Encoded', st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Checkup_Encoded' in important_features:
    input_fields.append(('Checkup_Encoded', st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
        format_func=lambda x: ["Never", "5+ years ago", "Within 5 years", "Within 2 years", "Within 1 year"][x])))
if 'Depression_Encoded' in important_features:
    input_fields.append(('Depression_Encoded', st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Height_(cm)' in important_features:
    input_fields.append(('Height_(cm)', st.number_input("Height (cm)", 50, 300, 170)))
if 'Weight_(kg)' in important_features:
    input_fields.append(('Weight_(kg)', st.number_input("Weight (kg)", 10.0, 500.0, 70.0, step=0.1)))
if 'BMI' in important_features:
    input_fields.append(('BMI', st.number_input("BMI", 10.0, 100.0, 25.0, step=0.1)))
if 'Alcohol_Consumption' in important_features:
    input_fields.append(('Alcohol_Consumption', st.number_input("Alcohol (drinks/week)", 0, 100, 0)))
if 'Fruit_Consumption' in important_features:
    input_fields.append(('Fruit_Consumption', st.number_input("Fruits/day", 0, 100, 30)))
if 'Green_Vegetables_Consumption' in important_features:
    input_fields.append(('Green_Vegetables_Consumption', st.number_input("Green Veg/day", 0, 100, 30)))
if 'FriedPotato_Consumption' in important_features:
    input_fields.append(('FriedPotato_Consumption', st.number_input("Fried Potato/week", 0, 100, 0)))
if 'Exercise_Encoded' in important_features:
    input_fields.append(('Exercise_Encoded', st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])))
if 'Other_Cancer_Encoded' in important_features:
    input_fields.append(('Other_Cancer_Encoded', st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])))

# Place inputs into col1 and col2 alternatively
for i, (label, widget) in enumerate(input_fields):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        user_inputs[label] = widget

# Prepare data
input_data = {}
for feature in important_features:
    if feature in user_inputs:
        input_data[feature] = user_inputs[feature]
    elif feature == 'Female':
        input_data['Female'] = 1 if user_inputs.get('Male', 'Male') == 'Female' else 0
    elif feature == 'Male':
        input_data['Male'] = 1 if user_inputs.get('Male', 'Male') == 'Male' else 0
    elif feature == 'BMI_Category_Encoded':
        bmi = user_inputs.get('BMI', 25.0)
        if bmi < 18.5:
            input_data['BMI_Category_Encoded'] = 0
        elif 18.5 <= bmi < 25:
            input_data['BMI_Category_Encoded'] = 1
        elif 25 <= bmi < 30:
            input_data['BMI_Category_Encoded'] = 2
        else:
            input_data['BMI_Category_Encoded'] = 0
    else:
        input_data[feature] = 0

# Right column for prediction
with col3:
    st.markdown("### 🧪 Prediction Result")
    if st.button("Predict", use_container_width=True):
        user_input_row = {f: 0 for f in feature_names}
        for f in input_data:
            user_input_row[f] = input_data[f]
        user_input_df = pd.DataFrame([user_input_row], columns=feature_names)
        user_input_scaled = scaler.transform(user_input_df)
        prediction_proba = model.predict_proba(user_input_scaled)[:, 1]
        percentage = prediction_proba[0] * 100

        st.markdown(f"<h4 style='text-align: center;'>Heart Disease Risk: {percentage:.2f}%</h4>", unsafe_allow_html=True)
        if percentage > 50:
            st.warning("⚠️ Higher likelihood of heart disease.")
        else:
            st.success("✅ Lower likelihood of heart disease.")

    st.markdown("---")
    st.markdown("### 📊 Data Insights Dashboard")
    st.markdown("[Go to Dashboard ➡️](YOUR_DASHBOARD_LINK_HERE)")
