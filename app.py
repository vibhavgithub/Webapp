import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the trained model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_adaboost_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop() # Stop the app if the model file is not found

# Load the scaler used during training
# IMPORTANT: You need to save and load the scaler as well!
# For now, we'll create a dummy scaler, but you should replace this
# with the actual scaler fitted on your training data.
# You would typically save the scaler like this after fitting:
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
# And load it like this:
# try:
#     with open('scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
# except FileNotFoundError:
#     st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
#     st.stop()

# Dummy scaler for demonstration - REPLACE WITH YOUR ACTUAL SCALER
# You need to ensure this scaler has the same number of features as your model expects (20 features)
# and is fitted in the same way as the one used during model training.
# A simple way to get the feature names is from your X_train or X_resampled DataFrame columns.
# Assuming your X_train columns are the features:
feature_names = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
                 'Fruit_Consumption', 'Green_Vegetables_Consumption',
                 'FriedPotato_Consumption', 'Age', 'Checkup_Encoded',
                 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded',
                 'Other_Cancer_Encoded', 'Depression_Encoded', 'Arthritis_Encoded',
                 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
                 'BMI_Category']

# Create a dummy scaler and fit it with some dummy data to match the expected input shape
# This is a placeholder. You MUST use the actual scaler fitted on your training data.
dummy_data = pd.DataFrame(np.random.rand(10, len(feature_names)), columns=feature_names)
scaler = StandardScaler()
scaler.fit(dummy_data)


st.title("Cardiovascular Disease Prediction")

st.write("""
Enter the patient's information below to predict the likelihood of heart disease.
""")

# Create input fields for each feature
# You'll need to adjust the min_value, max_value, and step for each input based on your data's range
height = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
alcohol_consumption = st.number_input("Alcohol Consumption (drinks per week)", min_value=0, max_value=100, value=0)
fruit_consumption = st.number_input("Fruit Consumption (servings per day)", min_value=0, max_value=100, value=30)
green_vegetables_consumption = st.number_input("Green Vegetables Consumption (servings per day)", min_value=0, max_value=100, value=30)
fried_potato_consumption = st.number_input("Fried Potato Consumption (servings per week)", min_value=0, max_value=100, value=0)
age = st.number_input("Age", min_value=18, max_value=120, value=50)
checkup = st.selectbox("Last Checkup", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Within past year", "Within past 2 years", "Within past 5 years", "5+ years ago", "Never"][x])
general_health = st.selectbox("General Health", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x])
exercise = st.selectbox("Exercise", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
skin_cancer = st.selectbox("Skin Cancer", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
other_cancer = st.selectbox("Other Cancer", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
depression = st.selectbox("Depression", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
arthritis = st.selectbox("Arthritis", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: ["No", "Yes"][x]) # Assuming 0 for no, 1 for yes based on your data info
smoking_history = st.selectbox("Smoking History", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])
gender = st.selectbox("Gender", options=["Female", "Male"])

# Encode gender
female = 1 if gender == "Female" else 0
male = 1 if gender == "Male" else 0

# Assuming BMI Category is derived from BMI - you might need to add the logic here
# For simplicity, I'm adding a placeholder input, but you should calculate this from BMI
# based on the ranges used when creating your training data.
# For example:
# if bmi < 18.5: bmi_category = 0 # Underweight
# elif 18.5 <= bmi < 25: bmi_category = 1 # Healthy Weight
# elif 25 <= bmi < 30: bmi_category = 2 # Overweight
# else: bmi_category = 3 # Obese
bmi_category = st.selectbox("BMI Category (Select based on BMI ranges used in training)", options=[0, 1, 2, 3], format_func=lambda x: ["Underweight", "Healthy Weight", "Overweight", "Obese"][x])


# Create a button to make a prediction
if st.button("Predict"):
    # Prepare the input data as a pandas DataFrame
    user_input = pd.DataFrame([[height, weight, bmi, alcohol_consumption, fruit_consumption,
                                green_vegetables_consumption, fried_potato_consumption, age,
                                checkup, general_health, exercise, skin_cancer, other_cancer,
                                depression, arthritis, diabetes, smoking_history, female, male,
                                bmi_category]],
                              columns=feature_names) # Use the defined feature names

    # Scale the user input
    # IMPORTANT: Use the same scaler object that was fitted on your training data
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction_proba = model.predict_proba(user_input_scaled)[:, 1] # Probability of Heart Disease (class 1)

    st.subheader("Prediction Result:")
    st.write(f"Probability of Heart Disease: **{prediction_proba[0]:.2f}**")

    # You can add conditional messages based on the probability
    if prediction_proba[0] > 0.5: # You can adjust this threshold
        st.warning("Based on the provided information, there is a higher likelihood of heart disease.")
    else:
        st.info("Based on the provided information, there is a lower likelihood of heart disease.")