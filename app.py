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

# Define feature names - ensuring they match the training data
feature_names = [
    'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
    'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
    'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
    'BMI_Category_Encoded'
]

# Get feature importances from the loaded model
try:
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

except AttributeError:
    st.error("Could not get feature importances from the loaded model.")
    st.stop()


# Get the features with non-zero importance
important_features = feature_importances_df[feature_importances_df['Importance'] > 0]['Feature'].tolist()

with left:
    st.markdown("### Enter Patient Information")
    user_inputs = {}
    with st.container():
        # Only include input fields for important features
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
        if 'Female' in important_features:
            # Gender is handled by 'Male' input, this is just to ensure the column is created
            pass # No direct input needed as it's derived from 'Male'
        if 'Smoking_History_Encoded' in important_features:
            user_inputs['Smoking_History_Encoded'] = st.selectbox("Smoking History", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Skin_Cancer_Encoded' in important_features:
            user_inputs['Skin_Cancer_Encoded'] = st.selectbox("Skin Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Checkup_Encoded' in important_features:
            user_inputs['Checkup_Encoded'] = st.selectbox("Last Checkup", [0, 1, 2, 3, 4],
                               format_func=lambda x: ["Never", "5+ years ago", "Within past 5 years",
                                                      "Within past 2 years", "Within past year"][x])
        if 'Depression_Encoded' in important_features:
            user_inputs['Depression_Encoded'] = st.selectbox("Depression", [0, 1], format_func=lambda x: ["No", "Yes"][x])

        # Include other inputs that might not have importance but are needed for the model (like BMI calculation)
        if 'Height_(cm)' not in user_inputs and 'Height_(cm)' in important_features:
             user_inputs['Height_(cm)'] = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
        if 'Weight_(kg)' not in user_inputs and 'Weight_(kg)' in important_features:
            user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
        if 'BMI' not in user_inputs and 'BMI' in important_features:
            user_inputs['BMI'] = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        if 'Alcohol_Consumption' not in user_inputs and 'Alcohol_Consumption' in important_features:
            user_inputs['Alcohol_Consumption'] = st.number_input("Alcohol Consumption (drinks/week)", 0, 100, 0)
        if 'Fruit_Consumption' not in user_inputs and 'Fruit_Consumption' in important_features:
            user_inputs['Fruit_Consumption'] = st.number_input("Fruit Consumption (servings/day)", 0, 100, 30)
        if 'Green_Vegetables_Consumption' not in user_inputs and 'Green_Vegetables_Consumption' in important_features:
             user_inputs['Green_Vegetables_Consumption'] = st.number_input("Green Vegetables (servings/day)", 0, 100, 30)
        if 'FriedPotato_Consumption' not in user_inputs and 'FriedPotato_Consumption' in important_features:
            user_inputs['FriedPotato_Consumption'] = st.number_input("Fried Potato (servings/week)", 0, 100, 0)
        if 'Exercise_Encoded' not in user_inputs and 'Exercise_Encoded' in important_features:
            user_inputs['Exercise_Encoded'] = st.selectbox("Exercise", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'Other_Cancer_Encoded' not in user_inputs and 'Other_Cancer_Encoded' in important_features:
             user_inputs['Other_Cancer_Encoded'] = st.selectbox("Other Cancer", [0, 1], format_func=lambda x: ["No", "Yes"][x])
        if 'BMI_Category_Encoded' not in user_inputs and 'BMI_Category_Encoded' in important_features:
            # BMI Category is derived from BMI, this is just to ensure the column is created
            pass # No direct input needed as it's derived from BMI


with right:
    st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold;'>Prediction Result</div>", unsafe_allow_html=True)

    # Prepare input data based on user_inputs dictionary
    input_data = {}
    for feature in important_features:
        if feature in user_inputs:
            input_data[feature] = user_inputs[feature]
        else:
             # Handle features that are not directly input but needed for the model
             if feature == 'Female':
                 input_data['Female'] = 1 if user_inputs.get('Male', 'Male') == 'Female' else 0
             elif feature == 'Male':
                 input_data['Male'] = 1 if user_inputs.get('Male', 'Male') == 'Male' else 0
             elif feature == 'BMI_Category_Encoded':
                 bmi_val = user_inputs.get('BMI', 25.0) # Use default if BMI not in important features
                 if bmi_val < 18.5:
                     input_data['BMI_Category_Encoded'] = 0
                 elif 18.5 <= bmi_val < 25:
                     input_data['BMI_Category_Encoded'] = 1
                 elif 25 <= bmi_val < 30:
                     input_data['BMI_Category_Encoded'] = 2
                 else:
                     input_data['BMI_Category_Encoded'] = 0
             else:
                 # For other features not in user_inputs and not derived, set to 0 or a default value
                 input_data[feature] = 0 # Or a more appropriate default based on the feature


    if st.button("Predict", use_container_width=True):
        # Create DataFrame with all original features, filling missing important features with their values
        # and setting non-important features to 0 or a default
        # Ensure the column names and order match the training data used for the scaler
        full_input_features = [
            'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
            'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Age',
            'Checkup_Encoded', 'General_Health_Encoded', 'Exercise_Encoded', 'Skin_Cancer_Encoded', 'Other_Cancer_Encoded',
            'Depression_Encoded', 'Arthritis_Encoded', 'Diabetes_Encoded', 'Smoking_History_Encoded', 'Female', 'Male',
            'BMI_Category_Encoded'
        ]

        user_input_row = {}
        for feature in full_input_features:
            if feature in input_data:
                user_input_row[feature] = input_data[feature]
            elif feature == 'Female':
                 user_input_row['Female'] = 1 if user_inputs.get('Male', 'Male') == 'Female' else 0
            elif feature == 'Male':
                 user_input_row['Male'] = 1 if user_inputs.get('Male', 'Male') == 'Male' else 0
            elif feature == 'BMI_Category_Encoded':
                 bmi_val = user_inputs.get('BMI', 25.0) # Use default if BMI not in important features
                 if bmi_val < 18.5:
                     user_input_row['BMI_Category_Encoded'] = 0
                 elif 18.5 <= bmi_val < 25:
                     user_input_row['BMI_Category_Encoded'] = 1
                 elif 25 <= bmi_val < 30:
                     user_input_row['BMI_Category_Encoded'] = 2
                 else:
                     user_input_row['BMI_Category_Encoded'] = 0
            else:
                # Set non-important features that are not derived to 0
                user_input_row[feature] = 0


        # Create the DataFrame with the correct column order
        user_input_df = pd.DataFrame([user_input_row], columns=full_input_features)


        # Scale the input
        user_input_scaled = scaler.transform(user_input_df)

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
