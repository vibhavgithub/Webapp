import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configure the page with a wider layout
st.set_page_config(layout="wide", page_title="Cardiovascular Disease Prediction", page_icon="❤️")

# Apply custom CSS for a new theme
st.markdown("""
    <style>
    /* General body styling */
    body {
        color: #333; /* Dark text color */
        background-color: #e0f2f7; /* Light blue background */
        font-family: 'Arial', sans-serif;
    }
    /* Main app container */
    .stApp {
        background-color: #e0f2f7; /* Match body background */
        padding-top: 0 !important; /* Remove top padding from the main app container */
    }
    /* Title styling */
    .stApp > header {
        background-color: #007ac1; /* Darker blue for header */
        padding: 15px;
        border-bottom: 2px solid #005b9f; /* Add a bottom border */
        margin-bottom: 0 !important; /* Remove bottom margin from header, using !important for override */
    }
    .stApp > header h1 {
        color: white; /* White text for title */
        text-align: center;
        margin: 0; /* Remove default margin */
        padding: 0; /* Remove default padding */
        margin-top: -30px; /* Adjust top margin to reduce gap above title */
    }
    /* Remove top margin from the main content area below the header - more specific selector */
    .stApp > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #004d80; /* Dark blue for headers */
        margin-top: 1.5rem; /* Add space above headers */
        margin-bottom: 1rem; /* Add space below headers */
    }
    /* Custom CSS for scrollable area */
    .scrollable-container {
        max-height: 600px; /* Adjust height as needed */
        overflow-y: auto;
        padding-right: 15px; /* Add some padding to the right for the scrollbar */
        border: 1px solid #b3e5fc; /* Light blue border */
        padding: 15px; /* Add padding inside the container */
        border-radius: 8px; /* Rounded corners */
        background-color: #ffffff; /* White background for the input area */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
    }
    /* Centering content in the right panel */
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px; /* Add some padding around centered content */
        border: 1px solid #b3e5fc; /* Light blue border */
        border-radius: 8px; /* Rounded corners */
        background-color: #ffffff; /* White background for the prediction area */
        margin-top: 20px; /* Add space above the prediction result */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
    }
    /* Style for the Predict button */
    .stButton>button {
        background-color: #00bcd4; /* Cyan background */
        color: white; /* White text */
        padding: 12px 28px; /* Increased padding */
        text-align: center; /* Centered text */
        text-decoration: none; /* Remove underline */
        display: inline-block;
        font-size: 18px; /* Increased font size */
        margin: 15px 2px; /* Increased margin */
        cursor: pointer; /* Mouse pointer on hover */
        border-radius: 8px; /* Rounded corners */
        border: none; /* Remove border */
        transition: background-color 0.3s ease; /* Smooth hover transition */
    }
     .stButton>button:hover {
        background-color: #00838f; /* Darker cyan on hover */
    }
    /* Style for the prediction result text */
    .prediction-text {
        font-size: 20px; /* Increased font size */
        font-weight: bold;
        margin-top: 20px; /* Space above the text */
        color: #00796b; /* Teal color for prediction text */
    }
     /* Style for the warning/info messages */
    .st-alert {
        margin-top: 15px;
        padding: 15px; /* Increased padding */
        border-radius: 8px; /* Rounded corners */
        font-size: 16px; /* Increased font size */
    }
    /* Style for input labels */
    .stTextInput label, .stSelectbox label, .stNumberInput label {
        font-weight: bold; /* Make labels bold */
        color: #004d80; /* Dark blue color for labels */
    }
    </style>
""", unsafe_allow_html=True)


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

# Layout: left for inputs (ratio 1), spacer for gap, right for prediction (ratio 3)
left, gap, right = st.columns([1, 0.3, 3]) # Adjusted ratio and gap


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
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
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
        # Ensure these are included if they are in important_features, even if not directly used for filtering inputs
        if 'Height_(cm)' in important_features:
             user_inputs['Height_(cm)'] = st.number_input("Height (cm)", min_value=50, max_value=300, value=170.0)
        if 'Weight_(kg)' in important_features:
            user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
        if 'BMI' in important_features:
            user_inputs['BMI'] = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
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
        if 'BMI_Category_Encoded' in important_features:
            # BMI Category is derived from BMI, this is just to ensure the column is created
            pass # No direct input needed as it's derived from 'Male'
        st.markdown('</div>', unsafe_allow_html=True)


with right:
    # Apply centering to the content within the right column
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 22px; font-weight: bold; color: black;'>Prediction Result</div>", unsafe_allow_html=True)

    # Prepare input data based on user_inputs dictionary
    # Initialize a dictionary with default values for all features
    user_input_data = {feature: 0 for feature in feature_names}

    # Update with user inputs for important features
    for feature, value in user_inputs.items():
        user_input_data[feature] = value

    # Handle derived features (Gender and BMI Category)
    user_input_data['Female'] = 1 if user_inputs.get('Male', 'Male') == 'Female' else 0
    user_input_data['Male'] = 1 if user_inputs.get('Male', 'Male') == 'Male' else 0

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
        # Create the DataFrame with the correct column order using feature_names
        user_input_df = pd.DataFrame([user_input_data], columns=feature_names)


        # Scale the input
        user_input_scaled = scaler.transform(user_input_df)

        # Make prediction
        prediction_proba = model.predict_proba(user_input_scaled)[:, 1]
        st.markdown(
            f"<div class='prediction-text'>Probability of Heart Disease: {prediction_proba[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        if prediction_proba[0] > 0.5:
            st.warning("Higher likelihood of heart disease.")
        else:
            st.info("Lower likelihood of heart disease.")

    # Add a link to the Data Insights Dashboard below the right panel
    st.markdown("---") # Add a horizontal rule for separation
    st.markdown("[Data Insights Dashboard Link](YOUR_DASHBOARD_LINK_HERE)") # Replace YOUR_DASHBOARD_LINK_HERE with the actual link

    st.markdown('</div>', unsafe_allow_html=True) # Close the centered-content div
