import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide", page_title="Cardiovascular Disease Prediction", page_icon="❤️")

# Define page navigation
page = st.sidebar.radio("Navigation", ["🏠 Welcome", "🔍 Prediction", "📊 Dashboard"])

if page == "\ud83c\udfe0 Welcome":
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                color: white;
            }
            .main-title {
                font-size: 42px;
                font-weight: 700;
                background: -webkit-linear-gradient(45deg, #7F00FF, #E100FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 30px;
            }
            .welcome-box {
                background-color: rgba(255, 255, 255, 0.15);
                padding: 40px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>Welcome to Cardiovascular Disease Prediction App</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='welcome-box'>
        <p>This application helps in predicting the risk of cardiovascular disease based on user inputs.</p>
        <br>
        <a href="/?page=\ud83d\udd0d+Prediction" target="_self"><button style='padding:10px 20px; background-color:#00c9a7; color:white; font-weight:bold; border:none; border-radius:8px;'>Start Prediction</button></a>
        <br><br>
        <a href="/?page=\ud83d\udcca+Dashboard" target="_self"><button style='padding:10px 20px; background-color:#6c5ce7; color:white; font-weight:bold; border:none; border-radius:8px;'>Go to Dashboard</button></a>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Dashboard":
    st.markdown("<div class='main-title'>📊 Embedded Health Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""
    <iframe src="https://public.tableau.com/views/HeartHealthDashboard/Dashboard1" width="100%" height="600" frameborder="0"></iframe>
    """, unsafe_allow_html=True)

elif page == "🔍 Prediction":

    # Load the trained model
    try:
        with open('best_catboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure it's in the same directory.")
        st.stop()
    
    # Load the scaler
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
        st.stop()
    
    st.markdown("""
        <style>
            /* Main background */
            .stApp {
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                color: white;
            }
    
            /* Scrollable input section */
            .scrollable-container {
                max-height: 500px;
                overflow-y: auto;
                padding-right: 10px;
            }
    
            /* Input fields */
            .stNumberInput > div > input, .stSelectbox > div > div {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
            }
    
            /* Styled button */
            div.stButton > button {
                background-color: #00c9a7;
                color: white;
                font-weight: bold;
                border-radius: 10px;
                border: none;
                padding: 0.5rem 1rem;
            }
    
            div.stButton > button:hover {
                background-color: #00b49f;
            }
    
            /* Prediction result box */
            .prediction-text {
                background-color: rgba(255, 255, 255, 0.15);
                padding: 20px;
                margin-top: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
    
            .stAlert {
                margin-top: 20px;
                border-radius: 15px;
                font-weight: bold;
                padding: 10px;
            }
    
            /* Custom title styling */
            .main-title {
                font-size: 36px;
                font-weight: 700;
                background: -webkit-linear-gradient(45deg, #7F00FF, #E100FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 30px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Cardiovascular Disease Prediction")
    
    # Layout: left for inputs, spacer for gap, right for prediction
    left, gap, right = st.columns([1, 0.3, 2]) # Example ratio, can be adjusted
    
    
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
        st.markdown("#### Enter Patient Information")
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
    
            if 'Height_(cm)' in important_features:
                 user_inputs['Height_(cm)'] = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
            if 'Weight_(kg)' in important_features:
                user_inputs['Weight_(kg)'] = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
            if 'Height_(cm)' in important_features and 'Weight_(kg)' in important_features:
                height_cm = user_inputs.get('Height_(cm)', 170.0)
                weight_kg = user_inputs.get('Weight_(kg)', 70.0)
                bmi = weight_kg / ((height_cm / 100) ** 2)
                user_inputs['BMI'] = round(bmi, 2)
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
        st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; color: white;'>Prediction Result</div>", unsafe_allow_html=True)
    
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
                 f"""
        <div class='prediction-text' style='text-align: center; font-size: 22px; font-weight: bold; color: white;'>
            Probability of Heart Disease: {prediction_proba[0]*100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    
            )
    
            if prediction_proba[0] > 0.5:
              st.markdown(
            "<div style='text-align:center;'>"
            "<div class='stAlert' style='background-color:#fff3cd; color:#856404; padding: 6px; font-size:16px; border-radius: 6px;'>"
            "⚠️ Higher likelihood of heart disease."
            "</div></div>",
            unsafe_allow_html=True
        )
            else:
              st.markdown(
            "<div style='text-align:center;'>"
            "<div class='stAlert' style='background-color:#d1ecf1; color:#0c5460; padding: 6px;font-size:16px; border-radius: 6px;'>"
            "ℹ️ Lower likelihood of heart disease."
            "</div></div>",
            unsafe_allow_html=True
        )
    
    





