
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

st.set_page_config(layout="wide", page_title="Cardiovascular Disease Prediction", page_icon="❤️")
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

def show_tableau_dashboard():
    tableau_html = """
        <div class='tableauPlaceholder' id='viz1754368306136' style='position: relative'>
        <noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ca&#47;cardiovasculardiseaserepresentation&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript>
        <object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
        <param name='embed_code_version' value='3' /> 
        <param name='site_root' value='' />
        <param name='name' value='cardiovasculardiseaserepresentation&#47;Dashboard1' />
        <param name='tabs' value='yes' /><param name='toolbar' value='yes' />
        <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ca&#47;cardiovasculardiseaserepresentation&#47;Dashboard1&#47;1.png' />
        <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' /><param name='language' value='en-US' />
        <param name='filter' value='publish=yes' /></object>
        </div>                
        <script type='text/javascript'>var divElement = document.getElementById('viz1754368306136');var vizElement = divElement.getElementsByTagName('object')[0];                    
        if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
        else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
        else { vizElement.style.width='100%';vizElement.style.minHeight='1500px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     
        var scriptElement = document.createElement('script');scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);                
        </script>
    """
    st.components.v1.html(tableau_html, height=900)

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if st.session_state["page"] == "Home":
    st.markdown("""
        <div style=" display: flex; justify-content: center; align-items: center; height: 40vh; text-align: center;">
            <div>
                <h1>Welcome to Cardiovascular Disease Prediction Application</h1>
                <p>Choose an option below to proceed</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        def go_to_prediction():
            st.session_state["page"] = "Heart Disease Prediction"
            st.experimental_rerun()
        if st.button("Predict Heart Disease ❤️", use_container_width=True):
            go_to_prediction()
    with col3:
        def go_to_dashboard():
            st.session_state["page"] = "View Dashboard"
            st.experimental_rerun()

        if st.button("View Dashboard 📊", use_container_width=True):
            go_to_dashboard()


elif st.session_state["page"] == "View Dashboard":
    st.markdown("<style>#dashboard-container { max-width: 100%; }</style>", unsafe_allow_html=True)
    st.subheader("Interactive Tableau Dashboard")
    with st.spinner("Loading Dashboard..."):
        time.sleep(2)
    show_tableau_dashboard()
    col1, col2, col3= st.columns([1, 1, 1])  
    with col2:
        def go_home():
            st.session_state["page"] = "Home"
            st.experimental_rerun()

        if st.button("Back to Home", use_container_width=True):
            go_home()


elif st.session_state["page"] == "Heart Disease Prediction":

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
        def go_home():
            st.session_state["page"] = "Home"
            st.experimental_rerun()

        if st.button("Back to Home", use_container_width=True):
            go_home()
    
    





























