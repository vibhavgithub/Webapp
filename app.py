import streamlit as st
import pandas as pd
import pickle

# --- Page config ---
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center;'>Cardiovascular Disease Prediction</h1>", unsafe_allow_html=True)

# Load trained model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_adaboost_model.pkl' not found.")
    st.stop()

# --- Layout columns ---
left_col, right_col = st.columns([2, 3])

# --- LEFT PANEL: Scrollable inputs ---
with left_col:
    st.markdown(
        """
        <style>
        .scrollable-left {
            max-height: 75vh;
            overflow-y: auto;
            padding-right: 15px;
            border-right: 2px solid #444;
        }
        </style>
        <div class="scrollable-left">
        """,
        unsafe_allow_html=True
    )

    st.subheader("Enter Patient Information")
    height = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=100, value=0)
    fruit_consumption = st.number_input("Fruit Consumption (servings/day)", min_value=0, max_value=100, value=30)
    green_vegetables_consumption = st.number_input("Green Veg. Consumption (servings/day)", min_value=0, max_value=100, value=30)
    fried_potato_consumption = st.number_input("Fried Potato Consumption (servings/week)", min_value=0, max_value=100, value=0)
    age = st.number_input("Age", min_value=18, max_value=120, value=50)
    checkup = st.selectbox("Last Checkup", options=[0,1,2,3,4],
                           format_func=lambda x: ["Never","5+ years ago","Within past 5 years","Within past 2 years","Within past year"][x])
    general_health = st.selectbox("General Health", options=[0,1,2,3,4],
                                  format_func=lambda x: ["Poor","Fair","Good","Very Good","Excellent"][x])
    exercise = st.selectbox("Exercise", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    skin_cancer = st.selectbox("Skin Cancer", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    other_cancer = st.selectbox("Other Cancer", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    depression = st.selectbox("Depression", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    arthritis = st.selectbox("Arthritis", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    diabetes = st.selectbox("Diabetes", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    smoking_history = st.selectbox("Smoking History", options=[0,1], format_func=lambda x: ["No","Yes"][x])
    gender = st.selectbox("Gender", options=["Female", "Male"])

    st.markdown("</div>", unsafe_allow_html=True)

# Gender encoding and BMI category
female = 1 if gender == "Female" else 0
male = 1 if gender == "Male" else 0
if bmi < 18.5:
    bmi_category = 0
elif 18.5 <= bmi < 25:
    bmi_category = 1
elif 25 <= bmi < 30:
    bmi_category = 2
else:
    bmi_category = 0

# --- RIGHT PANEL: Prediction & Dashboard ---
with right_col:
    st.markdown(
        """
        <style>
        .right-panel {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            height: 75vh;
            padding: 20px;
        }
        .prediction-card {
            background: #222;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        .dashboard-link {
            text-align: center;
            margin-top: 20px;
        }
        </style>
        <div class="right-panel">
        """,
        unsafe_allow_html=True
    )

    # Prediction Card
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if st.button("Predict", use_container_width=True):
        user_input = pd.DataFrame([[height, weight, bmi, alcohol_consumption, fruit_consumption,
                                    green_vegetables_consumption, fried_potato_consumption, age,
                                    checkup, general_health, exercise, skin_cancer, other_cancer,
                                    depression, arthritis, diabetes, smoking_history, female, male,
                                    bmi_category]],
                                  columns=model.feature_names_in_)
        prediction_proba = model.predict_proba(user_input)[:, 1]
        st.metric(label="Probability of Heart Disease", value=f"{prediction_proba[0]*100:.2f}%")
        if prediction_proba[0] > 0.5:
            st.warning("Higher likelihood of heart disease.")
        else:
            st.info("Lower likelihood of heart disease.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Always visible Dashboard Link
    st.markdown(
        """
        <div class="dashboard-link">
            <h4>Data Insights Dashboard</h4>
            <a href='https://your-dashboard-link.com' target='_blank' style='font-size:20px; font-weight:bold;'>
                📊 Open Data Insights Dashboard
            </a>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )
