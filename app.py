import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide")

# Custom CSS for scrollable left panel and fixed right panel
st.markdown("""
    <style>
    .left-panel {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 15px;
    }
    .right-panel {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model
try:
    with open('best_adaboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_adaboost_model.pkl' not found.")
    st.stop()

# Layout columns
left_col, right_col = st.columns([2, 3])

# LEFT: Scrollable inputs
with left_col:
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
    st.subheader("Enter Patient Information")
    height = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    # ... (all other inputs as before)
    st.markdown("</div>", unsafe_allow_html=True)

# Gender encoding, BMI category, etc.
female = 1 if st.session_state.get("gender") == "Female" else 0
male = 1 if st.session_state.get("gender") == "Male" else 0

# RIGHT: Fixed prediction area
with right_col:
    st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if st.button("Predict", use_container_width=True):
        user_input = pd.DataFrame([[height, weight, bmi]], columns=model.feature_names_in_)
        prediction_proba = model.predict_proba(user_input)[:, 1]
        st.metric("Probability of Heart Disease", f"{prediction_proba[0]*100:.2f}%")

    st.markdown("---")
    st.markdown("<h4>Data Insights Dashboard</h4>", unsafe_allow_html=True)
    st.markdown(
        "<a href='https://your-dashboard-link.com' target='_blank' style='font-size:20px; font-weight:bold;'>📊 Open Data Insights Dashboard</a>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
