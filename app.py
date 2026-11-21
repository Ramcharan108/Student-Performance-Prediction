
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
model_path = Path(__file__).parent / "student_performance_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# App title
st.title("🎓 Student Performance Prediction")

st.write(
    "A Machine Learning project using "
    "Balanced Logistic Regression."
)

st.info(
    "Educational demonstration only. "
    "Predictions should not be used for real "
    "student assessment decisions."
)

st.subheader("Enter Student Details")

# Student inputs
gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

ethnicity = st.selectbox(
    "Race/Ethnicity Group",
    ["group A", "group B", "group C", "group D", "group E"]
)

parent_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox(
    "Lunch Type",
    ["standard", "free/reduced"]
)

test_preparation = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

# Prediction button
if st.button("Predict Performance", type="primary"):

    student = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [ethnicity],
        "parental level of education": [parent_education],
        "lunch": [lunch],
        "test preparation course": [test_preparation]
    })

    prediction = model.predict(student)[0]

    probabilities = model.predict_proba(student)[0]

    st.subheader("Prediction Result")

    if prediction == "Pass":
        st.success("Predicted Result: PASS")
    else:
        st.warning("Predicted Result: FAIL")

    st.write("Model Class Scores")

    for label, probability in zip(
        model.classes_, probabilities
    ):
        st.write(
            f"{label}: {probability * 100:.2f}%"
        )

    st.caption(
        "These scores are model estimates, "
        "not reliable real-world probabilities."
    )
