import streamlit as st
import pandas as pd
import joblib

#load trained model
MODEL_PATH = "tourism_project/deployment/best_model.pkl"
model = joblib.load(MODEL_PATH)

#page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️"
)

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

#user inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)
city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
number_of_person_visiting = st.number_input(
    "Number of Person Visiting",
    min_value=1,
    max_value=20,
    value=2
)
preferred_property_star = st.number_input(
    "Preferred Property Star",
    min_value=1,
    max_value=5,
    value=3
)
marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced",'Unmarried']
)
number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=3
)
passport = st.selectbox(
    "Passport",
    [0, 1]
)
own_car = st.selectbox(
    "Own Car",
    [0, 1]
)
number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)
designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)
pitch_satisfaction_score = st.number_input(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=20,
    value=3
)
duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=15
)

#create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "ProductPitched": [product_pitched],
    "NumberOfFollowups": [number_of_followups],
    "DurationOfPitch": [duration_of_pitch]
})

#prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Customer is predicted to purchase the package.")
        if hasattr(model, "predict_proba"):
            st.write(f"Purchase probability: {probability:.2%}")
    else:
        st.info("Customer is predicted not to purchase the package.")
        if hasattr(model, "predict_proba"):
            st.write(f"Purchase probability: {probability:.2%}")

