import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("Diabetes Prediction")

# Input fields for each parameter
Pregnancies = st.number("Pregnancies",  min_value=0.0, max_value=17.0, value=0.0, step=0.1)
Glucose = st.number("Glucose", min_value=0.0, max_value=199.0, value=.0, step=0.1)
BloodPressure = st.number_input("BloosPressure", min_value=0.0, max_value=122.0, value=0.0, step=0.1)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=99, value=0)
Insulin = st.number_input("Insulin", min_value=0, max_value=846, value=0)
BMI = st.number_input("BMI", min_value=0.0, max_value=67.0, value=0, step=0.1)
DiabetesPedigreeFunction = st.selectbox("DiabetesPedigreeFunction", min_value=0, max_value=2.42, value=0)

# Map the gender and embarked values to numeric
#gender_map = {'male': 0, 'female': 1}
#embarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age,

}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_survival(input_data)

        if pred == 1:
            # Diabetes
            st.success(f"Prediction: Diabetes with probability {prob:.2f}")
        else:
            # Not Diabetes
            st.error(f"Prediction: Did not have Diabetes with probability {prob:.2f}")
