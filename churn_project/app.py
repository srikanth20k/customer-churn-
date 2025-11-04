import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder



model = tf.keras.models.load_model('churn_model.h5')
import os

file_path = 'churn_model.h5'  # Adjust as needed
if not os.path.isfile(file_path):
    st.error(f"Model file not found at {file_path}")
else:
    model = tf.keras.models.load_model(file_path)
    st.success("Model loaded successfully.")



# Load the scaler and encoder objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
    
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


st.title("Customer Churn Prediction")

# Input fields


geography = st.selectbox("Geography", encoder.categories_[0])
gender = st.selectbox("Gender",label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([[gender]])[0]],
    # 'Geography': [geography], 
    'Age': [age],   
    'Tenure': [tenure],            
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})


## One-hot encode categorical variables
geo_encoded = encoder.transform([[geography]])
geo_df = pd.DataFrame(geo_encoded, columns=encoder.get_feature_names_out(['Geography']))

## Combine with the original input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        border: black;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        align-items: center;
        align-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """<style>
    .stselectbox>selectbox:hover {
        background-color: blue;
        color: green;
    }
    </style>""",
    unsafe_allow_html=True
)
# Predict churn

if st.button("Predict Churn",type='primary'):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    st.subheader("Churn Probability")
    st.write(f"{churn_probability:.2%}")
    if churn_probability > 0.5:
        st.warning("The customer is likely to churn. ⚠️")
    else:
        st.success("The customer is unlikely to churn. ✅") 
        st.balloons()