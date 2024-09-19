import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set Streamlit page configuration at the top
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")


# Load the pre-trained pipeline (which includes preprocessing and the classifier)
try:
    with open('svc_model_.pkl', 'rb') as file:
        logreg_pipeline = pickle.load(file)
        logreg_pipeline.named_steps['preprocessor'].feature_names_in_ = np.array([
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
            'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 
            'Last Interaction'
        ])
except Exception as e:
    st.error(f"An error occurred while loading the model pipeline: {e}")
st.title('Customer Churn Prediction App')
st.markdown("""
    This app uses a pre-trained Support Vector Classifier model to predict customer churn based on user inputs. 
    Please enter the customer details in the sidebar and click the "Predict Churn" button.
""")

# Sidebar for user inputs
st.sidebar.header('Input Customer Details')
st.sidebar.markdown('Please enter the customer details below:')

# Numeric inputs in the sidebar
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
tenure = st.sidebar.number_input('Tenure (months)', min_value=0, value=12)
usage_frequency = st.sidebar.number_input('Usage Frequency', min_value=0, value=10)
support_calls = st.sidebar.number_input('Support Calls', min_value=0, value=1)
payment_delay = st.sidebar.number_input('Payment Delay (days)', min_value=0, value=0)
total_spend = st.sidebar.number_input('Total Spend', min_value=0.0, value=500.0)
last_interaction = st.sidebar.number_input('Last Interaction (days)', min_value=0, value=30)

# Categorical inputs in the sidebar
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
subscription_type = st.sidebar.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
contract_length = st.sidebar.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annual'])

# Collect input data into a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Tenure': [tenure],
    'Usage Frequency': [usage_frequency],
    'Support Calls': [support_calls],
    'Payment Delay': [payment_delay],
    'Subscription Type': [subscription_type],
    'Contract Length': [contract_length],
    'Total Spend': [total_spend],
    'Last Interaction': [last_interaction]
   
})
st.write(input_data)

# Predict button
if st.sidebar.button('Predict Churn'):
    # Make prediction using the loaded pipeline (it will handle preprocessing)
    try:
        prediction = logreg_pipeline.predict(input_data)
        prediction_proba = logreg_pipeline.predict_proba(input_data)[:, 1]

        # Display results
        if prediction[0] == 1:
            st.error(f"Prediction: The customer is likely to churn. Probability: {prediction_proba[0]:.2f}")
        else:
            st.success(f"Prediction: The customer is not likely to churn. Probability: {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("""
    ---
    **Note:** This prediction is based on the input data provided and the pre-trained model. For professional use, please ensure the data is accurate and complete.
""")
