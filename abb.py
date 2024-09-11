import streamlit as st
import pandas as pd
import numpy
import pickle

with open('pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv('abb_task.csv')


gender = st.selectbox(label='Gender', options=['Male', 'Female'])
transaction = st.number_input(label='Count of Transaction', value=0, step=1)
average_interval = st.number_input(label='Transaction Interval', value=0, step=1)
balance = st.number_input(label='Current Balance', value=0, step=1)
median_cash = st.number_input(label='Median of Cash', value=0, step=1)
phone = st.selectbox(label='Phone', options=['ANDROID', 'IOS'])

input_features = pd.DataFrame({
    'gender': [gender],
    'count of transactions': [transaction],  
    'average day interval between atm transactions': [average_interval], 
    'current balance': [balance], 
    'median of cash transactions': [median_cash],
    'phone' : [phone]
})

if st.button('Predict'):

    try:
        prediction = model.predict(input_features)
        st.write(f'The prediction is: {prediction[0]}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
