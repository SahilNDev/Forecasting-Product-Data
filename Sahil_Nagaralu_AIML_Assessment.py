import streamlit as st
import pandas as pd
import pickle

# Load the Label Encoder
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Load the Best Model
model = pickle.load(open('model.pkl', 'rb'))

# Load the Data
data = pd.read_csv('dataset/Test Dataset.csv')
data.dropna(inplace=True)

st.title("Forecasting Sourcing Cost of Product Data")

select_pt = st.selectbox('Select ProductType', data['ProductType'].unique())
select_m = st.selectbox('Select Manufacturer', data['Manufacturer'].unique())
select_ac = st.selectbox('Select Area Code', data['Area Code'].unique())
select_sc = st.selectbox('Select Sourcing Channel', data['Sourcing Channel'].unique())
select_ps = st.selectbox('Select Product Size', data['Product Size'].unique())
select_prt = st.selectbox('Select Product Type', data['Product Type'].unique())
button = st.button('Predict')
if button:
    # Transform the Input
    input_data = pd.DataFrame({
        'ProductType': [select_pt],
        'Manufacturer': [select_m],
        'Area Code': [select_ac],
        'Sourcing Channel': [select_sc],
        'Product Size': [select_ps],
        'Product Type': [select_prt],
        'Year': [2021],
        'Month': [6]
    })

    # Label Encode the Input
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            encoder = encoders[col]
            input_data[col] = encoder.transform(input_data[col])

    # Make the Prediction
    prediction = model.predict(input_data)

    # Display the Prediction
    st.write(f"The Predicted Sourcing Cost is: {prediction[0]}")

    # Actual Sourcing Cost
    try:
        actual_cost = data[(data['ProductType'] == select_pt) & (data['Manufacturer'] == select_m) & (data['Area Code'] == select_ac) & (data['Sourcing Channel'] == select_sc) & (data['Product Size'] == select_ps) & (data['Product Type'] == select_prt) & (data['Month of Sourcing'] == "Jun-21")]['Sourcing Cost'].values[0]
        st.write(f"The Actual Sourcing Cost is: {actual_cost}")
    except Exception as e:
        st.write("Actual Sourcing Cost not available")