import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Define date range for data fetching
start = '2010-01-01'
end = '2022-12-31' # Extended the end date for more data

# --- Streamlit App Layout ---

st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, TSLA)", 'AAPL')

# Fetch data from Yahoo Finance
try:
    df = yf.download(user_input, start=start, end=end)
    if df.empty:
        st.error("No data found for the given ticker. Please check the ticker symbol or the date range.")
    else:
        # --- Display Data and Basic Plots ---
        
        # Describing the data
        st.subheader(f'Data for {user_input} from {start} to {end}')
        st.write(df.describe())

        # Visualizations
        st.subheader('Closing Price vs. Time Chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs. Time with 100-Day Moving Average')
        ma100 = df['Close'].rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.plot(ma100, label='100-Day MA', color='red')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Closing Price vs. Time with 100 & 200-Day Moving Averages')
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.plot(ma100, label='100-Day MA', color='red')
        plt.plot(ma200, label='200-Day MA', color='green')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

        # --- Data Preprocessing and Model Prediction ---
        
        # Splitting data into Training (70%) and Testing (30%)
        data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Scaling the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit the scaler ONLY on the training data to prevent data leakage
        scaler.fit(data_train)
        
        # Transform the training and testing data
        data_train_array = scaler.transform(data_train)

        # Prepare the test data
        past_100_days = data_train.tail(100)
        final_df = pd.concat([past_100_days, data_test], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # --- Load Model and Make Predictions ---
        
        # Add a placeholder while the model loads and predicts
        with st.spinner('Loading Model and Making Predictions...'):
            model = load_model('keras_model.h5')
            
            # Making predictions
            y_predicted = model.predict(x_test)

            # Inverse transform to get original price values
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

        # --- Final Prediction Plot ---
        st.subheader('Original Price vs. Predicted Price')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")