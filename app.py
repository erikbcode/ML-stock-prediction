import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas_datareader
from datetime import date


def predict_historical():
    # Use longer term model to predict historical
    window = 60
    model = tf.keras.models.load_model('./models/LSTMmodel_' + str(window))

    df = pandas_datareader.DataReader(ticker.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    df.reset_index(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] > pd.to_datetime(startdate))]




    closedata = df.reset_index()['Adj Close']
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    close_scaled = scaler.fit_transform(np.array(closedata).reshape(-1,1))
    training_size = round(len(close_scaled) * 0.75)
    training_data = close_scaled[:training_size]
    test_data = close_scaled[training_size:]
    X_train = []
    y_train = []
    X_test = []
    y_test = []


    for i in range(window, len(training_data)):
        # Training will use the trailing 60 days (can tweak this parameter)
        X_train.append(close_scaled[i-window:i, 0])
        y_train.append(close_scaled[i, 0])

    for i in range(len(training_data)+window, len(close_scaled)):
        # Testing will use the trailing 60 days (can tweak this parameter)
        X_test.append(close_scaled[i-window:i, 0])
        y_test.append(close_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    prediction = model.predict(X_test)
    mse = math.sqrt(mean_squared_error(y_test, prediction))
    predicted_price = scaler.inverse_transform(prediction)

    # Create prediction df where the date is matched up with the predicted closing price for test data
    prediction_df = pd.DataFrame(predicted_price, columns=['Close'])
    prediction_df['Date'] = df.iloc[training_size+window:]['Date'].values
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    stock_dateind = df.copy()
    stock_dateind['Date'] = pd.to_datetime(stock_dateind['Date'])
    stock_dateind = stock_dateind.set_index('Date')

    prediction = prediction_df.set_index(['Date'])

    # Same as window for training above
    look_back = window

    # Create the plot for model's predictions on the test data 
    visualize = plt.figure(figsize=(20,10))
    # Plot the actual price of the test data
    plt.plot(stock_dateind[training_size+look_back:len(close_scaled)]['Close'], label='Validation')
    # Plot the prediction on test data
    plt.plot(prediction['Close'], label='Prediction')
    # Plot the historicals 
    plt.plot(stock_dateind[:training_size+look_back]['Close'], label='Historical')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    st.pyplot(visualize)
    st.write('Root Mean Squared Error (RMSE) of historical:', mse)

def display_forecast():
    # Use shorter term model to do forecasting
    window = 30
    model = tf.keras.models.load_model('./models/LSTMmodelrecent_' + str(window))
    df = pandas_datareader.DataReader(ticker.lower(), 'yahoo', start=pd.to_datetime('2021-01-01'), end = date.today().strftime("%Y-%m-%d"))
    df.reset_index(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] > pd.to_datetime(startdate))]




    closedata = df.reset_index()['Adj Close']
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    close_scaled = scaler.fit_transform(np.array(closedata).reshape(-1,1))
    training_size = round(len(close_scaled) * 0.75)
    training_data = close_scaled[:training_size]
    test_data = close_scaled[training_size:]
    X_train = []
    y_train = []
    X_test = []
    y_test = []


    for i in range(window, len(close_scaled)):
        # Testing will use the trailing 'window' days (can tweak this parameter)
        X_test.append(close_scaled[i-window:i, 0])
        y_test.append(close_scaled[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    prediction = model.predict(X_test)
    mse = math.sqrt(mean_squared_error(y_test, prediction))
    predicted_price = scaler.inverse_transform(prediction)

    # Create prediction df where the date is matched up with the predicted closing price for test data
    prediction_df = pd.DataFrame(predicted_price, columns=['Close'])
    prediction_df['Date'] = df.iloc[window:]['Date'].values
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    stock_dateind = df.copy()
    stock_dateind['Date'] = pd.to_datetime(stock_dateind['Date'])
    stock_dateind = stock_dateind.set_index('Date')

    prediction = prediction_df.set_index(['Date'])

    # Do forecasting 
    forecast_days = 5

    start_forecast = X_test[-1]
    start_forecast = np.reshape(start_forecast, (start_forecast.shape[0], start_forecast.shape[1], 1))
    start_forecast = start_forecast.reshape(1, window, 1)
    forecast_list = start_forecast
    for i in range(forecast_days):
        x = forecast_list[-window:]
        x = x.reshape(1, window, 1)
        out = model.predict(x)[0][0]
        forecast_list = np.append(forecast_list, out)

    forecast_list = scaler.inverse_transform(forecast_list.copy().reshape(-1, 1))

    def forecast_dates(df):
        final = df['Date'].values[-1]
        forecast_dates = pd.date_range(final, periods=forecast_days).tolist()
        return forecast_dates
    
    forecast_df = pd.DataFrame(forecast_list[window:], columns=['Close'])
    forecast_df['Date'] = forecast_dates(df)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)

    # Same as window for training above
    look_back = window


    # Plot the forecast
    # Create the plot for model's predictions on the test data 
    visualize = plt.figure(figsize=(20,10))
    # Plot the actual price of the test data
    plt.plot(stock_dateind[-50:]['Close'], label='Validation')
    # Plot the prediction on test data
    plt.plot(prediction[-50:]['Close'], label='Prediction')
    # Plot the historicals 
    plt.plot(forecast_df['Close'], label='Forecast')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    st.pyplot(visualize)
    st.write('Root Mean Squared Error (RMSE) of historical:', mse)


st.title('Stock Price Prediction using Keras LSTM and Streamlit')

st.write('Enter the ticker of the company you want to predict and the start date for historical data below. The model\'s prediction for the'
+ ' most recent quarter of that data will be displayed, as well as the Root Mean Squared Error of the prediction compared to the actual price.')

st.write('ENTER COMPANY TICKER')
ticker = st.text_input(label='Enter stock ticker') 

#data = st.file_uploader('Upload here', type='csv')

st.write('SELECT START DATE')
startdate = st.date_input(label='Select the start date for historical price info that will be used. '
    + 'For a company such as AAPL it may be relevant to select the date that the iPhone was announced, since the ' 
    + 'business shifted in a major way at that time (as an example)', value=pd.to_datetime('2010-01-01'))


st.button(label='Predict Historical', on_click=predict_historical)

st.write('FORECAST')
st.write('Below button will display a 5-day forecast using a 15-day window LSTM model trained on data from the beginning of 2021.')
st.write('NOTE: Start date for prediction will always use 2021/01/01.')
st.button(label='Forecast Future', on_click=display_forecast)




    

