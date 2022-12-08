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

# Use the best model to perform predictions (50 day look-back trained on SPY)
window = 50
model = tf.keras.models.load_model('./models/LSTMmodelSPY_' + str(window))

def predict_historical():

    # Gather historical data 
    stock = pandas_datareader.DataReader(ticker.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    vix = pandas_datareader.DataReader('^vix'.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    tenyr = pandas_datareader.DataReader('^tnx'.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    stock.reset_index(inplace=True)
    vix.reset_index(inplace=True)
    tenyr.reset_index(inplace=True)
    
    # Convert date column to datetime from string so that it can easily be operated on
    stock['Date'] = pd.to_datetime(stock['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'])
    tenyr['Date'] = pd.to_datetime(tenyr['Date'])

    # Rename columns
    # Rename columns appropriately data 
    stock.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    vix.rename(columns={'Adj Close': 'VIX_AdjClose'}, inplace=True)
    tenyr.rename(columns={'Adj Close': 'TenYr_AdjClose'}, inplace=True)

    # Clean data
    stock.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
    vix.drop(['Volume', 'High', 'Low', 'Close', 'Open', 'Volume'], axis=1, inplace=True)
    tenyr.drop(['High', 'Low', 'Volume', 'Open', 'Close'], axis=1, inplace=True)

    # Join dataframes together based on date 
    spyvix = stock.merge(vix, how='inner', on=['Date'])
    df = spyvix.merge(tenyr, how='inner', on=['Date'])

    # Set index to date so that we can display a stock chart
    df.set_index('Date', inplace=True)

    # Normalize data 
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # We will normalize between 0 and 1
    data_scaled = scaler.fit_transform(df) # Perform normalization
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Select how much data we want to train / test on (75% for LSTM is appropriate)
    # Important to note that train and test sets must be near each other since this is time-series data
    training_size = round(len(data_scaled) * 0.75)
    training_data = data_scaled[:training_size]
    test_data = data_scaled[training_size:]

    for i in range(window, len(training_data)):
        # Training will use the trailing window days (can tweak this parameter)
        X_train.append(data_scaled[i-window:i, :])
        # Test uses the day following the previous window days' adjusted close
        y_train.append(data_scaled[i, 0])

    for i in range(len(training_data)+window, len(data_scaled)):
        # Testing will use the trailing window days (can tweak this parameter)
        X_test.append(data_scaled[i-window:i, :])
        y_test.append(data_scaled[i, 0])

    # Convert each dataset to numpy arrays again
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape into 3D arrays 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))


    # Perform predictions
    predicted_price = model.predict(X_test)
    
    # Get something that can be inverse transformed
    prediction_extended = np.zeros((len(predicted_price), 3))
    prediction_extended[:, 0] = predicted_price[:, 0]

    # Calculate root mean squared error for expected y values and the predictions from the model
    mse = math.sqrt(mean_squared_error(y_test, predicted_price))

    # Transform predicted normalized values back to prices in dollars
    predicted_price = scaler.inverse_transform(prediction_extended)

    # Create prediction df where the date is matched up with the predicted closing price for test data
    prediction_df = pd.DataFrame(predicted_price[:, 0], columns=['Close'])

    df.reset_index(inplace=True)

    prediction_df['Date'] = df.iloc[training_size+window:]['Date'].values
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    # Create new dataframe with the Date as the index (so that dates appear on x-axis of plots)
    df_dateind = df.copy()
    df_dateind['Date'] = pd.to_datetime(df_dateind['Date'])
    df_dateind = df_dateind.set_index('Date')

    # Set index of prediction dataframe to the Date so that it can be plotted with the entire dataset
    prediction = prediction_df.set_index(['Date'])

    # Create the plot for model's predictions on the test data 
    visualize = plt.figure(figsize=(20,10))
    # Plot the actual price of the test data
    plt.plot(df_dateind[training_size+window:len(data_scaled)]['AdjClose'], label='Validation')
    # Plot the prediction on test data
    plt.plot(prediction['Close'], label='Prediction')
    plt.plot(df_dateind[:training_size+window]['AdjClose'], label='Historical')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    st.pyplot(visualize)
    st.write('Root Mean Squared Error (RMSE) of historical:', mse)
    

# Method that displays a forecast as well as historical prediction data for a stock ticker 
def display_forecast():

    # Gather historical data    
    spy = pandas_datareader.DataReader(ticker.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    spy.reset_index(inplace=True)
    vix = pandas_datareader.DataReader('^vix'.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    vix.reset_index(inplace=True)
    tenyr = pandas_datareader.DataReader('^tnx'.lower(), 'yahoo', start=startdate, end = date.today().strftime("%Y-%m-%d"))
    tenyr.reset_index(inplace=True)

    # Convert date column to datetime from string so that it can easily be operated on
    spy['Date'] = pd.to_datetime(spy['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'])
    tenyr['Date'] = pd.to_datetime(tenyr['Date'])

    # Clean SPY data 
    spy.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    spy.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis=1, inplace=True)

    # Clean and relabel vix data 
    vix.rename(columns={'Adj Close': 'VIX_AdjClose'}, inplace=True)
    vix.drop(['Volume', 'High', 'Low', 'Close', 'Open', 'Volume'], axis=1, inplace=True)

    # Clean and relabel bond data 
    tenyr.rename(columns={'Adj Close': 'TenYr_AdjClose'}, inplace=True)
    tenyr.drop(['High', 'Low', 'Volume', 'Open', 'Close'], axis=1, inplace=True)

    # Join dataframes together based on date 
    spyvix = spy.merge(vix, how='inner', on=['Date'])
    df = spyvix.merge(tenyr, how='inner', on=['Date'])

    # Set index to date so that we can display a stock chart
    df.set_index('Date', inplace=True)

    # Normalize data 
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # We will normalize between 0 and 1
    data_scaled = scaler.fit_transform(df) # Perform normalization

    # Get recent means that will be inputted along with forecasted data
    vix_recentmean = data_scaled[-window:][:, 1].mean()
    tenyr_recentmean = data_scaled[-window:][:, 2].mean()

    # Train test split
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Select how much data we want to train / test on (75% for LSTM is appropriate)
    # Important to note that train and test sets must be near each other since this is time-series data
    training_size = round(len(data_scaled) * 0.75)
    training_data = data_scaled[:training_size]
    test_data = data_scaled[training_size:]

    for i in range(window, len(training_data)):
        # Training will use the trailing window days (can tweak this parameter)
        X_train.append(data_scaled[i-window:i, :])
        # Test uses the day following the previous window days' adjusted close
        y_train.append(data_scaled[i, 0])

    for i in range(len(training_data)+window, len(data_scaled)):
        # Testing will use the trailing window days (can tweak this parameter)
        X_test.append(data_scaled[i-window:i, :])
        y_test.append(data_scaled[i, 0])

    # Convert each dataset to numpy arrays again
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape into 3D arrays 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))
    
    prediction = model.predict(X_test)
    mse = math.sqrt(mean_squared_error(y_test, prediction))
    prediction_extended = np.zeros((len(prediction), 3))
    prediction_extended[:, 0] = prediction[:, 0]
    predicted_price = scaler.inverse_transform(prediction_extended)

    # Create prediction df where the date is matched up with the predicted closing price for test data
    prediction_df = pd.DataFrame(predicted_price[:, 0], columns=['Close'])
    df.reset_index(inplace=True)
    prediction_df['Date'] = df.iloc[training_size+window:]['Date'].values
    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

    stock_dateind = df.copy()
    stock_dateind['Date'] = pd.to_datetime(stock_dateind['Date'])
    stock_dateind = stock_dateind.set_index('Date')

    prediction_dateind = prediction_df.set_index(['Date'])

    # Begin forecasting the future using the most recent window of data
    start_forecast = X_test[-1]
    start_forecast = np.reshape(start_forecast, (start_forecast.shape[0], start_forecast.shape[1], 1))
    # Reshape so that data looks like X_test input from earlier cells 
    start_forecast = start_forecast.reshape(1, window, 3)

    forecast_list = start_forecast
    forecast_list.reshape(1, window, 3)

    # Number of days into the future that we will predict 
    forecast_days = 5

    # Perform actual forecasts
    for i in range(forecast_days):
        x = forecast_list[-window:] # Get the most recent "window" days (need this many for a prediction)
        out = model.predict(x) # Get the value of the next day that is predicted
        out = np.append(out, vix_recentmean)
        out = np.append(out, tenyr_recentmean)
        forecast_list = np.append(forecast_list[0], out)
        forecast_list = forecast_list.reshape(1, window+i+1, 3)
    

    forecast_list = scaler.inverse_transform(forecast_list.copy().reshape(-1, 3))

    # Define a function to get the dates of values that we are forecasting
    def forecast_dates(df):
        final = df['Date'].values[-1]
        forecast_dates = pd.date_range(final, periods=forecast_days).tolist()
        return forecast_dates
    
    forecast_df = pd.DataFrame(forecast_list[window:][: , 0], columns=['Close'])
    forecast_df['Date'] = forecast_dates(df.reset_index())
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)

    # Plot the forecast
    # Create the plot for model's predictions on the test data 
    visualize = plt.figure(figsize=(20,10))
    # Plot the actual price of the test data
    plt.plot(stock_dateind[-50:]['AdjClose'], label='Validation')
    # Plot the prediction on test data
    plt.plot(prediction_dateind[-50:]['Close'], label='Prediction')
    # Plot the historicals 
    plt.plot(forecast_df['Close'], label='Forecast')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    st.pyplot(visualize)
    st.write('Root Mean Squared Error (RMSE) of historical:', mse)


st.title('Stock Price Prediction using Keras LSTM and Streamlit')

st.write('IMPORTANT: This model was trained on SPY (an S&P 500 tracking ETF), and therefore may not be accurate for all stocks.')

st.write('Enter the ticker of the company you want to predict and the start date for historical data below. The model\'s prediction for the'
+ ' most recent quarter of that data will be displayed, as well as the Root Mean Squared Error of the prediction compared to the actual price.')


st.write('ENTER COMPANY TICKER')
ticker = st.text_input(label='Enter stock ticker') 

#data = st.file_uploader('Upload here', type='csv')

st.write('SELECT START DATE')
startdate = st.date_input(label='Select the start date for historical price info that will be used. '
    + 'For a company such as AAPL it may be relevant to select the date that the iPhone was announced, since the ' 
    + 'business shifted in a major way at that time (as an example)', value=pd.to_datetime('2010-01-01'))


st.button(label='Display Historical Prediction', on_click=predict_historical)

st.write('FORECAST')
st.write('The below button will display a 5-day forecast of the selected stock based on the model. Alternate forecast lengths coming soon.')
st.button(label='Forecast Future', on_click=display_forecast)




    

