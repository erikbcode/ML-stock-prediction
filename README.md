# ML-stock-prediction
## ML Stock Prediction App

A project for CS 451 Data Science Course. This application forecasts stock prices using a myriad of Machine Learning models, leveraging historical stock prices, movements, VIX data, and more.
Features

- Forecast stock prices on various time horizons.
- Uses models ranging from LSTM, Logistic Regression, to Classifiers.
- Integrates a diverse set of data including:
        - VIX data
        - Historical stock prices
        - Stock movements

## Getting Started
### Prerequisites

- Python 3.8+ (It may work on older versions but this is the version it was tested on)
- Relevant Python libraries (e.g. pandas, numpy, tensorflow, keras, sklearn)

### Installation
Clone this repository:

git clone [repository-url]
cd [repository-name]

Install the required packages:

pip install -r requirements.txt

Run the application:
python main.py

### Usage
1. Load your stock data in the format sample_stock_data.csv.
2. Adjust model and prediction parameters as desired in config.json.
3. Run the app to train the models and forecast stock prices.
4. View results in output/forecast_results.csv.

Models Included

- LSTM (Long Short-Term Memory): A recurrent neural network used for sequence prediction problems.
- Logistic Regression: Used primarily to predict the direction of stock price movement.
- Classifiers: Predict stock price categories (e.g., bullish, bearish, neutral).


This application is meant for educational purposes only. Predictions made by this tool should not be considered as financial advice. Always consult with a financial advisor before making any trading decisions.
