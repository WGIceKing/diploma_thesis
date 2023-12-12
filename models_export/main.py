import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product
from Models import ARIMAModel, ARIMA_GARCHModel, Kernel_Regression, Gaussian_Processes
DATA_PATH = "Data/" # path to folder with data

# Loading dataset with BTC prices.
btc_price = pd.read_csv(DATA_PATH + 'BTC_data.csv', low_memory = False)
btc_price['timestamp'] = pd.to_datetime(btc_price['timestamp']).dt.tz_localize(None)
print(btc_price.info())

# Initializing a models
ARIMA = ARIMAModel(order=(1,1,1))
ARIMA_GARCH = ARIMA_GARCHModel(order_ARIMA=(1,1,1), order_GARCH=(1,1))
KERNEL_REGRESSION = Kernel_Regression()
GAUSSIAN_PROCESSES = Gaussian_Processes()

# Fit Models with data from first day
# We give parameters, not values, to models! Models return a predictable rate change, not a numerical value.
ARIMA.fit(btc_price['returns'].iloc[:24])
ARIMA_GARCH.fit(btc_price['returns'].iloc[:24])
KERNEL_REGRESSION.fit(btc_price['returns'].iloc[:24])
GAUSSIAN_PROCESSES.fit(btc_price['returns'].iloc[:24])

# To get the prediction for the next bitcoin price:
for value in range (25,len(btc_price)):
    # Wait for the user to press Enter
    input("Press Enter for the next prediction: \n")

    # Make a predictions for the next price using the current value
    ARIMA_next_return = ARIMA.predict_next(btc_price['returns'].iloc[value])
    ARIMA_next_5 = ARIMA.forecast(5)
    ARIMA_GARCH_next_return = ARIMA_GARCH.predict_next(btc_price['returns'].iloc[value])
    ARIMA_GARCH_next_5 = ARIMA_GARCH.forecast(5)
    KERNEL_REGRESSION_next_return = KERNEL_REGRESSION.predict_next(btc_price['returns'].iloc[value])
    KERNEL_REGRESSION_next_5 = KERNEL_REGRESSION.forecast(5)
    GAUSSIAN_PROCESSES_next_return = GAUSSIAN_PROCESSES.predict_next(btc_price['returns'].iloc[value])
    GAUSSIAN_PROCESSES_next_5 = GAUSSIAN_PROCESSES.forecast(5)

    print(f"Time: {btc_price['timestamp'].iloc[value]}")
    print("-----------------ARIMA-----------------")
    # Print the next predicted return
    print(f"The predicted next return is: {ARIMA_next_return}")
    print(f"The predicted next 5 returns are: {ARIMA_next_5}")
    predicted_price = round(btc_price['price'].iloc[value] * (1 + ARIMA_next_return/100), 2)     # example of conversion to price
    print(f"The current price: {btc_price['price'].iloc[value]}, the next predicted price is: {predicted_price}")     # price forecast
 
    print("--------------ARIMA_GARCH--------------")
    # Print the next predicted 95% Confidence Interval
    print(f"The predicted 95% Confidence Interval is: {ARIMA_GARCH_next_return}")
    print(f"The predicted next 5 Confidence Intervals are: {ARIMA_GARCH_next_5[0]}, {ARIMA_GARCH_next_5[1]}")
    predicted_price_up = round(btc_price['price'].iloc[value] * (1 + ARIMA_GARCH_next_return[0]/100), 2)     # example of conversion to min price
    predicted_price_down = round(btc_price['price'].iloc[value] * (1 + ARIMA_GARCH_next_return[1]/100), 2)     # example of conversion to max price
    print(f"The current price: {btc_price['price'].iloc[value]}, the next predicted price in 95% confidence interval: ({predicted_price_up}, {predicted_price_down})")     # price forecast

    print("-----------Kernel Regression-----------")
    # Print the next predicted return
    print(f"The predicted next return is: {KERNEL_REGRESSION_next_return}")
    print(f"The predicted next 5 returns are: {KERNEL_REGRESSION_next_5}")
    predicted_price = round(btc_price['price'].iloc[value] * (1 + KERNEL_REGRESSION_next_return/100), 2)     # przykład przeliczenia na cenę
    print(f"The current price: {btc_price['price'].iloc[value]}, the next predicted price is: {predicted_price}")    # price forecast

    print("-----------Gaussian Processes----------")
    # Print the next predicted return
    print(f"The predicted next return is: {GAUSSIAN_PROCESSES_next_return}")
    print(f"The predicted next 5 returns are: {GAUSSIAN_PROCESSES_next_5}")
    predicted_price = round(btc_price['price'].iloc[value] * (1 + GAUSSIAN_PROCESSES_next_return/100), 2)     # example of conversion to price
    print(f"The current price: {btc_price['price'].iloc[value]}, the next predicted price is: {predicted_price}")     # price forecast
    print('\n')





