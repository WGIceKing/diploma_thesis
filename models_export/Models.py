import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, Matern, RationalQuadratic
from sklearn.metrics.pairwise import euclidean_distances

class ARIMAModel:
    def __init__(self, order):
        self.order = order  # ARIMA model order
        self.model = None
        self.model_fit = None
        self.history = []  # Historical data

    def fit(self, returns):
        # Fits the ARIMA model to the historical returns
        self.history = [x for x in returns]
        self.model = ARIMA(self.history, order=self.order)
        self.model_fit = self.model.fit()

    def predict_next(self, next_value):
        # Updates the model with a new actual observed value and predicts the next value
        self.history.append(next_value)
        self.model = ARIMA(self.history, order=self.order)
        self.model_fit = self.model.fit()  # Re-fit the model
        forecast = self.model_fit.forecast(steps=1)  # Forecast the next step
        return round(forecast[0], 3)  # Return the forecasted value

    def forecast(self, steps=1):
        # Makes predictions for a specified number of future steps
        forecast = self.model_fit.forecast(steps=steps)
        return np.round(forecast, 3)  # Return the forecasted values

class ARIMA_GARCHModel:
    def __init__(self, order_ARIMA, order_GARCH):
        self.order_ARIMA = order_ARIMA  # ARIMA model order
        self.order_GARCH = order_GARCH  # GARCH model order
        self.ARIMA_model = None
        self.GARCH_model = None
        self.ARIMA_model_fit = None
        self.GARCH_model_fit = None
        self.history = []  # Historical data

    def fit(self, returns):
        # Fits the models to the historical returns
        self.history = [x for x in returns]
        self.ARIMA_model = ARIMA(self.history, order=self.order_ARIMA) 
        self.ARIMA_model_fit = self.ARIMA_model.fit()
        # Get ARIMA residuals for GARCH
        residuals = self.ARIMA_model_fit.resid 
        self.GARCH_model = arch_model(residuals, vol='Garch', p = self.order_GARCH[0], q = self.order_GARCH[1])
        self.GARCH_model_fit = self.GARCH_model.fit(disp='off')

    def predict_next(self, next_value):
        # Updates the models with a new actual observed value and predicts the next value
        self.history.append(next_value)
        self.ARIMA_model = ARIMA(self.history, order=self.order_ARIMA)
        self.ARIMA_model_fit = self.ARIMA_model.fit()  # Re-fit the model
        ARIMA_forecast = self.ARIMA_model_fit.forecast(steps=1)  # Forecast the next step
        # Get ARIMA residuals for GARCH
        residuals = self.ARIMA_model_fit.resid 
        self.GARCH_model = arch_model(residuals, vol='Garch', p = self.order_GARCH[0], q = self.order_GARCH[1])
        self.GARCH_model_fit = self.GARCH_model.fit(disp='off')  # Re-fit the model
        GARCH_forecast = self.GARCH_model_fit.forecast(start=0, horizon=1, reindex=False)

        # Combining the forecasts
        lower_bound = ARIMA_forecast[0] - 1.96 * np.sqrt(GARCH_forecast.variance.values[-1,:][0])
        upper_bound = ARIMA_forecast[0] + 1.96 * np.sqrt(GARCH_forecast.variance.values[-1,:][0])
    
        return round(lower_bound,3), round(upper_bound,3)  # Return the forecasted values

    def forecast(self, steps=1):
        # Makes predictions for a specified number of future steps
        ARIMA_forecast = self.ARIMA_model_fit.forecast(steps=steps)
        residuals = self.ARIMA_model_fit.resid 
        self.GARCH_model = arch_model(residuals, vol='Garch', p = self.order_GARCH[0], q = self.order_GARCH[1])
        self.GARCH_model_fit = self.GARCH_model.fit(disp='off')
        GARCH_forecast = self.GARCH_model_fit.forecast(start=0, horizon=steps, reindex=False)
        lower_bounds = []
        upper_bounds = []
        for n in range(steps):
            lower_bounds.append(ARIMA_forecast[n] - 1.96 * np.sqrt(GARCH_forecast.variance.values[-1,:][n]))
            upper_bounds.append(ARIMA_forecast[n] + 1.96 * np.sqrt(GARCH_forecast.variance.values[-1,:][n]))

        return np.round(lower_bounds,3), np.round(upper_bounds,3)  # Return the forecasted values

class Kernel_Regression:
    def __init__(self):
        # self.model = None
        # self.model_fit = None
        self.history = []  # Historical data

    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-0.5 * (distance / bandwidth) ** 2)

    def nadaraya_watson_kernel_regression(self, X_train, y_train, X_pred, bandwidth=1.0):
        K = self.gaussian_kernel(euclidean_distances(X_train, X_pred), bandwidth)
        weights = K / np.sum(K, axis=0)
        return np.dot(weights.T, y_train)

    def fit(self, returns):
        # Fits the Kernel Regression model to the historical returns
        self.history = [x for x in returns]
        X_train = np.arange(len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history)
        X_pred = np.array([[len(self.history)]])
        bandwidth = 1.0  # Bandwidth for the Gaussian kernel
        y_pred = self.nadaraya_watson_kernel_regression(X_train, y_train, X_pred, bandwidth)

    def predict_next(self, next_value):
        # Updates the model with a new actual observed value and predicts the next value
        self.history.append(next_value)
        X_train = np.arange(len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history)
        X_pred = np.array([[len(self.history)]])
        bandwidth = 1.0  # Bandwidth for the Gaussian kernel
        y_pred = self.nadaraya_watson_kernel_regression(X_train, y_train, X_pred, bandwidth)
        forecast = y_pred[0]
        return round(forecast,3)  # Return the forecasted value

    def forecast(self, steps=1):
        # Makes predictions for a specified number of future steps
        X_train = np.arange(len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history)
        bandwidth = 1.0  # Bandwidth for the Gaussian kernel
        forecast = []
        for n in range(steps):
            X_pred = np.array([[len(self.history)+n]])
            forecast.append(self.nadaraya_watson_kernel_regression(X_train, y_train, X_pred, bandwidth)[0])

        return np.round(forecast,3)  # Return the forecasted values

class Gaussian_Processes:
    def __init__(self, max_history=24, length_scale=1.0, alpha=1e-10, n_restarts_optimizer=10):
        # Initialize the Gaussian Process model
        self.kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        self.history = []  # Historical data
        self.max_history = max_history  # Maximum history length

    def fit(self, returns):
        # Fits the Gaussian Process model to the historical returns
        self.history = [x for x in returns]
        X_train = np.arange(max(0, len(self.history) - self.max_history), len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history[-self.max_history:])
        self.model.fit(X_train, y_train)

    def predict_next(self, next_value):
        # Updates the model with a new actual observed value and predicts the next value
        self.history.append(next_value)
        X_train = np.arange(max(0, len(self.history) - self.max_history), len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history[-self.max_history:])
        self.model.fit(X_train, y_train)  # Re-fit the model
        prediction_index = np.array([[len(self.history)]])
        y_pred, _ = self.model.predict(prediction_index, return_std=True)
        return round(y_pred[0], 3)  # Return the forecasted value

    def forecast(self, steps=1):
        # Makes predictions for a specified number of future steps
        X_train = np.arange(max(0, len(self.history) - self.max_history), len(self.history)).reshape(-1, 1)
        y_train = np.array(self.history[-self.max_history:])
        self.model.fit(X_train, y_train)  # Re-fit the model with relevant historical data
        X_forecast = np.arange(len(self.history), len(self.history) + steps).reshape(-1, 1)
        forecast, _ = self.model.predict(X_forecast, return_std=True)
        return np.round(forecast, 3)  # Return the forecasted values