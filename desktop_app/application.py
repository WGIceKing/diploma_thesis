import sys
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QSlider, QGridLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtGui import QFont
from Models.Models import ARIMAModel, ARIMA_GARCHModel, Kernel_Regression, Gaussian_Processes
from datetime import timedelta, datetime
from Models.predict import Predictor

MIN_WIDTH = 800
MIN_HEIGHT = 700

PREDICTION_NUM = 5

# Initializing a models
ARIMA = ARIMAModel(order=(1,1,1))
ARIMA_GARCH = ARIMA_GARCHModel(order_ARIMA=(1,1,1), order_GARCH=(1,1))
KERNEL_REGRESSION = Kernel_Regression()
GAUSSIAN_PROCESSES = Gaussian_Processes()
LSTM_PRED = Predictor()

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        # Database Connection setup
        self.db_connection = psycopg2.connect(
            host="localhost",
            database="bitcoin_app",
            user="postgres",
            password="admin"
        )
        self.cursor = self.db_connection.cursor()
        self.current_date = datetime.strptime('2019-06-11 04:00:00','%Y-%m-%d %H:%M:%S')
        self.current_price = 0

        query = f"SELECT date, price, returns, squared_returns FROM btc_data WHERE date >= '{self.current_date}'::timestamp - INTERVAL '5 days' and date <= '{self.current_date}'"
        # Execute the query and store the result in a pandas DataFrame
        btc_price = pd.read_sql_query(query, self.db_connection)
        # Loading dataset with BTC price [Hourly].
        btc_price['date'] = pd.to_datetime(btc_price['date']).dt.tz_localize(None)
        # Compute daily returns
        btc_price = btc_price.dropna()

        # Fit Models with data from first day
        # We give parameters, not values, to models! Models return a predictable rate change, not a numerical value.
        ARIMA.fit(btc_price['returns'].iloc[:48])
        ARIMA_GARCH.fit(btc_price['returns'].iloc[:48])
        KERNEL_REGRESSION.fit(btc_price['returns'].iloc[:48])
        GAUSSIAN_PROCESSES.fit(btc_price['returns'].iloc[:48])

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Bitcoin Price Predictor')
        self.resize(1000, 800)  # Set initial size of the window

        # Set the minimum size of the window to const values
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        self.cursor.execute("SELECT balance, bitcoin FROM user_tab WHERE id = %s", (1,))
        result = self.cursor.fetchone()
        if result:
            account_balance, bitcoin_balance = result
        self.cursor.execute(f"SELECT open FROM bitcoin_data_hourly WHERE date = '{self.current_date}'")
        result = self.cursor.fetchone()
        if result:
            self.current_price = result[0]

        grid = QGridLayout()

        # Place the date_label at the top-left (row 0, column 0)
        self.date_label = QLabel(f"Current date: {self.current_date}", self)
        grid.addWidget(self.date_label, 0, 0)
        self.price_label = QLabel(f"Current BTC price: ${self.current_price}", self)
        grid.addWidget(self.price_label, 0, 4)

        self.start_btn = QPushButton('>', self)
        grid.addWidget(self.start_btn, 0, 1)  # Adjust the position as needed

        self.stop_btn = QPushButton('||', self)
        grid.addWidget(self.stop_btn, 0, 2)  # Adjust the position as needed

        self.timer = QTimer(self)
        self.timer.setInterval(5000)  # 5 seconds in milliseconds
        self.timer.timeout.connect(self.update_date)

        self.start_btn.clicked.connect(self.start_action)
        self.stop_btn.clicked.connect(self.stop_action)

        # Bitcoin Price Chart (spanning multiple rows and columns)
        self.browser = QWebEngineView()
        grid.addWidget(self.browser, 1, 0, 8, 5)  # Spanning rows 1-8 and columns 0-4

        # Models suggestions 
        # Section for setting model's suggestion UI 
        self.arima_label = QLabel(f"ARIMA:", self)
        grid.addWidget(self.arima_label, 9, 0)  # Left half of the screen
        self.arima_pred = QPushButton('NEUTRAL', self)
        self.arima_pred.setFixedWidth(100)
        self.arima_pred.setStyleSheet("background-color: orange; color: black")
        # Create a QVBoxLayout
        arima_vbox = QVBoxLayout()
        arima_vbox.addWidget(self.arima_pred, alignment=Qt.AlignTop | Qt.AlignHCenter)  # Align top and horizontal center
        # Add the QVBoxLayout to the grid
        grid.addLayout(arima_vbox, 10, 0)  # Assuming you want to add it to row 10, column 0

        self.kernel_label = QLabel(f"KERNEL REG.:", self)
        grid.addWidget(self.kernel_label, 9, 1)  # Left half of the screen
        self.kernel_pred = QPushButton('NEUTRAL', self)
        self.kernel_pred.setFixedWidth(100)
        self.kernel_pred.setStyleSheet("background-color: orange; color: black")
        # Create a QVBoxLayout
        kernel_vbox = QVBoxLayout()
        kernel_vbox.addWidget(self.kernel_pred, alignment=Qt.AlignTop | Qt.AlignHCenter)  # Align top and horizontal center
        # Add the QVBoxLayout to the grid
        grid.addLayout(kernel_vbox, 10, 1)  # Assuming you want to add it to row 10, column 0

        self.gaussian_label = QLabel(f"GAUSSIAN:", self)
        grid.addWidget(self.gaussian_label, 9, 3)  # Left half of the screen
        self.gaussian_pred = QPushButton('NEUTRAL', self)
        self.gaussian_pred.setFixedWidth(100)
        self.gaussian_pred.setStyleSheet("background-color: orange; color: black")
        # Create a QVBoxLayout
        gaussian_vbox = QVBoxLayout()
        gaussian_vbox.addWidget(self.gaussian_pred, alignment=Qt.AlignTop | Qt.AlignHCenter)  # Align top and horizontal center
        # Add the QVBoxLayout to the grid
        grid.addLayout(gaussian_vbox, 10, 3)  # Assuming you want to add it to row 10, column 0

        self.arimagarch_label = QLabel(f"ARIMA/GARCH:", self)
        grid.addWidget(self.arimagarch_label, 9, 2)  # Middle of the screen
        self.arimagarch_pred = QPushButton('NEUTRAL', self)
        self.arimagarch_pred.setFixedWidth(100)
        self.arimagarch_pred.setStyleSheet("background-color: orange; color: black")
        # Create a QVBoxLayout
        arimagarch_vbox = QVBoxLayout()
        arimagarch_vbox.addWidget(self.arimagarch_pred, alignment=Qt.AlignTop | Qt.AlignHCenter)  # Align top and horizontal center
        # Add the QVBoxLayout to the grid
        grid.addLayout(arimagarch_vbox, 10, 2)  # Assuming you want to add it to row 10, column 2

        self.lstm_label = QLabel(f"LSTM:", self)
        grid.addWidget(self.lstm_label, 9, 4)  # Right half of the screen
        self.lstm_pred = QPushButton('NEUTRAL', self)
        self.lstm_pred.setFixedWidth(100)
        self.lstm_pred.setStyleSheet("background-color: orange; color: black")
        # Create a QVBoxLayout
        lstm_vbox = QVBoxLayout()
        lstm_vbox.addWidget(self.lstm_pred, alignment=Qt.AlignTop | Qt.AlignHCenter)  # Align top and horizontal center
        # Add the QVBoxLayout to the grid
        grid.addLayout(lstm_vbox, 10, 4)  # Assuming you want to add it to row 10, column 4

        # Balance Labels and Refresh Button
        self.balance_label = QLabel(f"Account Balance: ${account_balance}", self)
        grid.addWidget(self.balance_label, 11, 0, 1 ,2)  # Left half of the screen

        self.refresh_btn = QPushButton('Refresh Balances', self)
        self.refresh_btn.clicked.connect(self.refresh_balances)
        grid.addWidget(self.refresh_btn, 11, 2)  # Between balance_label and bitcoin_balance_label

        self.bitcoin_balance_label = QLabel(f"Bitcoin Balance: {bitcoin_balance if bitcoin_balance > 0 else 0} BTC", self)
        grid.addWidget(self.bitcoin_balance_label, 11, 3, 1, 2)  # Right half of the screen

        # Buy and Sell Components
        # Buy
        self.buy_amount_label = QLabel('Buy amount ($): 0', self)
        grid.addWidget(self.buy_amount_label, 12, 0, 1, 2)
        self.buy_slider = QSlider(Qt.Horizontal, self)
        self.buy_slider.setMinimum(0)
        account_balance_cents = int(float(account_balance) * 100)
        self.buy_slider.setMaximum(int(account_balance_cents))
        self.buy_slider.valueChanged.connect(self.update_buy_amount)
        grid.addWidget(self.buy_slider, 13, 0, 1, 2)
        self.buy_btn = QPushButton('Buy', self)
        grid.addWidget(self.buy_btn, 14, 0, 1, 2)

        # Sell
        self.sell_amount_label = QLabel('Sell amount (BTC): 0', self)
        grid.addWidget(self.sell_amount_label, 12, 3, 1, 2)
        self.sell_slider = QSlider(Qt.Horizontal, self)
        self.sell_slider.setMinimum(0)
        bitcoin_balance_units = int(float(bitcoin_balance) * 1e8)
        self.sell_slider.setMaximum(int(bitcoin_balance_units))
        self.sell_slider.valueChanged.connect(self.update_sell_amount)
        grid.addWidget(self.sell_slider, 13, 3, 1, 2)
        self.sell_btn = QPushButton('Sell', self)
        grid.addWidget(self.sell_btn, 14, 3, 1, 2)

        # Modify button click events
        self.buy_btn.clicked.connect(self.buy_bitcoins)
        self.sell_btn.clicked.connect(self.sell_bitcoins)

        # Design
        balance_labels = [self.balance_label, self.bitcoin_balance_label,self.sell_amount_label,self.buy_amount_label,self.price_label,self.date_label,
                          self.arima_label,self.arimagarch_label,self.lstm_label,self.kernel_label,self.gaussian_label]
        for label in balance_labels:
            label.setFont(QFont('Arial', 10))
            label.setStyleSheet("color: #333;")
            label.setAlignment(Qt.AlignCenter)

        # Set equal stretch factor for each column
        for i in range(5):  # Adjust the range if you have a different number of columns
            grid.setColumnStretch(i, 1)

        # Set the central widget with the grid layout
        central_widget = QWidget()
        central_widget.setLayout(grid)
        # [Rest of your code]
        self.setCentralWidget(central_widget)
        self.plot()
        self.show()

    def plot(self):
        # Create a query to select the required data
        query = f"SELECT date, open, high, low, close, volume_btc, volume_usd FROM bitcoin_data_hourly WHERE date >= '{self.current_date}'::timestamp - INTERVAL '1 days' and date <= '{self.current_date}'"

        # Execute the query and store the result in a pandas DataFrame
        df = pd.read_sql_query(query, self.db_connection)
        df.sort_values(by=['date'], ascending=False, ignore_index=True)

        query = f"SELECT date, price, returns, squared_returns FROM btc_data WHERE date >= '{self.current_date}'::timestamp - INTERVAL '1 days' and date <= '{self.current_date}'"
        btc_price2 = pd.read_sql_query(query, self.db_connection)

        query = f"SELECT text FROM tweets_data WHERE timestamp='{self.current_date}'"
        tweets = pd.read_sql_query(query, self.db_connection)
        predictions = []
        if tweets.size != 0:
            tweets = tweets['text'].to_list()
            predictions = LSTM_PRED.predictions(tweets)
        
        if len(predictions) != 0:
            # Calculate the mean
            mean = sum(predictions) / len(predictions)
        
        # Convert datetime to a readable format for the axis labels only
        # Generate a list of the first days of the months in your datetime range
        first_days_of_months = df['date'].dt.to_period('D').drop_duplicates().dt.to_timestamp()

        # Format the first days for ticktext
        ticktext = [date.strftime('%d-%b-%Y') for date in first_days_of_months]
        # Get the corresponding tickvals from the original datetime column
        tickvals = [date for date in first_days_of_months if date in df['date'].values]

        # Create a subplot with 2 rows and 1 column
        # The first subplot for the candlestick chart, the second for the volume bar chart
        # Create a candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('', 'Volume'), 
                            row_width=[0.2, 0.7])

        # Candlestick chart in the first row
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC price candlestick"), 
            row=1, col=1)

        # Volume bar chart in the second row
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume_usd'],
            name='Volume USD',
            marker_color='blue'), 
            row=2, col=1)
        
        # Last known closing price for further predictions
        last_known_price = df[df['date']==self.current_date]['open'].iloc[-1]  # Last known closing price

        # Arima next 5 predictions
        ARIMA_next_return = ARIMA.predict_next(btc_price2[btc_price2['date']==self.current_date]['returns'].iloc[-1])
        ARIMA_next_5 = ARIMA.forecast(5)
        for i in range(len(ARIMA_next_5)):
            if i != 0:
                ARIMA_next_5[i] = round(ARIMA_next_5[i-1] * (1 + ARIMA_next_5[i]/100), 2)
            else:
                ARIMA_next_5[i] = round(last_known_price * (1 + ARIMA_next_5[i]/100), 2)

        # Arima Garch next 5 predictions 
        ARIMA_GARCH_next_return = ARIMA_GARCH.predict_next(btc_price2[btc_price2['date']==self.current_date]['returns'].iloc[-1])
        ARIMA_GARCH_next_5 = ARIMA_GARCH.forecast(PREDICTION_NUM)
        ARGA_MIN_MAX = np.zeros((2,len(ARIMA_GARCH_next_5[0])))
        for i in range(len(ARIMA_GARCH_next_5[0])):
            if i == 0:
                ARGA_MIN_MAX[0][i] = round(last_known_price * (1 + ARIMA_GARCH_next_5[0][i]/100), 2)     # example of conversion to min price
                ARGA_MIN_MAX[1][i] = round(last_known_price * (1 + ARIMA_GARCH_next_5[1][i]/100), 2)     # example of conversion to max price
            else:
                ARGA_MIN_MAX[0][i] = round(ARGA_MIN_MAX[0][i-1] * (1 + ARIMA_GARCH_next_5[0][i]/100), 2)     # example of conversion to min price
                ARGA_MIN_MAX[1][i] = round(ARGA_MIN_MAX[1][i-1] * (1 + ARIMA_GARCH_next_5[1][i]/100), 2)     # example of conversion to max price

        # Kernel Regression next 5 predictions 
        KERNEL_REGRESSION_next_return = KERNEL_REGRESSION.predict_next(btc_price2[btc_price2['date']==self.current_date]['returns'].iloc[-1])
        KERNEL_REGRESSION_next_5 = KERNEL_REGRESSION.forecast(PREDICTION_NUM)
        for i in range(len(KERNEL_REGRESSION_next_5)):
            if i != 0:
                KERNEL_REGRESSION_next_5[i] = round(KERNEL_REGRESSION_next_5[i-1] * (1 + KERNEL_REGRESSION_next_5[i]/100), 2)
            else:
                KERNEL_REGRESSION_next_5[i] = round(last_known_price * (1 + KERNEL_REGRESSION_next_5[i]/100), 2)
        
        # Gaussian Processes next 5 predictions 
        GAUSSIAN_PROCESSES_next_return = GAUSSIAN_PROCESSES.predict_next(btc_price2[btc_price2['date']==self.current_date]['returns'].iloc[-1])
        GAUSSIAN_PROCESSES_next_5 = GAUSSIAN_PROCESSES.forecast(PREDICTION_NUM)
        for i in range(len(GAUSSIAN_PROCESSES_next_5)):
            if i != 0:
                GAUSSIAN_PROCESSES_next_5[i] = round(GAUSSIAN_PROCESSES_next_5[i-1] * (1 + GAUSSIAN_PROCESSES_next_5[i]/100), 2)
            else:
                GAUSSIAN_PROCESSES_next_5[i] = round(last_known_price * (1 + GAUSSIAN_PROCESSES_next_5[i]/100), 2)

        # Generate timestamps for forecasted data
        last_timestamp = df[df['date']==self.current_date]['date'].iloc[-1]
        forecast_timestamps = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, 9)]

        # Update arima prediction label 
        if (self.current_price < ARIMA_next_5[0]):
            self.arima_pred.setStyleSheet("background-color: green; color: black")
            self.arima_pred.setText("BUY")
        elif (self.current_price > ARIMA_next_5[0]):
            self.arima_pred.setStyleSheet("background-color: red; color: black")
            self.arima_pred.setText("SELL")
        else:
            self.arima_pred.setStyleSheet("background-color: orange; color: black")
            self.arima_pred.setText("NEUTRAL")

        # Update arima garch prediction label 
        if (ARGA_MIN_MAX[1][0]- last_known_price > last_known_price - ARGA_MIN_MAX[0][0]):
            self.arimagarch_pred.setStyleSheet("background-color: green; color: black")
            self.arimagarch_pred.setText("BUY")
        elif (ARGA_MIN_MAX[1][0]- last_known_price < last_known_price - ARGA_MIN_MAX[0][0]):
            self.arimagarch_pred.setStyleSheet("background-color: red; color: black")
            self.arimagarch_pred.setText("SELL")
        else:
            self.arimagarch_pred.setStyleSheet("background-color: orange; color: black")
            self.arimagarch_pred.setText("NEUTRAL")

        # Update Kernel Reg prediction label 
        if (self.current_price < KERNEL_REGRESSION_next_5[0]):
            self.kernel_pred.setStyleSheet("background-color: green; color: black")
            self.kernel_pred.setText("BUY")
        elif (self.current_price > KERNEL_REGRESSION_next_5[0]):
            self.kernel_pred.setStyleSheet("background-color: red; color: black")
            self.kernel_pred.setText("SELL")
        else:
            self.kernel_pred.setStyleSheet("background-color: orange; color: black")
            self.kernel_pred.setText("NEUTRAL")

        # Update Kernel Reg prediction label 
        if (self.current_price < GAUSSIAN_PROCESSES_next_5[0]):
            self.gaussian_pred.setStyleSheet("background-color: green; color: black")
            self.gaussian_pred.setText("BUY")
        elif (self.current_price > GAUSSIAN_PROCESSES_next_5[0]):
            self.gaussian_pred.setStyleSheet("background-color: red; color: black")
            self.gaussian_pred.setText("SELL")
        else:
            self.gaussian_pred.setStyleSheet("background-color: orange; color: black")
            self.gaussian_pred.setText("NEUTRAL")

        # Update lstm prediction label 
        if (len(predictions)!=0):
            if (mean >= 1.25):
                self.lstm_pred.setStyleSheet("background-color: green; color: black")
                self.lstm_pred.setText("BUY")
            elif (mean <= 0.75):
                self.lstm_pred.setStyleSheet("background-color: red; color: black")
                self.lstm_pred.setText("SELL")
            else:
                self.lstm_pred.setStyleSheet("background-color: orange; color: black")
                self.lstm_pred.setText("NEUTRAL")
        else:
            self.lstm_pred.setStyleSheet("background-color: gray; color: black")
            self.lstm_pred.setText("NO TWEETS")

        # Add a new trace for the forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=ARIMA_next_5,
            mode='lines+markers',
            name='ARIMA Price Forecast',
            line=dict(color='grey', width=2, dash='dash')
        ), row=1, col=1)

        # Add a new trace for the forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=GAUSSIAN_PROCESSES_next_5,
            mode='lines+markers',
            name='GAUSSIAN Price Forecast',
            line=dict(color='blue', width=2, dash='dash')
        ), row=1, col=1)

        # Add a new trace for the forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=KERNEL_REGRESSION_next_5,
            mode='lines+markers',
            name='Kernel Reg. Price Forecast',
            line=dict(color='black', width=2, dash='dash')
        ), row=1, col=1)

        # Customize layout
        fig.update_layout(
            title='Bitcoin Prices',
            yaxis_title='Price in USD',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False
        )

        # Plot the first line
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=ARGA_MIN_MAX[0],
                                mode='lines', name='ARIMA GARCH Threshold',
                                line=dict(color='rgba(107, 176, 125,0.2)')))

        # Plot the second line and fill the area between
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=ARGA_MIN_MAX[1],
                                mode='lines', name='ARIMA GARCH Threshold',
                                fill='tonexty',  # Fill to next trace
                                line=dict(color='rgba(107, 176, 125,0.2)')))

        # Customize x-axis for the candlestick subplot
        fig.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45,
            showgrid=True,
            gridwidth=1,
            row=1, col=1
        )

        # Customize x-axis for the volume bar chart subplot (optional, if they are shared it might not be necessary)
        fig.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            showgrid=True,
            gridwidth=1,
            row=2, col=1
        )

        # Customize y-axis for both subplots
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', row=1, col=1)
        fig.update_yaxes(title_text="Volume USD", showgrid=True, gridwidth=1, gridcolor='LightPink', row=2, col=1)

        fig.update_layout(title="Bitcoin Prices")

        # Save the figure to a temporary HTML file
        temp_filename = "C:\\Users\\kprze\\OneDrive\\Pulpit\\Uni\Diploma\\temp_plot.html"
        fig.write_html(temp_filename)

        # Load the HTML file using QWebEngineView
        self.browser.load(QUrl.fromLocalFile(temp_filename))

    def update_date(self):
        self.current_date = self.current_date + timedelta(hours=1)
        self.date_label.setText(f"Current date: {self.current_date}")

        self.cursor.execute(f"SELECT open FROM bitcoin_data_hourly WHERE date = '{self.current_date}'")
        result = self.cursor.fetchone()
        if result:
            self.current_price = result[0]
        self.price_label.setText(f"Current BTC price: ${self.current_price}")
        # Refresh the data that depends on the current date
        #self.refresh_data()  # You might need to implement this method
        self.plot()

    def update_buy_amount(self):
        amount_cents = self.buy_slider.value()
        amount_dollars = amount_cents / 100
        self.buy_amount_label.setText(f'Buy amount ($): {amount_dollars:.2f}')

    def update_sell_amount(self):
        amount_satoshis = self.sell_slider.value()
        amount_bitcoins = amount_satoshis / 1e8
        self.sell_amount_label.setText(f'Sell amount (BTC): {amount_bitcoins if amount_bitcoins>0 else 0:.8f}')

    def buy_bitcoins(self):
        self.cursor.execute("SELECT balance, bitcoin FROM user_tab WHERE id = %s", (1,))
        result = self.cursor.fetchone()
        if result:
            account_balance, bitcoin_balance = result
        amount_to_buy = float(self.buy_slider.value())/100
        current_price = float(self.current_price)
        bitcoins_to_buy = round(amount_to_buy / current_price if current_price else 0,8)

        # Update the database with the new balances
        self.cursor.execute(f"UPDATE user_tab SET balance={account_balance}-{amount_to_buy}, bitcoin={bitcoin_balance}+{bitcoins_to_buy} WHERE id = 1")
        self.refresh_balances()

    def sell_bitcoins(self):
        self.cursor.execute("SELECT balance, bitcoin FROM user_tab WHERE id = %s", (1,))
        result = self.cursor.fetchone()
        if result:
            account_balance, bitcoin_balance = result
        bitcoins_to_sell = float(self.sell_slider.value())/ 1e8
        current_price = float(self.current_price)
        amount_to_sell = round(bitcoins_to_sell * current_price,2)

        # Update the database with the new balances
        self.cursor.execute(f"UPDATE user_tab SET balance={account_balance}+{amount_to_sell}, bitcoin={bitcoin_balance}-{bitcoins_to_sell} WHERE id = 1")
        self.refresh_balances()

    def refresh_balances(self):
        # Fetches the balances from the database and updates the QLabels
        try:
            self.cursor.execute("SELECT balance, bitcoin FROM user_tab WHERE id = %s", (1,))
            result = self.cursor.fetchone()
            if result:
                account_balance, bitcoin_balance = result
                self.balance_label.setText(f"Account Balance: ${account_balance}")
                self.bitcoin_balance_label.setText(f"Bitcoin Balance: {bitcoin_balance if bitcoin_balance>0 else 0} BTC")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            self.db_connection.commit()
        
        # `account_balance` is a decimal.Decimal value from the database
        account_balance_cents = int(float(account_balance) * 100)  # Convert to cents and to integer
        self.buy_slider.setMaximum(int(account_balance_cents))

        # `bitcoin_balance` is a decimal.Decimal value
        bitcoin_balance_units = int(float(bitcoin_balance) * 1e8)  # Convert to satoshis and to integer
        self.sell_slider.setMaximum(int(bitcoin_balance_units))

    def set_start_button_active(self):
        self.start_btn.setStyleSheet("background-color: green; color: white;")

    def reset_start_button(self):
        self.start_btn.setStyleSheet("")

    def set_stop_button_active(self):
        self.stop_btn.setStyleSheet("background-color: red; color: white;")

    def reset_stop_button(self):
        self.stop_btn.setStyleSheet("")

    def start_action(self):
        self.set_start_button_active()
        self.reset_stop_button()
        self.timer.start()

    def stop_action(self):
        self.reset_start_button()
        self.set_stop_button_active()
        self.timer.stop()

    def closeEvent(self, event):
        # Properly close the database connection when the app is closed
        self.cursor.close()
        self.db_connection.close()
        super().closeEvent(event)

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())