import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.enums import AssetClass
from datetime import datetime, timezone, timedelta

import math
import traceback
import requests
import json
import time
import pytz
import threading
import pandas as pd
import numpy as np
from typing import List, Tuple
import re
import csv
import os

# Alpaca-py API Key
api_key = 'PKYSM0SOBK48QBC0KSQB'
api_secret = '3h8YhPEEsTiAl4X4PgppYAtsPsTdKutXFR0WqcRm'
api_base_url = 'https://paper-api.alpaca.markets'
symbol = 'BTC/USD'

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = '53O3CQJ465X7N5BW'


class TradingBot:
    def __init__(self, api_key, api_secret, api_base_url, symbol, timeframe, periods, multiplier, stop_loss_pct,
                 take_profit_pct, trade_pct):
        # instantiate Alpaca API
        self.api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
        self.trading_client = TradingClient(api_key, api_secret)
        self.Stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.Crypto_client = CryptoHistoricalDataClient(api_key, api_secret)

        # define variables
        self.symbol = symbol  # symbol to trade
        self.timeframe = timeframe  # timeframe for super trend indicator
        self.periods = periods  # periods for super trend indicator
        self.multiplier = multiplier  # multiplier for super trend indicator
        self.stop_loss_pct = stop_loss_pct  # stop loss percentage
        self.take_profit_pct = take_profit_pct  # take profit percentage
        self.trade_pct = trade_pct  # percentage of available funds per trade
        self.api_key = api_key
        self.api_secret = api_secret

        # Get our account information.
        account = self.trading_client.get_account()

        self.buying_power = float(account.buying_power)
        # Check our current balance vs. our balance at the last market close
        self.balance_change = float(account.equity) - float(account.last_equity)

        # create thread
        self.thread = threading.Thread(target=self.run, daemon=True)

        # start thread
        self.thread.start()

    # define function to calculate super trend indicator
    def calculate_super_trend(self):
        # get historical prices for symbol and timeframe
        response = requests.get(
            f'https://api.twelvedata.com/time_series?symbol={self.symbol}&interval={self.timeframe}&outputsize=5000&apikey=49873ea4e58840dfb31d5a70a54ced6f')
        prices = json.loads(response.content)['values']

        # calculate ATR and super trend
        atr = 0
        supertrend = []
        for i in range(len(prices)):
            if i < self.periods:
                supertrend.append(0)
                continue
            else:
                atr = sum(
                    [abs(prices[j]['high'] - prices[j]['low']) for j in range(i - self.periods, i)]) / self.periods
                if i == self.periods:
                    supertrend.append((prices[i]['high'] + prices[i]['low']) / 2)
                else:
                    if supertrend[-1] > prices[i]['low']:
                        supertrend.append(prices[i]['high'] - atr * self.multiplier)
                    else:
                        supertrend.append(prices[i]['low'] + atr * self.multiplier)
        return supertrend

    # define function to place bracket order
    def place_bracket_order(self, side, limit_price, stop_loss_price, take_profit_price):
        qty = int(self.buying_power * self.trade_pct / limit_price)
        if qty > 0:
            self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='gtc',
                limit_price=limit_price,
                order_class='bracket',
                stop_loss={'stop_price': stop_loss_price, 'limit_price': stop_loss_price},
                take_profit={'limit_price': take_profit_price}
            )

    # define function to run bot
    def run(self):
        while True:
            try:
                asset = self.trading_client.get_asset(self.symbol)
                if asset.asset_class == AssetClass.CRYPTO:
                    print('CRYPTO')
                    # Creating request object
                    client = CryptoHistoricalDataClient()
                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=[self.symbol],
                        timeframe=TimeFrame.Minute,
                        start=datetime.now(timezone.utc) - timedelta(days=1),
                        end=datetime.now(timezone.utc)
                    )
                    bars = client.get_crypto_bars(request_params)
                    bars = bars.df

                else:
                    # get current market price
                    ticker = self.api.get_last_trade(symbol)

                asset.fractionable
                asset.easy_to_borrow
                asset.tradable

                current_price = self.api.get_latest_trade(self.symbol)

                # .get_last_trade(self.symbol).price

                # calculate super trend indicator
                supertrend = self.calculate_super_trend()

                # calculate stop loss and take profit prices
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                take_profit_price = current_price * (1 + self.take_profit_pct)

                # check if price is above super trend indicator
                if current_price > supertrend[-1]:
                    # place long trade with bracket order
                    limit_price = supertrend[-1]
                    self.place_bracket_order('buy', limit_price, stop_loss_price, take_profit_price)

                # wait for next interval
                time.sleep(60)

            except Exception as e:
                print(e)
                time.sleep(60)


# Get an instance of an Alpaca trading client
alpaca_trading_client = TradingClient(api_key, api_secret)
# Get the Alpaca account information.
account = alpaca_trading_client.get_account()
# Get the account Buying Power
MARGIN_BUYING_POWER = float(account.buying_power)
# Get the account Buying Power
CASH_BUYING_POWER = float(account.cash)
# Check our current balance vs. our balance at the last market close
DAILY_BALANCE_CHANGE = float(account.equity) - float(account.last_equity)
# 4 percent of Margin is the total amount to use on a single trade when trading on MARGIN
MARGIN_TO_RISK_PER_TRADE = MARGIN_BUYING_POWER * 0.02
# 2 Percent of cash is the total amount to use on a single trade when trading on CASH
CASH_TO_RISK_PER_TRADE = CASH_BUYING_POWER * 0.02
# 20 percent of the account is the Maximum amount to use for all open trades
MARGIN_RISK_ON_ALL_TRADES = MARGIN_BUYING_POWER * 0.20
CASH_RISK_ON_ALL_TRADES = CASH_BUYING_POWER * 0.20

# Set the number of periods for the short and long moving averages
SHORT_PERIODS = 50
LONG_PERIODS = 200

# Set the amount of currency to use for each trade
CURRENCY_AMOUNT = 1000

SYMBOL = 'BTC/USD'


def log_to_csv(filename, message):
    # get current time
    now = datetime.now()

    # create CSV file if it doesn't already exist
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Time', 'Message'])

    # open CSV file for writing, in 'append' mode
    with open(filename, 'a', newline='') as csv_file:
        # create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # write log message to CSV file
        csv_writer.writerow([now, message])


def get_security_type(ticker_symbol):
    try:
        asset = alpaca_trading_client.get_asset(ticker_symbol)
    except:
        if "/usd" in ticker_symbol.lower():
            return 'Crypto'
        else:
            return 'Stock'

    if asset.asset_class == AssetClass.CRYPTO:
        return 'Crypto'
    elif asset.asset_class == AssetClass.US_EQUITY:
        return 'Stock'
    else:
        return 'None'


def can_be_fractionally_traded(ticker_symbol):
    try:
        asset_type = ""
        # Get the asset information
        try:
            asset = alpaca_trading_client.get_asset(ticker_symbol)
            if asset.asset_class == 'us_equity':
                asset_type = 'us_equity'
            else:
                asset_type = 'crypto'
        except:
            if "/usd" in ticker_symbol.lower():
                asset_type = 'crypto'
            else:
                asset_type = 'us_equity'

        if asset_type == 'us_equity' and asset.tradable and asset.fractionable:
            return True
        elif asset_type == 'crypto':
            return True
        else:
            return False

    except Exception as e:
        print(f"Error retrieving asset information: {e}")
        return False


# Define a function to get the current price of the symbol
def get_current_price(symbol, default_to_currency="USD"):
    # Get the first part before the '/' of symbols that look like this BTC/USD  or ETH/USD etc.
    parts = symbol.split("/")
    ticker_symbol = parts[0]
    # Check the type of Symbol
    symbol_type = get_security_type(symbol)
    if symbol_type == 'Crypto':
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={ticker_symbol}" \
              f"&to_currency={default_to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        crypto_price = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return float(crypto_price)
    elif symbol_type == 'Stock':
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_symbol}' \
              f'&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        stock_price = data['Global Quote']['05. price']
        return float(stock_price)
    else:
        return float(0.00)


# Define a function to get the historical prices of the symbol
def get_historical_prices(symbol, **kwargs):
    reverse_data = True
    # Get the arguments
    interval = kwargs.get('interval', '60min')
    time_series = kwargs.get('time_series', 'INTRADAY')
    get_all_prices = kwargs.get('get_all_prices', False)
    default_to_currency = kwargs.get('default_to_currency', 'USD')

    # Get the first part before the '/' of symbols that look like this BTC/USD  or ETH/USD etc.
    parts = symbol.split("/")
    ticker_symbol = parts[0]
    # is this a stock or Crypto
    sec_type = get_security_type(symbol)
    if sec_type == 'Stock':
        reverse_data = False
        # keys required for stock historical data client
        client = StockHistoricalDataClient(api_key, api_secret)
        # use regular expressions to match the number and text parts
        match = re.match(r'(\d+)(\w+)', interval)
        # extract the matched groups into a list
        timef = [match.group(1), match.group(2)]
        end = str(datetime.now(timezone.utc)).split(" ")[0] + " 00:00:00"
        start = str(datetime.now(timezone.utc) - timedelta(days=120)).split(" ")[0] + " 00:00:00"
        # multi symbol request - single symbol is similar
        request_params = StockBarsRequest(symbol_or_symbols=[symbol],
                                          timeframe=TimeFrame(int(timef[0]),
                                                              TimeFrameUnit(str(timef[1][0].upper() + timef[1][1:]))),
                                          start=start,
                                          end=end
                                          )

        bars = client.get_stock_bars(request_params).df
        if get_all_prices == False:
            bars = bars[['close']]
            bars.columns = ['price']
            return bars, reverse_data
        else:
            bars = bars.drop(columns=['volume'])
            bars = bars.drop(columns=['vwap'])
            bars = bars.drop(columns=['trade_count'])
            return bars, reverse_data
        #
        # url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{time_series}&symbol={ticker_symbol}' \
        #       f'&interval={interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
    elif sec_type == 'Crypto':
        if time_series == 'INTRADAY':
            function = 'CRYPTO_INTRADAY'
        else:
            function = f'DIGITAL_CURRENCY_{time_series}'
        url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker_symbol}' \
              f'&market={default_to_currency}&interval={interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
    else:
        return f'ERROR: the Symbol {symbol} can not be found', False
    # print(url)
    again = 1
    while (again):
        response = requests.get(url)
        data = response.json()
        output = json.dumps(data)
        if "error message" in output.lower():
            time.sleep(1 * 60)
            pass
        else:
            again = 0

    # get current UTC time
    utc_now = datetime.now(pytz.utc)
    # format the UTC time as a string
    utc_time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"Iteration ...{utc_time_str}")

    parse_str = 'Daily'
    if time_series == 'INTRADAY':
        parse_str = interval
    elif time_series == 'WEEKLY':
        parse_str = 'Weekly'
    elif time_series == 'MONTHLY':
        parse_str = 'Monthly'

    if sec_type == 'Stock':
        df = pd.DataFrame.from_dict(data[f'Time Series ({parse_str})'], orient='index')
    if sec_type == 'Crypto':
        try:
            df = pd.DataFrame.from_dict(data[f'Time Series Crypto ({parse_str})'], orient='index')
        except:
            print(f"1 ==> {url}")
            print(f"2 ==> {data}")
            print(f'3 ==> Time Series Crypto ({parse_str})')

    df = df.astype(float)

    # Assuming you have a DataFrame called 'df' with the mentioned column names
    column_names = list(df.columns)
    # Remove the numeric prefixes from the column names
    new_column_names = [name.split('. ')[1] for name in column_names]
    # Assign the new column names to the DataFrame
    df.columns = new_column_names

    # Calculate the True Range
    # df['true_range'] = true_range(df['high'], df['low'], df['close'])
    # super_df = supertrend(df)

    if not get_all_prices:
        df = df[['close']]
        df.columns = ['price']
        return df, reverse_data
    else:
        # Remove the Volume column
        df = df.drop(columns=['volume'])
        # Reverse the rows make the last first and the first last so that the data
        # goes from oldest at the top  to newest at the bottom
        reversed_df = df.iloc[::-1].reset_index(drop=False)
        # reversed_df['row'] = reversed_df.index
        # df['row'] = range(len(df))

        return reversed_df, reverse_data


# Define a function to get the historical prices of the symbol
def calculate_macd(symbol, interval='60min', fastpariod=26, slowperiod=100, signalperiod=5, type='close'):
    # Remove the '/' from Crypto name such as BTC/USD and convert to BTCUSD
    ticker_symbol = symbol.replace("/", "")
    # Build the API request string
    url = f'https://www.alphavantage.co/query?function=MACD&symbol={ticker_symbol}&interval={interval}' \
          f'&fastperiod={fastpariod}&slowperiod={slowperiod}&signalperiod={signalperiod}' \
          f'&series_type={type}&apikey={ALPHA_VANTAGE_API_KEY}'
    # Send the request
    response = requests.get(url)
    data = response.json()
    # Get the result data
    df = pd.DataFrame.from_dict(data[f'Technical Analysis: MACD'], orient='index')

    df = df.astype(float)
    MACD = df['MACD']
    MACD_Signal = df['MACD_Signal']
    MACD_Hist = df['MACD_Hist']
    return MACD, MACD_Signal, MACD_Hist


# Define a function to calculate the EMA for a given number of periods
def calculate_ema(symbol, period=200, interval='60min', type='close'):
    # Remove the '/' from Crypto name such as BTC/USD and convert to BTCUSD
    ticker_symbol = symbol.replace("/", "")
    # Build the API request string
    url = f'https://www.alphavantage.co/query?function=EMA&symbol={ticker_symbol}&interval={interval}' \
          f'&time_period={period}&series_type={type}&apikey={ALPHA_VANTAGE_API_KEY}'
    # Send the request
    response = requests.get(url)
    data = response.json()
    # Get the result data
    df = pd.DataFrame.from_dict(data[f'Technical Analysis: EMA'], orient='index')

    df = df.astype(float)
    ema = df['EMA']
    return ema


# Define a function to calculate the SMA for a given number of periods
def calculate_sma(symbol, period=200, interval='60min', type='close'):
    # Remove the '/' from Crypto name such as BTC/USD and convert to BTCUSD
    ticker_symbol = symbol.replace("/", "")
    # Build the API request string
    url = f'https://www.alphavantage.co/query?function=SMA&symbol={ticker_symbol}&interval={interval}' \
          f'&time_period={period}&series_type={type}&apikey={ALPHA_VANTAGE_API_KEY}'
    # Send the request
    response = requests.get(url)
    data = response.json()
    # Get the result data
    df = pd.DataFrame.from_dict(data[f'Technical Analysis: SMA'], orient='index')

    df = df.astype(float)
    sma = df['SMA']
    return sma


# Define a function to calculate the RSI for a given number of periods
def calculate_rsi(symbol, period=200, interval='60min', type='close'):
    # Remove the '/' from Crypto name such as BTC/USD and convert to BTCUSD
    ticker_symbol = symbol.replace("/", "")
    # Build the API request string
    url = f'https://www.alphavantage.co/query?function=RSI&symbol={ticker_symbol}&interval={interval}' \
          f'&time_period={period}&series_type={type}&apikey={ALPHA_VANTAGE_API_KEY}'
    # Send the request
    response = requests.get(url)
    data = response.json()
    # Get the result data
    df = pd.DataFrame.from_dict(data[f'Technical Analysis: RSI'], orient='index')

    df = df.astype(float)
    rsi = df['RSI']
    return rsi


# Check if the stock market is open or closed
def is_market_open(symbol, time_zone='US/Eastern'):
    # Create a timezone object for the US/Eastern timezone
    computer_timezone = pytz.timezone(time_zone)

    if get_security_type(symbol) == 'Crypto':
        # Create datetime objects for today's date and the local open/close times in the US/Eastern timezone
        c_open_datetime = computer_timezone.localize(
            datetime.combine(datetime.today(), datetime.strptime('00:00', '%H:%M').time())).astimezone(pytz.utc)
        c_close_datetime = computer_timezone.localize(
            datetime.combine(datetime.today(), datetime.strptime('23:59', '%H:%M').time())).astimezone(pytz.utc)
        return True, c_open_datetime, c_close_datetime
    # Remove the '/' from Crypto name such as BTC/USD and convert to BTCUSD
    ticker_symbol = symbol.replace("/", "")
    # Build the API request string
    url = f'https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHA_VANTAGE_API_KEY}'
    # Send the request
    response = requests.get(url)
    data = response.json()

    # Loop through the markets array and find the object for the United States region
    us_market = None
    for market in data['markets']:
        if market['region'] == 'United States':
            us_market = market
            break

    # Get the current status of the United States market
    if us_market is not None:
        us_status = us_market['current_status']
        open_time = us_market['local_open']
        close_time = us_market['local_close']

        # Create datetime objects for today's date and the local open/close times in the US/Eastern timezone
        utc_open_datetime = computer_timezone.localize(
            datetime.combine(datetime.today(), datetime.strptime(open_time, '%H:%M').time())).astimezone(pytz.utc)
        utc_close_datetime = computer_timezone.localize(
            datetime.combine(datetime.today(), datetime.strptime(close_time, '%H:%M').time())).astimezone(pytz.utc)
    else:
        us_status = 'unknown'
        utc_open_datetime = '00:00'
        utc_close_datetime = '00:00'

    if us_status == 'open':
        return True, utc_open_datetime, utc_close_datetime
    else:
        return False, utc_open_datetime, utc_close_datetime


# Define a function to place a buy order for the symbol
def place_buy_order(currency_amount):
    print(f'Placing buy order for {SYMBOL} with {currency_amount} currency...')
    # Place your buy order here using your preferred trading API


# Define a function to place a sell order for the symbol
def place_sell_order(currency_amount):
    print(f'Placing sell order for {SYMBOL} with {currency_amount} currency...')
    # Place your sell order here using your preferred trading API


# Define the main trading loop
def main():
    while True:
        # Get the current price and historical prices for the symbol
        current_price = get_current_price(SYMBOL)
        historical_prices = get_historical_prices(SYMBOL)

        # Calculate the short and long SMAs
        short_sma = calculate_sma(historical_prices, SHORT_PERIODS)
        long_sma = calculate_sma(historical_prices, LONG_PERIODS)

        # Determine the position of the short and long SMAs
        if short_sma.iloc[-1] > long_sma.iloc[-1]:
            position = 'above'
        else:
            position = 'below'

        res1 = short_sma.iloc[-1]
        res2 = long_sma.iloc[-1]
        print(str(res1))
        print(str(res2))

        # Check if we need to place a buy or sell order
        if short_sma.iloc[-1] > long_sma.iloc[-1] and position == 'below':
            place_buy_order(CURRENCY_AMOUNT)
        elif short_sma.iloc[-1] < long_sma.iloc[-1] and position == 'above':
            place_sell_order(CURRENCY_AMOUNT)

        # Wait for the next iteration
        time.sleep(60)


def ha_market_bias(data, show_ha=False, ha_len=100, ha_len2=100, osc_len=7):
    # Calculate Heikin Ashi values using exponential moving averages
    o = data['Open'].ewm(span=ha_len).mean()
    c = data['Close'].ewm(span=ha_len).mean()
    h = data['High'].ewm(span=ha_len).mean()
    l = data['Low'].ewm(span=ha_len).mean()

    haclose = (o + h + l + c) / 4
    haopen = ((o + c) / 2).shift(1)
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    hahigh = np.maximum(h, np.maximum(haopen, haclose))
    halow = np.minimum(l, np.minimum(haopen, haclose))

    # Calculate smoothed Heikin Ashi values
    o2 = haopen.ewm(span=ha_len2).mean()
    c2 = haclose.ewm(span=ha_len2).mean()
    h2 = hahigh.ewm(span=ha_len2).mean()
    l2 = halow.ewm(span=ha_len2).mean()

    ha_avg = (h2 + l2) / 2

    # Calculate the oscillator bias and oscillator smooth
    osc_bias = 100 * (c2 - o2)
    osc_smooth = osc_bias.ewm(span=osc_len).mean()

    # Determine the color for the oscillator bias plot
    sig_color = []
    for bias, smooth in zip(osc_bias, osc_smooth):
        if bias > 0 and bias >= smooth:
            sig_color.append('UP')
        elif bias > 0 and bias < smooth:
            sig_color.append('UP')
        elif bias < 0 and bias <= smooth:
            sig_color.append('DOWN')
        elif bias < 0 and bias > smooth:
            sig_color.append('DOWN')
        else:
            sig_color.append(np.nan)

    result = {'HA_Avg': ha_avg, 'Osc_Bias': osc_bias, 'Osc_Smooth': osc_smooth, 'Sig_Color': sig_color}
    if show_ha:
        result.update({'Open2': o2, 'High2': h2, 'Low2': l2, 'Close2': c2})

    return pd.DataFrame(result, index=data.index)


# ----------------------------------------------------------------------------------------------------------------------
#    SMART MONEY CONCEPTS  /////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------------------------------


def find_pivot_high_chcoh(data: pd.DataFrame, length: int) -> List[int]:
    pivot_high_chcoh = []
    for i in range(length, len(data) - length):
        if data['high'][i] > data['high'][i - length:i].max() and data['high'][i] > data['high'][
                                                                                    i + 1:i + length + 1].max():
            pivot_high_chcoh.append(i)
    return pivot_high_chcoh


def find_pivot_high_bos(data: pd.DataFrame, length: int) -> List[int]:
    pivot_high_bos = []
    for i in range(length, len(data) - length):
        if data['low'][i] > data['low'][i - length:i].max() and data['low'][i] > data['low'][
                                                                                 i + 1:i + length + 1].max():
            pivot_high_bos.append(i)
    return pivot_high_bos


def find_pivot_low_chcoh(data: pd.DataFrame, length: int) -> List[int]:
    pivot_low_chcoh = []
    for i in range(length, len(data) - length):
        if data['low'][i] < data['low'][i - length:i].min() and data['low'][i] < data['low'][
                                                                                 i + 1:i + length + 1].min():
            pivot_low_chcoh.append(i)
    return pivot_low_chcoh


def find_pivot_low_bos(data: pd.DataFrame, length: int) -> List[int]:
    pivot_low_bos = []
    for i in range(length, len(data) - length):
        if data['high'][i] < data['high'][i - length:i].min() and data['high'][i] < data['high'][
                                                                                    i + 1:i + length + 1].min():
            pivot_low_bos.append(i)
    return pivot_low_bos


def smart_money_concepts(data: pd.DataFrame, length: int = 5) -> Tuple[List[int], List[int]]:
    pivot_high_chcoh = find_pivot_high_chcoh(data, length)
    pivot_high_bos = find_pivot_high_bos(data, length)
    pivot_low_chcoh = find_pivot_low_chcoh(data, length)
    pivot_low_bos = find_pivot_low_bos(data, length)

    buy_signals = [i for i in pivot_high_chcoh if i + 1 in pivot_high_bos]
    sell_signals = [i for i in pivot_low_chcoh if i + 1 in pivot_low_bos]

    return buy_signals, sell_signals


# This function will calculate the number of units one can afford given cash to spend and
# latest price, and round it down according to order of the precision factor.
def calculate_order_size(cash_to_spend, latest_price):
    precision_factor = 10000
    units_to_buy = math.floor(cash_to_spend * precision_factor / latest_price)
    units_to_buy /= precision_factor
    return units_to_buy


# ----------------------------------------------------------------------------------------------------------------------
#   trend_follower indicator  //////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------------------------------
def supertrend(data, period=10, multiplier=3):
    """
    Returns the entry point to trade based on the SuperTrend strategy.

    Parameters:
    data (pandas.DataFrame): The data containing the high, low, and close prices.
    period (int): The period used to calculate the ATR and SuperTrend line. Default is 10.
    multiplier (float): The multiplier used to calculate the upper and lower bands. Default is 3.

    Returns:
    int: 1 if the entry signal is to buy, -1 if the entry signal is to sell, and 0 if there is no entry signal.
    """
    data['ATR'] = 0
    data['SuperTrend'] = 0
    data['Direction'] = 0
    data['Signal'] = 0

    # Calculate ATR
    for i in range(1, len(data)):
        tr = max(data['high'][i] - data['low'][i], abs(data['high'][i] - data['close'][i - 1]),
                 abs(data['low'][i] - data['close'][i - 1]))
        data['ATR'][i] = (data['ATR'][i - 1] * (period - 1) + tr) / period

    # Calculate SuperTrend
    for i in range(period, len(data)):
        upper_band = (data['high'][i] + data['low'][i]) / 2 + multiplier * data['ATR'][i]
        lower_band = (data['high'][i] + data['low'][i]) / 2 - multiplier * data['ATR'][i]
        if data['SuperTrend'][i - 1] == 0 or data['close'][i - 1] > data['SuperTrend'][i - 1]:
            data['SuperTrend'][i] = lower_band
        else:
            data['SuperTrend'][i] = upper_band
        if data['close'][i] > data['SuperTrend'][i]:
            data['Direction'][i] = 1
        else:
            data['Direction'][i] = -1

    # Generate entry signal
    for i in range(period, len(data)):
        if data['Direction'][i] == 1 and data['Direction'][i - 1] == -1:
            data['Signal'][i] = 1
        elif data['Direction'][i] == -1 and data['Direction'][i - 1] == 1:
            data['Signal'][i] = -1

    if data['Signal'][len(data) - 1] == 1:
        return 1
    elif data['Signal'][len(data) - 1] == -1:
        return -1
    else:
        return 0


def trend_follower_indicator(symbol, **kwargs) -> pd.DataFrame:
    # Get the arguments
    interval = kwargs.get('interval', '60min')
    time_series = kwargs.get('time_series', 'INTRADAY')
    get_all_prices = kwargs.get('get_all_prices', False)
    default_to_currency = kwargs.get('default_to_currency', 'USD')

    # Get historical data from Alpha Vantage API
    close, reverse = get_historical_prices(symbol, interval=interval,
                                           time_series=time_series,
                                           get_all_prices=get_all_prices,
                                           default_to_currency=default_to_currency)
    # rename the price column close
    close.columns = ['close']

    if reverse:
        # Reverse close series to oldest to newest . the df becomes an array
        close = close["close"].iloc[::-1]
        # Convert the array to a Pandas DataFrame
        close = pd.DataFrame(close, columns=['close'])

    # Define variables
    count_buy = 0
    count_sell = 0
    fast_ema_period = 9
    slow_ema_period = 21
    macd_fast_length = 12
    macd_slow_length = 26
    macd_signal_length = 9
    ema_length = 200
    ema_offset = 0

    # Calculate indicators
    fast_ema = close.ewm(span=fast_ema_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_ema_period, adjust=False).mean()

    macd_fast = close.ewm(span=macd_fast_length, adjust=False).mean()
    macd_slow = close.ewm(span=macd_slow_length, adjust=False).mean()
    macd = macd_fast - macd_slow
    macd_signal = macd.ewm(span=macd_signal_length, adjust=False).mean()
    macd_hist = macd - macd_signal

    ema = close.rolling(window=ema_length, min_periods=ema_length).mean()
    # smoothing_line = ema.ewm(span=5, adjust=False).mean()

    # Generate buy/sell signals
    buy_signal = (fast_ema >= slow_ema) & (close >= ema) & (macd_hist >= 0)
    sell_signal = (fast_ema < slow_ema) & (close < ema) & (macd_hist < 0)

    buy_signal['close'] = buy_signal['close'].replace({False: '', True: 'Buy'})
    sell_signal['close'] = sell_signal['close'].replace({False: '', True: 'Sell'})

    close['sig'] = buy_signal['close'] + sell_signal['close']

    # add new column 'signal'
    close['signal'] = ''

    # loop through rows and populate 'signal' column
    buy_seen = False
    sell_seen = False
    for i, row in close.iterrows():
        if (row['sig'] == 'Buy') & (buy_seen == False):
            close.loc[i, 'signal'] = 'Buy'
            buy_seen = True
            sell_seen = False
        elif (row['sig'] == 'Sell') & (sell_seen == False):
            close.loc[i, 'signal'] = 'Sell'
            sell_seen = True
            buy_seen = False
        elif row['sig'] == '':
            sell_seen = False
            buy_seen = False
        else:
            continue  # ignore cells with empty strings

    return close


# Manage Money
def manage_trades(symbol, stock_price,
                  max_per_trade=0.02, max_per_account=0.2, stop_loss=0.95,
                  take_profit=1.05):
    trading_client = TradingClient(api_key, api_secret)

    # Get our account information.
    my_account = trading_client.get_account()

    account_balance = float(my_account.buying_power)

    # Check our current balance vs. our balance at the last market close
    balance_change = float(my_account.equity) - float(my_account.last_equity)

    # Calculate the maximum amount of money to spend per trade
    max_amount_per_trade = account_balance * max_per_trade  # 5% of account balance



    # Calculate the maximum number of shares to buy
    max_shares_to_buy = int(max_amount_per_trade / stock_price)

    # Check if there are any open orders
    open_orders = trading_client.get_orders()
    open_positions = trading_client.get_all_positions()

    # Calculate the total amount of money tied up in open orders
    total_open_order_amount = 0
    for order in open_orders:
        total_open_order_amount += float(order.qty) * float(order.limit_price)

    # Calculate the total amount of money tied up in open positions
    total_open_position_amount = 0
    for position in open_positions:
        total_open_position_amount += float(position.qty) * float(position.current_price)

    # Calculate the maximum amount of money to spend on a new trade
    max_trade_amount = account_balance * max_per_account - total_open_order_amount - total_open_position_amount

    if max_trade_amount < max_amount_per_trade:
        print(f"Not enough available cash to make a new trade at {stock_price}")
        return None

    # Calculate the actual amount of money to spend on this trade
    actual_trade_amount = min(max_trade_amount, max_amount_per_trade)

    price = stock_price

    if can_be_fractionally_traded(symbol):
        if price > max_amount_per_trade:
            price = max_amount_per_trade
            # Calculate the actual amount of money to spend on this trade
            actual_trade_amount = min(max_trade_amount, max_amount_per_trade)

    # Calculate the actual number of shares to buy
    actual_shares_to_buy = int(actual_trade_amount / price)

    # Calculate the stop loss price and take profit price
    stop_loss_price = stock_price * stop_loss  # 5% stop loss
    take_profit_price = stock_price * take_profit  # 5% take profit

    # # Place the bracket order
    # order = api.submit_order(
    #     symbol='<stock-symbol>',
    #     qty=actual_shares_to_buy,
    #     side='buy',
    #     type='limit',
    #     time_in_force='gtc',
    #     limit_price=stock_price,
    #     stop_loss=dict(
    #         stop_price=stop_loss_price,
    #         limit_price=stop_loss_price,
    #     ),
    #     take_profit=dict(
    #         limit_price=take_profit_price,
    #     ),
    # )
    #
    # # Return the order information
    # return order
    print(
        f"\r\n max_trade_amount = account_balance * max_per_account - total_open_order_amount - total_open_position_amount "
        f"\r\n {max_trade_amount} = {account_balance} * {max_per_account} - {total_open_order_amount} - {total_open_position_amount} "
        f"\r\n max_trade_amount < max_amount_per_trade "
        f"\r\n {max_trade_amount} < {max_amount_per_trade} "
        f"\r\n account balance = {account_balance} "
        f"\r\n Number of Shares to buy = {actual_shares_to_buy} at price = {price}"
        f"\r\n Actual stock price = {stock_price} "
        f"\r\n stop_loss = {stop_loss_price} "
        f"\r\n take_profit = {take_profit_price}")
    return account_balance, actual_shares_to_buy, stock_price, stop_loss_price, take_profit_price


# Trade
def trade(symbol, filename='D:/code/python/trading_results/trading_record.xls'):
    while 1:
        try:
            stop_loss_pct = 0.0015  # 0.00139
            take_profit_pct = 0.003  # 0.00183
            percent_of_pot = 0.003  # 0.00083

            offset = 0.0008  # Tmp offset so that the data matches Trading view data

            # Check the indicator
            df1 = trend_follower_indicator(symbol, time_series='INTRADAY', interval='30min')

            # get the current symbol
            signal = df1.iloc[-1]['signal']
            # signal = df1.iloc[-1]['sig']

            # get current UTC time
            utc_now = datetime.now(pytz.utc)
            # format the UTC time as a string
            utc_time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')

            if is_market_open(symbol):
                message = ''
                # current_price = get_current_price(symbol)
                current_price = df1.iloc[-1]['close']
                limit_price = float("{:.2f}".format(current_price))
                offset_price = float("{:.2f}".format(limit_price * offset))
                limit_price = float("{:.2f}".format(limit_price + offset_price))

                if signal == 'Buy':

                    a, b, c, d, e = manage_trades(symbol, limit_price)

                    # calculate stop loss and take profit prices
                    stop_loss_price = "{:.2f}".format(limit_price * (1 - stop_loss_pct))
                    take_profit_price = "{:.2f}".format(limit_price * (1 + take_profit_pct))

                    message = f'Bought <{symbol}> @ ${limit_price} stop price = ${stop_loss_price}  take profit = ${take_profit_price} @ {utc_time_str}'
                    print(message)
                    pass
                elif signal == 'Sell':

                    a, b, c, d, e = manage_trades(symbol, limit_price)

                    # calculate stop loss and take profit prices
                    stop_loss_price = "{:.2f}".format(limit_price * (1 + stop_loss_pct))
                    take_profit_price = "{:.2f}".format(limit_price * (1 - take_profit_pct))

                    message = f'Sold <{symbol}> @ ${limit_price} stop price = ${stop_loss_price}  take profit = ${take_profit_price} @ {utc_time_str}'
                    print(message)
                    pass
                else:
                    # Debug
                    # calculate stop loss and take profit prices
                    # stop_loss_price = "{:.2f}".format(limit_price * (1 - stop_loss_pct))
                    # take_profit_price = "{:.2f}".format(limit_price * (1 + take_profit_pct))
                    # print(f'{limit_price}  ---  {take_profit_price} ---  {stop_loss_price}' )
                    # message = f'OPEN MARKET but NO TRADE was taken @ {utc_time_str}'
                    # print(message)
                    pass
                log_to_csv(filename, message)
            else:
                message = f'CLOSE MARKET for {symbol} so NO TRADE @ {utc_time_str}'
                print(message)
                log_to_csv(filename, message)
        except:
            # get current UTC time
            utc_now = datetime.now(pytz.utc)
            # format the UTC time as a string
            utc_time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f'An ERROR occured .. skipping this iteration @ {utc_time_str}'
            print(message)
            log_to_csv(filename, message)
            traceback.print_exc()
            pass
        # pause for period in minutes / 2
        time.sleep(30 * 60)


if __name__ == '__main__':
    symbol = "BTC/USD"
    symbol2 = "ETH/USD"
    # create a new thread
    thread1 = threading.Thread(target=trade(symbol))
    thread2 = threading.Thread(target=trade(symbol2))
    # start the thread
    thread1.start()
    thread2.start()

    # symbol = "BTC/USD"
    #
    # symbol = "SPY"
    # df1 = trend_follower_indicator(symbol, time_series='INTRADAY', interval='15min')
    # print(df1)
    #
    # symbol = "BTC/USD"
    # df2 = trend_follower_indicator(symbol, time_series='INTRADAY', interval='15min')
    # print(df2)

    pass

# # Example usage
# if __name__ == '__main__':
#     # Load historical price data as a pandas DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
#     historical_price_data = get_historical_prices('BTC/USD', get_all_prices=True)
#
#     buy_signals, sell_signals = smart_money_concepts(historical_price_data)
#
#     print('Buy signals:', buy_signals)
#     print('Sell signals:', sell_signals)

# ######################################################################################################################

# if __name__ == "__main__":
#     mstat1 = is_market_open('BTC/USD')
#     mstat2 = is_market_open('TSLA')
#
#     MACD1, MACD_Signal1, MACD_Hist1 = calculate_macd('BTC/USD')
#     MACD2, MACD_Signal2, MACD_Hist2 = calculate_macd('TSLA')
#
#     res = MACD_Hist1.iloc[0]
#
#     rsi1 = calculate_rsi('BTC/USD')
#     rsi2 = calculate_rsi('TSLA')
#
#     sma1 = calculate_sma('BTC/USD')
#     sma2 = calculate_sma('TSLA')
#
#     ema1 = calculate_ema('BTC/USD')
#     ema2 = calculate_ema('TSLA')
#
#     data2 = get_historical_prices('TSLA')
#     data1 = get_historical_prices('BTC/USD')
#
#     sec_type1 = get_security_type('BTC/USD')
#     sec_type2 = get_security_type('TSLA')
#
#     price1 = get_current_price('BTC/USD')
#     price2 = get_current_price('TSLA')
#
#     pass
#
#     main()
#     # instantiate TradingBot objects for AAPL, TSLA, and BTC/USD
#     # aapl_bot = TradingBot(api_key, api_secret, api_base_url, 'AAPL', '1day', 10, 3, 0.02, 0.05, 0.05)
#     # tsla_bot = TradingBot(api_key, api_secret, api_base_url, 'TSLA', '1day', 10, 3, 0.02, 0.05, 0.05)
#     # btc_bot = TradingBot(api_key, api_secret, api_base_url, 'BTC/USD', '1day', 10, 3, 0.02, 0.05, 0.05)
#
#     # run bots
#     # aapl_bot.run()
#     # tsla_bot.run()
#     # btc_bot.run()
