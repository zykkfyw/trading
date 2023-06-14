import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.enums import AssetClass
from datetime import datetime, timezone, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca.data.timeframe import TimeFrame as ttframe
from alpaca_trade_api.entity import Order
import multiprocessing
from threading import Lock

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
import uuid

# Alpaca-py API Key
api_key = 'PK2UYVA156FQMHCP0JCA'
api_secret = 'biNIVI7ZU5b2gTgSR65I2C7aCDIExPXpCyLkfo6S'
api_base_url = 'https://paper-api.alpaca.markets'

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = '53O3CQJ465X7N5BW'

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

# Configure the Trend Predictor algorithm
trend_fast_ema_period = 9
trend_slow_ema_period = 21
trend_macd_fast_length = 12
trend_macd_slow_length = 26
trend_macd_signal_length = 9
trend_ema_length = 20
trend_ema_offset = 0
saved_previous_price = {}


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


def get_historical_price_data(symbol, **kwargs):
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
        tfunit = str(timef[1][0].upper() + timef[1][1:])
        time_frame = ttframe(int(timef[0]), TimeFrameUnit(tfunit))

        # multi symbol request - single symbol is similar
        request_params = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=time_frame, start=start, end=end)

        cont = 0
        while cont == 0:
            try:
                bars = client.get_stock_bars(request_params).df
                cont = 1
            except:
                time.sleep(1)
                bars = client.get_stock_bars(request_params).df
                break

        if get_all_prices == False:
            bars = bars[['close']]
            bars.columns = ['price']
            return bars, reverse_data
        else:
            bars = bars.drop(columns=['volume'])
            bars = bars.drop(columns=['vwap'])
            bars = bars.drop(columns=['trade_count'])
            return bars, reverse_data

    elif sec_type == 'Crypto':
        reverse_data = False
        # keys required for stock historical data client
        client = CryptoHistoricalDataClient(api_key, api_secret)
        # use regular expressions to match the number and text parts
        match = re.match(r'(\d+)(\w+)', interval)
        # extract the matched groups into a list
        timef = [match.group(1), match.group(2)]
        end = str(datetime.now(timezone.utc)).split(" ")[0] + " 00:00:00"
        start = str(datetime.now(timezone.utc) - timedelta(days=120)).split(" ")[0] + " 00:00:00"
        tfunit = str(timef[1][0].upper() + timef[1][1:])
        time_frame = ttframe(int(timef[0]), TimeFrameUnit(tfunit))

        # multi symbol request - single symbol is similar
        request_params = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=time_frame, start=start, end=end)

        conn = 0
        while conn == 0:
            try:
                bars = client.get_crypto_bars(request_params).df
                conn +=1
            except:
                time.sleep(1)
                bars = client.get_crypto_bars(request_params).df
                break

        if not get_all_prices:
            bars = bars[['close']]
            bars.columns = ['price']
            return bars, reverse_data
        else:
            bars = bars.drop(columns=['volume'])
            bars = bars.drop(columns=['vwap'])
            bars = bars.drop(columns=['trade_count'])
            return bars, reverse_data

    else:
        return f'ERROR: the Symbol {symbol} can not be found', False
    # print(url)
    again = 1
    while (again):
        response = requests.get(url)
        data = response.json()
        output = json.dumps(data)
        if "error" in output.lower():
            time.sleep(1 * 60)
            pass
        else:
            again = 0

    # get current UTC time
    utc_now = datetime.now(pytz.utc)
    # format the UTC time as a string
    utc_time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"{symbol} --> Iteration ...{utc_time_str}")

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


def get_the_trend(symbol, **kwargs) -> pd.DataFrame:
    # Get the arguments
    interval = kwargs.get('interval', '60min')
    time_series = kwargs.get('time_series', 'INTRADAY')
    get_all_prices = kwargs.get('get_all_prices', False)
    default_to_currency = kwargs.get('default_to_currency', 'USD')

    # Get historical data from Alpha Vantage API
    close, reverse = get_historical_price_data(symbol, interval=interval,
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
    fast_ema_period = trend_fast_ema_period
    slow_ema_period = trend_slow_ema_period
    macd_fast_length = trend_macd_fast_length
    macd_slow_length = trend_macd_slow_length
    macd_signal_length = trend_macd_signal_length
    ema_length = trend_ema_length
    ema_offset = trend_ema_offset

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
    sig_history_list = []
    for i, row in close.iterrows():
        if (row['sig'] == 'Buy') & (buy_seen == False):
            close.loc[i, 'signal'] = 'Buy'
            sig_history_list.append('Buy')
            buy_seen = True
            sell_seen = False
        elif (row['sig'] == 'Sell') & (sell_seen == False):
            close.loc[i, 'signal'] = 'Sell'
            sig_history_list.append('Sell')
            sell_seen = True
            buy_seen = False
        elif row['sig'] == '':
            sell_seen = False
            buy_seen = False
        else:
            continue  # ignore cells with empty strings

    return close, sig_history_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def money_management(symbol, stock_price, fraction_to_buy=1, buy_sell='buy',
                     max_per_account=0.2, max_per_trade=0.02, stop_loss=0.02, take_profit=0.04):
    trading_client = TradingClient(api_key, api_secret)

    # Get our account information.
    my_account = trading_client.get_account()

    account_balance = float(my_account.buying_power)

    # Check our current balance vs. our balance at the last market close
    balance_change = float(my_account.equity) - float(my_account.last_equity)

    # Calculate the maximum amount of money to spend per trade
    max_amount_per_trade = account_balance * max_per_trade  # 2% of account balance

    # Check if there are any open orders
    total_open_position_amount = 0
    open_orders = trading_client.get_orders()
    try:
        open_positions = trading_client.get_all_positions()
        # Calculate the total amount of money tied up in open positions
        for position in open_positions:
            total_open_position_amount += float(position.qty) * float(position.current_price)
    except:
        pass

    # Calculate the total amount of money tied up in open orders
    total_open_order_amount = 0
    for order in open_orders:
        if order.symbol.lower() == symbol.lower():
            print(f'SKIPPING {symbol}...there is already an existing order for {symbol}')
            return None
        total_open_order_amount += float(order.qty) * float(order.limit_price)
        print(f'{total_open_order_amount} += {float(order.qty)} * {float(order.limit_price)}')

    # Calculate the maximum amount of money to spend on a new trade
    max_trade_amount = (account_balance * max_per_account) - total_open_order_amount - total_open_position_amount

    if max_trade_amount < max_amount_per_trade:
        print(
            f'max_trade_amount = ( account_balance * max_per_account ) - ( total_open_order_amount - total_open_position_amount )')
        print(
            f'{round(max_trade_amount, 2)} = ( {round(account_balance, 2)} * {max_per_account} ) - ( {round(total_open_order_amount, 2)} - {total_open_position_amount} ) ')
        print(f"Not enough available cash to make a new trade at {round(stock_price, 2)}")
        print(f"Account Balance =  {round(account_balance, 2)}")
        print(f"max_trade_amount=  {round(max_trade_amount, 2)}")
        print(f"Max available per Trade =  {round(max_amount_per_trade, 2)}")
        return None

    # Calculate the actual amount of money to spend on this trade
    actual_trade_amount = min(max_trade_amount, max_amount_per_trade)

    price_paid_per_share = stock_price
    actual_shares_to_buy = 1

    if can_be_fractionally_traded(symbol):
        if price_paid_per_share > max_amount_per_trade:
            price_paid_per_share = max_amount_per_trade
            if fraction_to_buy < 1:
                price_paid = price_paid_per_share * fraction_to_buy
            # Calculate the actual amount of money to spend on this trade
            actual_trade_amount = min(max_trade_amount, max_amount_per_trade)
            # Calculate the actual number of shares to buy
            actual_shares_to_buy = round(price_paid_per_share / stock_price, 2)
        else:
            actual_shares_to_buy = int(max_amount_per_trade / stock_price)
    else:
        if price_paid_per_share > max_amount_per_trade:
            price_paid_per_share = stock_price
            # Calculate the actual number of shares to buy
            actual_shares_to_buy = int(price_paid_per_share / stock_price)
        else:
            actual_shares_to_buy = int(max_amount_per_trade / stock_price)

    if str(buy_sell).lower() == 'buy':
        # Calculate the stop loss price and take profit price
        stop_loss_price = stock_price - (stock_price * stop_loss)  # 2% stop loss
        take_profit_price = stock_price + (stock_price * take_profit)  # 5% take profit
    elif str(buy_sell).lower() == 'sell':
        # Calculate the stop loss price and take profit price
        stop_loss_price = stock_price + (stock_price * stop_loss)  # 2% stop loss
        take_profit_price = stock_price - (stock_price * take_profit)  # 5% take profit
    else:
        print(f" money_management: Please specify the direction - Buy or Sell")
        return None

    total_cost = round(actual_shares_to_buy * price_paid_per_share)

    print(
        f"\r\n account balance = {round(account_balance, 2)} "
        f"\r\n max_trade_amount = account_balance * max_per_account - total_open_order_amount - total_open_position_amount "
        f"\r\n {round(max_trade_amount, 2)} = {round(account_balance, 2)} * {max_per_account} - "
        f"{total_open_order_amount} - {total_open_position_amount} "
        f"\r\n max_trade_amount < max_amount_per_trade "
        f"\r\n {round(max_trade_amount, 2)} < {round(max_amount_per_trade, 2)} "
        f"\r\n actual_trade_amount < price_paid_per_share "
        f"\r\n {round(actual_trade_amount, 2)} < {round(price_paid_per_share, 2)} "
        f"\r\n Market TREND = {buy_sell} "
        f"\r\n Number of Shares to buy = {actual_shares_to_buy} at price = {round(price_paid_per_share, 2)}"
        f"\r\n Total Cost for this trade = {total_cost}"
        f"\r\n Stock Price = {round(stock_price, 2)} "
        f"\r\n Spend Limit  = {round(max_amount_per_trade, 2)} "
        f"\r\n stop_loss = {round(stop_loss_price, 2)} "
        f"\r\n take_profit = {round(take_profit_price, 2)}")
    return account_balance, actual_shares_to_buy, stock_price, price_paid_per_share, stop_loss_price, take_profit_price


def api_get_current_price(api_key, api_secret, api_base_url, symbol):
    headers = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': api_secret}
    asset_class = 'crypto'
    asset_class = get_security_type(symbol)
    # Make the request
    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest'
    if asset_class.lower() == 'crypto':
        url = f'https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes?symbols={symbol}'
    idx = 0
    while idx < 4:
        idx += 1
        try:
            response = requests.get(url, headers=headers)
            break
        except:
            try:
                response = requests.get(url, headers=headers)
                break
            except:
                pass

    if response.status_code != 200:
        idx = 0
        while idx < 2:
            idx += 1
            response = requests.get(url, headers=headers)

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        if asset_class.lower() == 'crypto':
            current_price = data['quotes'][symbol]['ap']
        else:
            current_price = data['trade']['p']
        saved_previous_price[symbol] = current_price
        return current_price
    else:
        if symbol in saved_previous_price:
            print(f'[{symbol}] an error occurred while fetching the price. Returned previous price ${saved_previous_price[symbol]}')
            return saved_previous_price[symbol]
        else:
            print(f'[{symbol}] an error occurred while fetching the price Returned $0.00')
            return 0.00


class BracketOrder:
    def __init__(self, api, symbol, side, quantity, buy_price, stop_loss_price, take_profit_price, id, tapi_key=api_key,
                 tapi_secret=api_secret):
        self.api = api
        self.api_key = tapi_key
        self.api_secret = tapi_secret
        self.symbol = symbol
        self.quantity = quantity
        self.buy_price = buy_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.id = id
        self.side = side
        self.buy_order = ''
        self.trading_client = TradingClient(self.api_key, self.api_secret)

    def old_submit_order(self, side, type, price, quantity=1, order_id=''):
        return self.api.submit_order(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            type=type,
            client_order_id=order_id,
            time_in_force='gtc',
            limit_price=round(price, 2)
        )

    def submit_order(self, side, type, price, quantity=1, order_id=''):
        return self.api.submit_order(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            type=type,
            client_order_id=order_id,
            time_in_force='gtc'
        )

    def submit_limit_order(self, side, type, price, stop_price, quantity=1):
        return self.api.submit_order(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            type=type,
            time_in_force='gtc',
            stop_price=stop_price,
            limit_price=round(price, 2)
        )

    def monitor_order(self, order, lock):
        with lock:
            # Once the buy order is filled, send stop-loss and take-profit orders
            while str(order.status).lower() != 'filled' and str(order.status).lower() != 'new':
                order = self.api.get_order(order.id)
                time.sleep(1)
        print(f'{self.symbol} ... ORDER Status = {order.status} ...')

        # Monitor the status of the stop-loss and take-profit orders
        ct = 0
        with lock:
            while True:
                ct = ct + 1
                cprice = 0
                try:
                    cprice = api_get_current_price(api_key, api_secret, api_base_url, self.symbol)
                    cprice = round(float(cprice), 2)
                except:
                    cprice = api_get_current_price(api_key, api_secret, api_base_url, self.symbol)
                    cprice = round(float(cprice), 2)

                # Check the indicator to determine the current trend
                df1, df1_hist = get_the_trend(self.symbol, time_series='INTRADAY', interval='30min')
                # get the current signal
                signal = df1.iloc[-1]['signal']
                prev_signal = df1_hist[-2]

                if signal != '' and prev_signal != signal:
                    position = self.api.get_position(self.symbol.replace('/', ''))
                    self.api.close_position(position.symbol, qty=position.qty)
                    break

                if (cprice >= self.take_profit_price or cprice <= self.stop_loss_price) and str(
                        order.status).lower() != 'new':
                    if cprice >= self.take_profit_price:
                        print(f'{self.symbol} --- PROFIT --- @ {cprice}')
                    else:
                        print(f'{self.symbol} --- LOSS --- @ {cprice}')
                    print(f'{ct}:  => Current {self.symbol} price ==> {cprice} ==> status = {order.status} '
                          f'Waiting for -->  stop_loss_order @ {self.stop_loss_price} '
                          f'" OR  '
                          f'take_profit_order @ {self.take_profit_price} "')

                    position = self.api.get_position(self.symbol.replace('/', ''))
                    self.api.close_position(position.symbol, qty=position.qty)

                    time.sleep(1)
                    break
                else:
                    print(f'{ct}:  => Current {self.symbol} price ==> {cprice} ==> status = {order.status} '
                          f'Waiting for -->  stop_loss_order @ {self.stop_loss_price} '
                          f'" OR  '
                          f'take_profit_order @ {self.take_profit_price} "')
                    time.sleep(1)

    def execute(self):
        # Send initial buy order
        self.buy_order = self.submit_order(self.side, 'market', self.buy_price, self.quantity, self.id)
        # Monitor the status of the buy order
        # create shared lock
        lock = Lock()
        main_thread = threading.Thread(target=self.monitor_order, args=(self.buy_order, lock,))
        return main_thread


def is_asset_in_pofolio(api, symbol):
    try:
        position = api.get_position(symbol.replace('/', ''))
        return True
    except:
        return False


def close_all_active_trades(api):
    api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
    counter = 0
    print('CLOSING ALL OPEN POSITIONS AND ORDERS ...')
    while counter < 3:
        counter += 1
        api.close_all_positions()
        time.sleep(2)
        api.cancel_all_orders()
        time.sleep(1)
    print('COMPLETED CLOSING ALL OPEN POSITIONS AND ORDERS')


if __name__ == '__main__':
    # initialize the api
    api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
    # Close all active trades
    close_all_active_trades(api)
    while True:
        # initialize the lists
        assets = ['COMM', 'AAPL', 'TSLA', 'ETH/USD', 'SPY', 'BTC/USD', 'INTC']
        # assets = ['BTC/USD']
        threads = []
        trends = []

        # initialize other variables
        # get current UTC time
        utc_now = datetime.now(pytz.utc)
        # format the UTC time as a string
        utc_time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')

        print('-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-')
        print(f'Run # = {utc_time_str}')
        print('-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-v-')
        # Trade each ASSET
        for symbol in assets:
            try:
                # Can we trade this asset? is the market open to trade it?
                is_the_market_open_to_trade_asset, opened_at, closed_at = is_market_open(symbol)
                if not is_the_market_open_to_trade_asset:
                    continue

                # Generate a unique trade id
                client_id = "order_" + str(uuid.uuid4().hex)

                # Check the indicator to determine the current trend
                df1, df1_hist = get_the_trend(symbol, time_series='INTRADAY', interval='30min')
                # get the current signal
                signal = df1.iloc[-1]['signal']
                # get the previous signal
                prev_signal = df1_hist[-2]
                print(f'{symbol} Current trend = {signal} and previous trend = {prev_signal}')
                # if no current signal is given use the previous
                if signal == '':
                    signal = prev_signal

                asset = api.get_asset(symbol)
                shortable = getattr(asset, 'shortable')
                easy_to_borrow = getattr(asset, 'easy_to_borrow')

                if (shortable == 'False' or easy_to_borrow == 'False') and str(signal).lower() == 'sell':
                    print(f'This ASSET -> {symbol} can not be shorted .. SKIPPING purchase')
                    continue

                if str(signal).lower() == 'buy' or str(signal).lower() == 'sell':
                    # skip if asset in your portfolio.
                    if is_asset_in_pofolio(api, symbol):
                        print(f'ASSET -> {symbol} already exist in your portfolio .. SKIPPING purchase')
                        continue

                    # Get the current asset price
                    symbol_price = api_get_current_price(api_key, api_secret, api_base_url, symbol)


                    account_balance, \
                        actual_shares_to_buy, \
                        stock_price, \
                        price_paid_per_share, \
                        stop_loss_price, \
                        take_profit_price = \
                        money_management(symbol, symbol_price, 1, signal, 0.2, 0.02, 0.01, 0.02 )

                    print(f'-----------------------------------------------------------------------')
                    print(f'{symbol}')
                    print(f'Signal = {signal}  Prev+Signal = {prev_signal}')
                    print(f'symbol_price ==> {round(symbol_price, 2)}')
                    print(f'stop_price ==> {round(stop_loss_price)}')
                    print(f'teke_profit_limit_price ==> {round(take_profit_price, 2)}')
                    print(f'DIFF to LOSS ==> {round(symbol_price - stop_loss_price, 2)}')
                    print(f'DIFF to GAIN ==> {round(take_profit_price - symbol_price, 2)}')
                    print(f'\r')

                    if signal == 'Buy':
                        message = f'Bought <{symbol}> @ ${round(symbol_price, 2)} stop price = ${round(stop_loss_price, 2)}  take profit = ${round(take_profit_price, 2)} @ {utc_time_str}'
                        print(message)
                    elif signal == 'Sell':
                        message = f'Sold  <{symbol}> @ ${round(symbol_price, 2)} stop price = ${round(stop_loss_price, 2)}  take profit = ${round(take_profit_price, 2)} @ {utc_time_str}'
                        print(message)
                    else:
                        message = f'Nothing Purchased <{symbol}> @ ${round(symbol_price, 2)} stop price = $0.00  take profit = $0.00 @ {utc_time_str}'
                        print(message)

                    print(
                        f"\r\n<0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0>")


                    # Create BracketOrder object and execute
                    order = BracketOrder(api, symbol, str(signal).lower(), actual_shares_to_buy, symbol_price,
                                         round(stop_loss_price, 2),
                                         round(take_profit_price, 2),
                                         client_id)
                    t1 = order.execute()
                    threads.append(t1)
            except Exception as e:
                print(e)
                pass
        for thread in threads:  # iterates over the threads
            thread.start()  # waits until the thread has finished work

        for thread in threads:  # iterates over the threads
            thread.join()  # waits until the thread has finished work

        time.sleep(1)
