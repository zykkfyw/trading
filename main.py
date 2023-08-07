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
api_key = 'PKMX7R0XCGIOA2MJ2S1O'
api_secret = 'zyqTbKJ5T6l5TrPUw004cxZ7zJyyIv2Tgi2uva2l'
api_base_url = 'https://paper-api.alpaca.markets'

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = '53O3CQJ465X7N5BW'

# Get an instance of an Alpaca trading client
alpaca_trading_client = TradingClient(api_key, api_secret)

#  Enable Logging to screen
print_additional_log = False
# Configure the Trend Predictor algorithm
trend_fast_ema_period = 9
trend_slow_ema_period = 21
trend_macd_fast_length = 12
trend_macd_slow_length = 26
trend_macd_signal_length = 9
trend_ema_length = 9
trend_ema_offset = 0
saved_previous_price = {}
stop_percent = {}
profit_percent = {}
stop_precentage = 0.03
profit_percentage = 0.08


###################################
#  Get support and resistance
###################################
def find_resistances(data, num_points=2, num_bars=100, support_resistence_diff=1.5):
    df = data.copy()
    # Start from older data
    # try:
    #     df['date'] = pd.to_datetime(df.index)
    # except Exception as e:
    #     print(e)
    # df.sort_values(by='date', ascending=True, inplace=True)

    # This will store potential resistances
    resistances = []

    for i in range(num_points, len(df) - num_bars):
        max_close = df.iloc[i - num_points:i]['high'].max()
        min_close = df.iloc[i:i + num_bars]['low'].min()

        if min_close > max_close * support_resistence_diff:
            resistances.append((df.index[i + num_bars], max_close))

        # Convert the list of resistances to a DataFrame
    resistances_df = pd.DataFrame(resistances, columns=['Date', 'Resistance'])
    return resistances_df


def find_supports(data, num_points=2, num_bars=100, support_resistence_diff=1.5):
    df = data.copy()
    # # Start from older data
    # df['date'] = pd.to_datetime(df.index)
    # df.sort_values(by='date', ascending=True, inplace=True)

    # This will store potential supports
    supports = []

    for i in range(num_points, len(df) - num_bars):
        min_close = df.iloc[i - num_points:i]['low'].min()
        max_close = df.iloc[i:i + num_bars]['high'].max()

        if max_close < min_close * support_resistence_diff:
            supports.append((df.index[i + num_bars], min_close))

        # Convert the list of supports to a DataFrame
    supports_df = pd.DataFrame(supports, columns=['Date', 'Support'])
    return supports_df


####################################
#  Super Trend Calculation
###################################
def calculate_atr(data, period=10):
    high = data['high'].tolist()
    low = data['low'].tolist()
    close = data['close'].tolist()
    high_low = [h - l for h, l in zip(high, low)]
    high_close = [abs(h - c) for h, c in zip(high, [0] + close[:-1])]
    low_close = [abs(l - c) for l, c in zip(low, [0] + close[:-1])]
    tr = [max(hl, hc, lc) for hl, hc, lc in zip(high_low, high_close, low_close)]
    atr = [sum(tr[i:i + period]) / period if i >= period else sum(tr[:i + 1]) / len(tr[:i + 1]) for i in range(len(tr))]
    return atr


def calculate_supertrend(data, period=7, multiplier=3):
    high = data['high'].tolist()
    low = data['low'].tolist()
    close = data['close'].tolist()
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    atr = calculate_atr(data, period)
    upper_band = [h + (multiplier * a) for h, a in zip(hl2, atr)]
    lower_band = [h - (multiplier * a) for h, a in zip(hl2, atr)]
    in_uptrend = [True] * len(high)
    for current in range(1, len(high)):
        previous = current - 1
        if close[current] > upper_band[previous]:
            in_uptrend[current] = True
        elif close[current] < lower_band[previous]:
            in_uptrend[current] = False
        else:
            in_uptrend[current] = in_uptrend[previous]
            if in_uptrend[current] and lower_band[current] < lower_band[previous]:
                lower_band[current] = lower_band[previous]
            if not in_uptrend[current] and upper_band[current] > upper_band[previous]:
                upper_band[current] = upper_band[previous]
    return in_uptrend


def generate_super_trend_signals(data, in_uptrend):
    signal = [''] * len(in_uptrend)
    super_trend_latest_signal = ''
    super_trend_signal = ''
    for i in range(1, len(in_uptrend)):
        if in_uptrend[i] and not in_uptrend[i - 1]:
            signal[i] = 'buy'
            super_trend_previous_signal = super_trend_signal
            super_trend_signal = signal[i]

        elif not in_uptrend[i] and in_uptrend[i - 1]:
            signal[i] = 'sell'
            super_trend_previous_signal = super_trend_signal
            super_trend_signal = signal[i]
        super_trend_latest_signal = signal[i]

        # print(f'--> {i} <-- Signal at {data.index[i]}: = "{signal[i]}"')

    return super_trend_latest_signal, super_trend_signal, super_trend_previous_signal


#####################################
#  PowerX Optimizer Calculation
#####################################

def calculate_MACD(df, short_span=12, long_span=26, signal_span=9):
    exp12 = df['close'].ewm(span=short_span, adjust=False).mean()
    exp26 = df['close'].ewm(span=long_span, adjust=False).mean()
    macdLine = exp12 - exp26
    signalLine = macdLine.ewm(span=signal_span, adjust=False).mean()
    histLine = macdLine - signalLine
    return histLine


def calculate_RSI(df, time_window=7):
    delta = df['close'].diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    average_gain = gain.rolling(window=time_window).mean()
    average_loss = abs(loss.rolling(window=time_window).mean())
    rs = average_gain / average_loss
    rsiLine = 100 - (100 / (1 + rs))
    return rsiLine


def calculate_StochasticOscillator(df, k_window=14, d_window=3):
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stochLine = k.rolling(window=d_window).mean()
    return stochLine


def assign_trend(df):
    trend_conditions = [
        df['signal'] == 'buy',
        df['signal'] == 'sell',
        df['signal'] == 'none'
    ]
    trend_choices = ['buy', 'sell', 'none']
    df['trend'] = np.select(trend_conditions, trend_choices)
    return df['trend']


def get_powerx(df, histLine, rsiLine, stochLine):
    conditions = [
        (histLine > 0) & (rsiLine > 50) & (stochLine > 50),
        (histLine <= 0) & (rsiLine <= 50) & (stochLine <= 50)
    ]
    choices = ['buy', 'sell']
    df['signal'] = np.select(conditions, choices, default='')
    return df


def generate_powerx_signal(data):
    histLine = calculate_MACD(data)
    rsiLine = calculate_RSI(data)
    stochLine = calculate_StochasticOscillator(data)
    df = get_powerx(data, histLine, rsiLine, stochLine)
    signal = assign_trend(df)
    signal = signal.iloc[-1]
    return signal


########################################
#  Helper functions
########################################

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


def get_assets_from_csv(filename='assets_to_trade.csv'):
    spdict = {}
    ppdict = {}
    file = open(filename, "r")
    data = list(csv.DictReader(file, delimiter=","))
    print(data)
    file.close()
    symbols = [str(row["symbol"]) for row in data]
    for row in data:
        spdict[str(row["symbol"])] = str(row["stop_percent"])
        ppdict[str(row["symbol"])] = str(row["profit_percent"])
    return symbols, spdict, ppdict


###############################################
#  Alpaca API functions
###############################################
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
    super_trend = []
    reverse_data = False
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

        raw_data = bars
        st_bars = bars
        # in_uptrend = calculate_supertrend(st_bars)
        latest_super_trend, super_trend, hist_super_trend = \
            generate_super_trend_signals(st_bars,
                                         calculate_supertrend(st_bars))
        powerx_signal = generate_powerx_signal(st_bars)

        if get_all_prices == False:
            bars = bars[['close']]
            bars.columns = ['price']
            return powerx_signal, latest_super_trend, super_trend, hist_super_trend, bars, reverse_data, raw_data
        else:
            bars = bars.drop(columns=['volume'])
            bars = bars.drop(columns=['vwap'])
            bars = bars.drop(columns=['trade_count'])
            return powerx_signal, latest_super_trend, super_trend, hist_super_trend, bars, reverse_data, raw_data

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
                conn += 1
            except:
                time.sleep(1)
                bars = client.get_crypto_bars(request_params).df
                break

        raw_data = bars
        st_bars = bars
        # in_uptrend = calculate_supertrend(st_bars)
        latest_super_trend, super_trend, hist_super_trend = \
            generate_super_trend_signals(st_bars,
                                         calculate_supertrend(st_bars))
        powerx_signal = generate_powerx_signal(st_bars)

        if not get_all_prices:
            bars = bars[['close']]
            bars.columns = ['price']
            return powerx_signal, latest_super_trend, super_trend, hist_super_trend, bars, reverse_data, raw_data
        else:
            bars = bars.drop(columns=['volume'])
            bars = bars.drop(columns=['vwap'])
            bars = bars.drop(columns=['trade_count'])
            return powerx_signal, latest_super_trend, super_trend, hist_super_trend, bars, reverse_data, raw_data

    else:
        print(f'ERROR: the Symbol {symbol} can not be found')
        return 0, 0, 0, 0, 0, False, 0


def get_the_trend(symbol, **kwargs) -> pd.DataFrame:
    # Get the arguments
    interval = kwargs.get('interval', '60min')
    time_series = kwargs.get('time_series', 'INTRADAY')
    get_all_prices = kwargs.get('get_all_prices', False)
    default_to_currency = kwargs.get('default_to_currency', 'USD')

    # Get historical data from Alpha Vantage API
    powerx_signal, latest_super_trend, super_trend, hist_super_trend, close, reverse, raw_data = \
        get_historical_price_data(symbol,
                                  interval=interval,
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

    cl = close.copy()
    cl['sig'] = buy_signal['close'] + sell_signal['close']
    # for i, row in close.iterrows():
    #     cl.loc[i, 'sig'] = buy_signal.loc[i, 'close'] + sell_signal.loc[i, 'close']

    # add new column 'signal'
    cl['signal'] = ''

    # loop through rows and populate 'signal' column
    buy_seen = False
    sell_seen = False
    sig_history_list = []
    for i, row in cl.iterrows():
        if (row['sig'] == 'Buy') & (buy_seen == False):
            cl.loc[i, 'signal'] = 'Buy'
            sig_history_list.append('Buy')
            buy_seen = True
            sell_seen = False
        elif (row['sig'] == 'Sell') & (sell_seen == False):
            cl.loc[i, 'signal'] = 'Sell'
            sig_history_list.append('Sell')
            sell_seen = True
            buy_seen = False
        elif row['sig'] == '':
            sell_seen = False
            buy_seen = False
        else:
            continue  # ignore cells with empty strings

    return raw_data, cl, sig_history_list, latest_super_trend, super_trend, hist_super_trend, powerx_signal


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
        print(f"SKIPPING {symbol} ... Not enough available cash to make a new trade at {round(stock_price, 4)}")
        if print_additional_log:
            print(
                f'max_trade_amount = ( account_balance * max_per_account ) - '
                f'( total_open_order_amount - total_open_position_amount )')
            print(f'{round(max_trade_amount, 4)} = ( {round(account_balance, 4)} * {max_per_account} ) - '
                  f'( {round(total_open_order_amount, 4)} - {total_open_position_amount} ) ')
            print(f"Account Balance =  {round(account_balance, 4)}")
            print(f"max_trade_amount=  {round(max_trade_amount, 4)}")
            print(f"Max available per Trade =  {round(max_amount_per_trade, 4)}")
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
            actual_shares_to_buy = round(price_paid_per_share / stock_price, 4)
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

    total_cost = round(actual_shares_to_buy * price_paid_per_share, 4)

    if print_additional_log:
        print(
            f"\r\n account balance = {round(account_balance, 4)} "
            f"\r\n max_trade_amount = account_balance * max_per_account - total_open_order_amount - "
            f"total_open_position_amount "
            f"\r\n {round(max_trade_amount, 4)} = {round(account_balance, 4)} * {max_per_account} - "
            f"{total_open_order_amount} - {total_open_position_amount} "
            f"\r\n max_trade_amount < max_amount_per_trade "
            f"\r\n {round(max_trade_amount, 4)} < {round(max_amount_per_trade, 4)} "
            f"\r\n actual_trade_amount < price_paid_per_share "
            f"\r\n {round(actual_trade_amount, 4)} < {round(price_paid_per_share, 4)} "
            f"\r\n Market TREND = {buy_sell} "
            f"\r\n Number of Shares to buy = {actual_shares_to_buy} at price = {round(price_paid_per_share, 4)}"
            f"\r\n Total Cost for this trade = {total_cost}"
            f"\r\n Stock Price = {round(stock_price, 4)} "
            f"\r\n Spend Limit  = {round(max_amount_per_trade, 4)} "
            f"\r\n stop_loss = {round(stop_loss_price, 4)} "
            f"\r\n take_profit = {round(take_profit_price, 4)}")
    return account_balance, actual_shares_to_buy, stock_price, price_paid_per_share, stop_loss_price, take_profit_price


def api_get_current_price(api_key, api_secret, api_base_url, symbol, signal):
    api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
    type = 'crypto'
    try:
        type = get_security_type(symbol)
    except:
        pass
    try:
        if type.lower() == 'crypto':
            symbol_list = [symbol]
            quote = api.get_latest_crypto_quotes(symbol_list)
        else:
            quote = api.get_latest_quote(symbol)
    except Exception as e:
        print(e)
        idx = 1
        while idx < 4:
            idx += 1
            time.sleep(1)
            try:
                if type.lower() == 'crypto':
                    symbol_list = [symbol]
                    quote = api.get_latest_crypto_quotes(symbol_list)
                else:
                    quote = api.get_latest_quote(symbol)
                break
            except:
                pass
        if idx >= 4:
            if symbol in saved_previous_price:
                print(
                    f'[{symbol}] an error occurred while fetching the price. '
                    f'Returned previous price ${saved_previous_price[symbol]}')
                return saved_previous_price[symbol]

    if str(signal).lower() == 'sell':
        if type.lower() == 'crypto':
            bid_price = quote[symbol].bp
        else:
            bid_price = quote.bp

        saved_previous_price[symbol] = round(bid_price, 4)
        return round(bid_price, 4)
    else:
        if type.lower() == 'crypto':
            ask_price = quote[symbol].ap
        else:
            ask_price = quote.ap

        saved_previous_price[symbol] = round(ask_price, 4)
        return round(ask_price, 4)


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

    def submit_order(self, side, type, price, quantity=1, order_id=''):
        return self.api.submit_order(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            type=type,
            client_order_id=order_id,
            time_in_force='gtc',
            limit_price=round(price, 4)
        )

    def submit_limit_order(self, side, type, price, stop_price, quantity=1):
        return self.api.submit_order(
            symbol=self.symbol,
            qty=quantity,
            side=side,
            type=type,
            time_in_force='gtc',
            stop_price=stop_price,
            limit_price=round(price, 4)
        )

    def monitor_order(self, order, lock):
        def close_position():
            try:
                position = self.api.get_position(self.symbol.replace('/', ''))
                self.api.close_position(position.symbol, qty=position.qty)
            except:
                pass

        with lock:
            while str(order.status).lower() not in ['filled', 'new']:
                order = self.api.get_order(order.id)
                time.sleep(1)
            print(f'{self.symbol} ... ORDER Status = {order.status} ...')

        # Monitor the status of the stop-loss and take-profit orders
        ct = 0
        with lock:
            while True:
                ct += 1

                try:
                    raw_data, df1, df1_hist, latest_super_trend, super_trend, hist_super_trend, powerx_signal = \
                        get_the_trend(self.symbol, time_series='INTRADAY', interval='30min')
                except:
                    time.sleep(1)
                    try:
                        raw_data, df1, df1_hist, latest_super_trend, super_trend, hist_super_trend, powerx_signal = \
                            get_the_trend(self.symbol, time_series='INTRADAY', interval='30min')
                    except:
                        time.sleep(5)
                        continue

                signal = df1.iloc[-1]['sig']
                prev_signal = df1_hist[-2]

                try:
                    cprice = round(float(api_get_current_price(api_key, api_secret,
                                                               api_base_url, self.symbol, self.side)), 4)
                except:
                    try:
                        cprice = round(float(api_get_current_price(api_key, api_secret,
                                                                   api_base_url, self.symbol, self.side)), 4)
                    except:
                        time.sleep(25 * 60)
                        continue

                order_status_not_new = str(order.status).lower() != 'new'
                if str(self.side).lower() == 'buy':
                    order_met = (cprice >= self.take_profit_price) or (cprice <= self.stop_loss_price)
                elif str(self.side).lower() == 'sell':
                    order_met = (cprice <= self.take_profit_price) or (cprice >= self.stop_loss_price)

                if (order_status_not_new and order_met) or (str(super_trend).lower() != str(self.side).lower()):
                    close_position()
                    if str(self.side).lower() == 'buy':
                        result = 'PROFIT' if cprice >= self.take_profit_price else 'LOSS'
                    elif str(self.side).lower() == 'sell':
                        result = 'PROFIT' if cprice <= self.take_profit_price else 'LOSS'
                    print(f'{self.symbol} position closes with a ---> {result} <--- @ ---> ${cprice} <---')

                print(f'{ct}:  => Current {self.symbol} price ==> {cprice} ==> status = {order.status} '
                      f'Waiting for -->  stop_loss_order @ {self.stop_loss_price} '
                      f'" OR  '
                      f'take_profit_order @ {self.take_profit_price} "')

                time.sleep(25 * 60)

    def execute(self):
        current_buy_price = api_get_current_price(api_key, api_secret, api_base_url, self.symbol, self.side)
        # Send initial buy order
        self.buy_order = self.submit_order(self.side, 'limit', current_buy_price, self.quantity, self.id)
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
    while counter < 2:
        counter += 1
        api.close_all_positions()
        time.sleep(2)
        api.cancel_all_orders()
        time.sleep(1)
    print('COMPLETED CLOSING ALL OPEN POSITIONS AND ORDERS')


def trade(api, assets, threads, lock):
    # create shared lock
    # with lock:
    #     while True:

    with lock:
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
                raw_data, df1, df1_hist, latest_super_trend, super_trend, hist_super_trend, powerx_trend = \
                    get_the_trend(symbol, time_series='INTRADAY', interval='30min')

                # get the current signal
                signal = df1.iloc[-1]['sig']
                prev_signal = df1_hist[-2]

                # if signal == '':
                #     signal = prev_signal

                print(
                    f'{symbol} Current trend = {signal} and '
                    f'SUPER_TREND = {super_trend} and '
                    f'PowerX_trend = {powerx_trend}')

                # signal = latest_super_trend

                # if no current signal is given use the previous
                # if signal == '':
                #     signal = super_trend
                # elif super_trend != '':
                #     signal = super_trend
                # elif signal != '' and super_trend == '':
                #     signal = signal
                # elif signal == '' and super_trend != '':
                #     signal = super_trend

                asset = api.get_asset(symbol)
                shortable = getattr(asset, 'shortable')
                easy_to_borrow = getattr(asset, 'easy_to_borrow')

                if (not shortable or not easy_to_borrow) and (str(signal).lower() == 'sell' and
                                                              str(super_trend).lower() == 'sell' and str(
                            powerx_trend).lower() == 'sell'):
                    print(f'This ASSET -> {symbol} can not be shorted .. SKIPPING purchase')
                    continue

                if (str(signal).lower() == 'buy' and
                    str(super_trend).lower() == 'buy' and
                    str(powerx_trend).lower() == 'buy') or \
                        (str(signal).lower() == 'sell' and
                         str(super_trend).lower() == 'sell' and
                         str(powerx_trend).lower() == 'sell'):
                    # skip if asset in your portfolio.
                    if is_asset_in_pofolio(api, symbol):
                        print(f'ASSET -> {symbol} already exist in your portfolio .. SKIPPING purchase')
                        continue

                    # Get the current asset price
                    symbol_price = api_get_current_price(api_key, api_secret, api_base_url, symbol, signal)
                    resistances = find_resistances(raw_data)
                    supports = find_supports(raw_data)

                    account_balance, \
                        actual_shares_to_buy, \
                        stock_price, \
                        price_paid_per_share, \
                        stop_loss_price, \
                        take_profit_price = \
                        money_management(symbol, symbol_price, 1, signal, 0.2, 0.02, stop_precentage, profit_percentage)

                    # print(f'-----------------------------------------------------------------------')
                    # print(f'{symbol}')
                    # print(f'Signal = {signal}  Prev+Signal = {prev_signal}')
                    # print(f'symbol_price ==> {round(symbol_price, 4)}')
                    # print(f'stop_price ==> {round(stop_loss_price)}')
                    # print(f'teke_profit_limit_price ==> {round(take_profit_price, 4)}')
                    # print(f'DIFF to LOSS ==> {round(symbol_price - stop_loss_price, 4)}')
                    # print(f'DIFF to GAIN ==> {round(take_profit_price - symbol_price, 4)}')
                    # print(f'\r')

                    if str(signal).lower() == 'buy':
                        message = f'Bought {actual_shares_to_buy} <{symbol}> @ ${round(symbol_price, 4)} ' \
                                  f'stop price = ${round(stop_loss_price, 4)}  ' \
                                  f'take profit = ${round(take_profit_price, 4)} @ {utc_time_str}'
                        print(message)
                    elif str(signal).lower() == 'sell':
                        message = f'Sold  {actual_shares_to_buy} <{symbol}> @ ${round(symbol_price, 4)} ' \
                                  f'stop price = ${round(stop_loss_price, 4)}  ' \
                                  f'take profit = ${round(take_profit_price, 4)} @ {utc_time_str}'
                        print(message)
                    else:
                        message = f'Nothing Purchased <{symbol}> @ ${round(symbol_price, 4)} ' \
                                  f'stop price = $0.00  take profit = $0.00 @ {utc_time_str}'
                        print(message)

                    print(
                        f"\r\n<0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><"
                        f"0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0>")

                    # Create BracketOrder object and execute
                    order = BracketOrder(api, symbol, str(signal).lower(), actual_shares_to_buy, symbol_price,
                                         round(stop_loss_price, 4),
                                         round(take_profit_price, 4),
                                         client_id)
                    t1 = order.execute()
                    threads.append(t1)
            except Exception as e:
                print(e)
                pass


if __name__ == '__main__':
    # initialize the api
    api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
    # initialize the lists
    assets, stop_percent, profit_percent = get_assets_from_csv('assets_to_trade.csv')
    # the list of all the threads
    threads = []

    # Close all active trades before starting the program
    close_all_active_trades(api)

    while True:
        lock = Lock()
        trade(api, assets, threads, lock)
        for thread in threads:  # iterates over the threads
            try:
                thread.start()  # waits until the thread has finished work
            except:
                pass

        trade_thread = threading.Thread(target=trade, args=(api, assets, threads, lock,))
        trade_thread.start()
        threads.append(trade_thread)

        time.sleep(60 * 25)
