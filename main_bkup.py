import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.enums import AssetClass
from datetime import datetime, timezone, timedelta

import requests
import json
import time
import threading
import pandas as pd
import time
import requests


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


from alpaca.trading.client import TradingClient

# Alpaca-py API Key
api_key = 'PKYSM0SOBK48QBC0KSQB'
api_secret = '3h8YhPEEsTiAl4X4PgppYAtsPsTdKutXFR0WqcRm'
api_base_url = 'https://paper-api.alpaca.markets'
symbol = 'BTC/USD'

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


def get_security_type(ticker_symbol):
    asset = alpaca_trading_client.get_asset(ticker_symbol)
    if asset.asset_class == AssetClass.CRYPTO:
        return 'Crypto'
    elif asset.asset_class == AssetClass.US_EQUITY:
        return 'Stock'
    else:
        return 'None'

    # Get the first part before the '/' of symbols that look like this BTC/USD  or ETH/USD etc.
    parts = ticker_symbol.split("/")
    ticker_symbol = parts[0]
    # Check to see if this is a stock or Crypto
    try:
        # Check if Crypto
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={ticker_symbol}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        check = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return 'Crypto'
    except:
        # Check if Stock
        try:
            url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
            response = requests.get(url)
            data = response.json()
            check = data['Global Quote']['05. price']
            return 'Stock'
        except:
            # The ticker is Neither Crypto / Stock
            return 'None'


# Define a function to get the current price of the symbol
def get_current_price(symbol, default_to_currency="USD"):
    # Get the first part before the '/' of symbols that look like this BTC/USD  or ETH/USD etc.
    parts = symbol.split("/")
    ticker_symbol = parts[0]
    # Check the type of Symbol
    symbol_type = get_security_type(symbol)
    if symbol_type == 'Crypto':
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={ticker_symbol}&to_currency={default_to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        crypto_price = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return float(crypto_price)
    elif symbol_type == 'Stock':
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        stock_price = data['Global Quote']['05. price']
        return float(stock_price)
    else:
        return float(0.00)


# Define a function to get the historical prices of the symbol
def get_historical_prices(symbol, interval='60min', time_series='INTRADAY', default_to_currency="USD"):
    # Get the first part before the '/' of symbols that look like this BTC/USD  or ETH/USD etc.
    parts = symbol.split("/")
    ticker_symbol = parts[0]
    # is this a stock or Crypto
    sec_type = get_security_type(symbol)
    if sec_type == 'Stock':
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{time_series}&symbol={ticker_symbol}&interval={interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
    elif sec_type == 'Crypto':
        url = f'https://www.alphavantage.co/query?function=CRYPTO_{time_series}&symbol={ticker_symbol}&market={default_to_currency}&interval={interval}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'
    else:
        return f'ERROR: the Symbol {symbol} can not be found'

    response = requests.get(url)
    data = response.json()

    parse_str = 'Daily'
    if time_series == 'INTRADAY':
        parse_str = interval

    if sec_type == 'Stock':
        df = pd.DataFrame.from_dict(data[f'Time Series ({parse_str})'], orient='index')
    if sec_type == 'Crypto':
        df = pd.DataFrame.from_dict(data[f'Time Series Crypto ({parse_str})'], orient='index')
    df = df.astype(float)
    df = df[['4. close']]
    df.columns = ['price']
    return df


# Define a function to calculate the SMA for a given number of periods
def calculate_sma(df, periods):
    return df['price'].rolling(window=periods).mean()


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


if __name__ == "__main__":
    data2 = get_historical_prices('TSLA')
    data1 = get_historical_prices('BTC/USD')

    sec_type1 = get_security_type('BTC/USD')
    sec_type2 = get_security_type('TSLA')

    price1 = get_current_price('BTC/USD')
    price2 = get_current_price('TSLA')

    main()
    # instantiate TradingBot objects for AAPL, TSLA, and BTC/USD
    # aapl_bot = TradingBot(api_key, api_secret, api_base_url, 'AAPL', '1day', 10, 3, 0.02, 0.05, 0.05)
    # tsla_bot = TradingBot(api_key, api_secret, api_base_url, 'TSLA', '1day', 10, 3, 0.02, 0.05, 0.05)
    # btc_bot = TradingBot(api_key, api_secret, api_base_url, 'BTC/USD', '1day', 10, 3, 0.02, 0.05, 0.05)

    # run bots
    # aapl_bot.run()
    # tsla_bot.run()
    # btc_bot.run()
