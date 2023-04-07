import yahoo_fin.stock_info as si
import pandas as pd
from datetime import datetime
import numpy as np

def bigb_data_ml(bigb_data, bigb_returns, bigb_annual_return, bigb_annual_volatility):
    # define the stock symbols for the index
    symbols = ["BAC", "C", "GS", "JPM", "MS", "WFC"]

    # create empty dictionary to store the dataframes for each stock symbol
    dfs = {}

    # iterate over each stock symbol and get the dividend-adjusted data for that symbol
    for symbol in symbols:
        # get the stock data for the current symbol
        df = si.get_data(symbol, start_date="2013-03-18", end_date="2023-04-07")

        # get the dividend data for the current symbol
        dividend_data = si.get_dividends(symbol)

        # adjust the stock data for dividends
        for index, row in dividend_data.iterrows():
            ex_date = pd.to_datetime(index).date()
            ex_datetime = datetime.combine(ex_date, datetime.min.time())
            div_amount = row["dividend"]
            df.loc[df.index >= ex_datetime, "open"] -= div_amount
            df.loc[df.index >= ex_datetime, "high"] -= div_amount
            df.loc[df.index >= ex_datetime, "low"] -= div_amount
            df.loc[df.index >= ex_datetime, "close"] -= div_amount

        # add the adjusted dataframe to the dictionary
        dfs[symbol] = df

    # Calculate the hypothetical OHLC data for the BIGB using the adjusted stock data
    bigb_data = pd.DataFrame()
    bigb_data['open'] = (dfs['BAC']['open'] + dfs['C']['open'] + dfs['GS']['open'] +
                         dfs['JPM']['open'] + dfs['MS']['open'] + dfs['WFC']['open']) / 6
    bigb_data['high'] = (dfs['BAC']['high'] + dfs['C']['high'] + dfs['GS']['high'] +
                         dfs['JPM']['high'] + dfs['MS']['high'] + dfs['WFC']['high']) / 6
    bigb_data['low'] = (dfs['BAC']['low'] + dfs['C']['low'] + dfs['GS']['low'] +
                        dfs['JPM']['low'] + dfs['MS']['low'] + dfs['WFC']['low']) / 6
    bigb_data['close'] = (dfs['BAC']['close'] + dfs['C']['close'] + dfs['GS']['close'] +
                          dfs['JPM']['close'] + dfs['MS']['close'] + dfs['WFC']['close']) / 6
    bigb_data['adj close'] = (bigb_data['open'] + bigb_data['high'] + bigb_data['low'] + bigb_data['close']) / 4

    # Drop any rows with missing data
    bigb_data = bigb_data.dropna()

    # calculate the past 3 years' annual return and annual volatility for BIGB:

    # Get the daily returns for the past 3 years
    bigb_returns = bigb_data["adj close"].pct_change().tail(756)

    # Calculate the annualized return
    bigb_annual_return = np.mean(bigb_returns) * 252

    # Calculate the annualized volatility
    bigb_annual_volatility = np.std(bigb_returns) * np.sqrt(252)

    return bigb_data, bigb_returns, bigb_annual_return, bigb_annual_volatility
