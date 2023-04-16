import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler

# From yahoo_fin get the data of the BIGB
def get_bigb_data():
    
    # define the stock symbols for the index
    symbols = ["BAC", "C", "GS", "JPM", "MS", "WFC"]

    # create empty dictionary to store the dataframes for each stock symbol
    df = {}

    # set end_date as today and start_date as 20 years ago
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")

    # iterate over each stock symbol and get the dividend-adjusted data for that symbol
    for symbol in symbols:
        try:
            # get the stock data for the current symbol
            df[symbol] = si.get_data(symbol, start_date=start_date, end_date=end_date)
        except Exception as e:
            # log the error and continue to the next symbol
            logging.error(f"Error retrieving data for symbol {symbol}: {e}")
            continue

    if not df:
        # handle the case where there's no data available
        logging.warning("No data available for the specified time period")
        return None, None, None, None

    # Calculate the hypothetical OHLC data for the BIGB using the adjusted stock data
    bigb_data = pd.concat(df.values()).groupby(level=0).mean(numeric_only=True)
    bigb_data.index.name = "date"

    # Drop any rows with missing data
    bigb_data = bigb_data.dropna()
    # Convert bigb_data to a dataframe
    bigb_df = pd.DataFrame(bigb_data)

    # save bigb_data as csv
    bigb_df.to_csv("bigb_df.csv")
    
    # Get the bigb datarame for trading algorithm analysis
    
    # calculate the daily returns for bigb
    bigb_daily_returns = bigb_df["close"].pct_change().dropna()
    # add a `actual_returns` column to the bigb_df
    bigb_df['actual_returns'] = bigb_daily_returns
    bigb_df = bigb_df.dropna()
    # careate a dataframe contains only 'close' and 'actual_returns' which can be used for trading_algorithm analysis
    bigb_trading_df = bigb_df[['close', 'actual_returns']]
    bigb_trading_df = pd.DataFrame(bigb_trading_df)
    bigb_trading_df = bigb_trading_df.dropna()
    # Get the bigb dataframe for clustering
    
    # Calculate the annualized return
    bigb_annual_return = np.mean(bigb_daily_returns) * 252
    # Calculate the annualized volatility
    bigb_annual_volatility = np.std(bigb_daily_returns) * np.sqrt(252)
    # set the start date for the past 3 years
    start_date_3years_ago = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    # slice the past 3 years of data from bigb_df
    bigb_cluster_df= bigb_df.loc[start_date_3years_ago:end_date]
    # create a new dataframe which can be used for trading_algrithm
    bigb_cluster_df = pd.DataFrame({"bigb_annual_return": [bigb_annual_return],
                                    "bigb_annual_volatility": [bigb_annual_volatility]})
    bigb_cluster_df = bigb_cluster_df.dropna()
    
    return bigb_df, bigb_trading_df, bigb_cluster_df

def get_benchmark_cluster_data():
    # Create a list of 20 mainstream benchmark symbols
    benchmark_symbols = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", 
                         "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "FXI", "INDA", "EWJ"]

    # Create an empty DataFrame to store the benchmark data
    benchmark_cluster_df = pd.DataFrame(index=benchmark_symbols, columns=["annual_return", "annual_volatility"])

    # Iterate over each benchmark symbol and get the data
    for symbol in benchmark_symbols:
        # Get the historical data, since there are no enough data for etf, we set the examing period = 3 years
        df = si.get_data(symbol, start_date=pd.Timestamp.now()-pd.DateOffset(years=3), end_date=pd.Timestamp.now())

        # Calculate the daily returns
        benchmark_daily_returns = df["close"].pct_change().dropna()

        # Calculate the annualized return and volatility
        benchmark_annual_return = np.mean(benchmark_daily_returns) * 252
        benchmark_annual_volatility = np.std(benchmark_daily_returns) * np.sqrt(252)

        # Add the data to the DataFrame
        benchmark_cluster_df.loc[symbol, "annual_return"] = benchmark_annual_return
        benchmark_cluster_df.loc[symbol, "annual_volatility"] = benchmark_annual_volatility
        benchmark_cluster_df = benchmark_cluster_df.dropna()
    
    return benchmark_cluster_df

def get_bank_etf_cluster_data():
    # Create a list of 20 mainstream bank ETF symbols
    bank_etf_symbols = ["XLF", "KBE", "KRE", "VFH", "IYF", "FNCL", "RWW", "IXG", "KBWB", "PSP", 
                        "AGG", "RYF", "FAS", "UYG", "IAI", "QABA", "KBE", "XKFF", "KBWR", "KBWP"]

    # Create an empty DataFrame to store the bank ETF data
    bank_etf_cluster_df = pd.DataFrame(index=bank_etf_symbols, columns=["annual_return", "annual_volatility"])

    # Iterate over each bank ETF symbol and get the data
    for symbol in bank_etf_symbols:
        # Get the historical data, since there are no enough data for etf, we set the examing period = 3 years
        df = si.get_data(symbol, start_date=pd.Timestamp.now()-pd.DateOffset(years=3), end_date=pd.Timestamp.now())

        # Calculate the daily returns
        bank_etf_daily_returns = df["close"].pct_change().dropna()

        # Calculate the annualized return and volatility
        bank_etf_annual_return = np.mean(bank_etf_daily_returns) * 252
        bank_etf_annual_volatility = np.std(bank_etf_daily_returns) * np.sqrt(252)

        # Add the data to the DataFrame
        bank_etf_cluster_df.loc[symbol, "annual_return"] = bank_etf_annual_return
        bank_etf_cluster_df.loc[symbol, "annual_volatility"] = bank_etf_annual_volatility
        bank_etf_cluster_df = bank_etf_cluster_df.dropna()

    return bank_etf_cluster_df

 