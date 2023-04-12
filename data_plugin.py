import yahoo_fin.stock_info as si
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging

def get_bigb_data():
    """
    Retrieves stock data for the BIGB index, calculates its daily returns, 
    and returns the annualized return and volatility.
    
    Returns:
        A tuple containing:
            - bigb_df: a pandas DataFrame containing the daily OHLC data for the BIGB index
            - bigb_daily_returns: a pandas Series containing the daily returns for the past 3 years
            - bigb_annual_return: the annualized return for the past 3 years
            - bigb_annual_volatility: the annualized volatility for the past 3 years
    """
    
    # define the stock symbols for the index
    symbols = ["BAC", "C", "GS", "JPM", "MS", "WFC"]

    # create empty dictionary to store the dataframes for each stock symbol
    df = {}

    # set end_date as today and start_date as 10 years ago
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")

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

    # Get the daily returns for the past 3 years
    bigb_daily_returns = bigb_df["close"].pct_change().tail(756)

    # Calculate the annualized return
    bigb_annual_return = np.mean(bigb_daily_returns) * 252

    # Calculate the annualized volatility
    bigb_annual_volatility = np.std(bigb_daily_returns) * np.sqrt(252)

    return bigb_df, bigb_daily_returns, bigb_annual_return, bigb_annual_volatility

def get_benchmark_cluster_data():
    # Create a list of 20 mainstream benchmark symbols
    benchmark_symbols = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", 
                         "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "FXI", "INDA", "EWJ"]

    # Create an empty DataFrame to store the benchmark data
    benchmark_df = pd.DataFrame(index=benchmark_symbols, columns=["annual_return", "annual_volatility"])

    # Iterate over each benchmark symbol and get the data
    for symbol in benchmark_symbols:
        # Get the historical data
        df = si.get_data(symbol, start_date=pd.Timestamp.now()-pd.DateOffset(years=3), end_date=pd.Timestamp.now())

        # Calculate the daily returns
        benchmark_daily_returns = df["close"].pct_change().dropna()

        # Calculate the annualized return and volatility
        benchmark_annual_return = np.mean(benchmark_daily_returns) * 252
        benchmark_annual_volatility = np.std(benchmark_daily_returns) * np.sqrt(252)

        # Add the data to the DataFrame
        benchmark_df.loc[symbol, "annual_return"] = benchmark_annual_return
        benchmark_df.loc[symbol, "annual_volatility"] = benchmark_annual_volatility

    return benchmark_df

def get_bank_etf_cluster_data():
    # Create a list of 20 mainstream bank ETF symbols
    bank_etf_symbols = ["XLF", "KBE", "KRE", "VFH", "IYF", "FNCL", "RWW", "IXG", "KBWB", "PSP", 
                        "AGG", "RYF", "FAS", "UYG", "IAI", "QABA", "KBE", "XKFF", "KBWR", "KBWP"]

    # Create an empty DataFrame to store the bank ETF data
    bank_etf_df = pd.DataFrame(index=bank_etf_symbols, columns=["annual_return", "annual_volatility"])

    # Iterate over each bank ETF symbol and get the data
    for symbol in bank_etf_symbols:
        # Get the historical data
        df = si.get_data(symbol, start_date=pd.Timestamp.now()-pd.DateOffset(years=3), end_date=pd.Timestamp.now())

        # Calculate the daily returns
        bank_etf_daily_returns = df["close"].pct_change().dropna()

        # Calculate the annualized return and volatility
        bank_etf_annual_return = np.mean(bank_etf_daily_returns) * 252
        bank_etf_annual_volatility = np.std(bank_etf_daily_returns) * np.sqrt(252)

        # Add the data to the DataFrame
        bank_etf_df.loc[symbol, "annual_return"] = bank_etf_annual_return
        bank_etf_df.loc[symbol, "annual_volatility"] = bank_etf_annual_volatility

    return bank_etf_df