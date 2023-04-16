import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler


def bigb_model_data_preparation(bigb_trading_df):
    
    # create a new dataframe to store the feartures and signals
    signals_df = bigb_trading_df.dropna()

    # Set the short window and long window
    short_window = 30
    long_window = 60

    # Generate the fast and slow simple moving averages (5 and 50 days, respectively)
    signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
    signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()

    signals_df = signals_df.dropna()
    
    # Initialize the new Signal column
    signals_df['Signal'] = 0.0
    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    signals_df.loc[(signals_df['actual_returns'] >= 0), 'Signal'] = 1
    # When Actual Returns are less than 0, generate signal to sell stock short
    signals_df.loc[(signals_df['actual_returns'] < 0), 'Signal'] = -1
    
    # Calculate the strategy returns and add them to the signals_df DataFrame
    signals_df['strategy_returns'] = signals_df['actual_returns'] * signals_df['Signal'].shift()
    
    # Generate the trainning dataset and the testing dataset
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna().copy()
    # Create the target set selecting the Signal column and assiging it to y
    y = signals_df['Signal']
    
    # Select the start and the end date of the training period
    # Set the training_begin
    training_begin = datetime.now() - timedelta(days=365*5)
    # Select the ending period for the training data with an offset of  years 
    training_end = training_begin + DateOffset(years=4)
    
    # Generate the X_train and y_train DataFrames
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]

    # Scale the features DataFrames
    scaler = StandardScaler()
    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)
    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, signals_df