import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt

    
def visualize_short_position_strategy(bigb_trading_df):
     
    # Prepare Signals dataframe for visualize the short position strategy using Signal_Short
    
    # create a new DataFrame for short postion strategy signal
    sp_signals_df = bigb_trading_df.copy()
    # Set the short window and long window
    sp_short_window = 20
    sp_long_window = 50
    # Generate the short and long moving averages 
    sp_signals_df['SMA20'] = sp_signals_df['close'].rolling(window=sp_short_window).mean()
    sp_signals_df['SMA50'] = sp_signals_df['close'].rolling(window=sp_long_window).mean()
    sp_signals_df.dropna(inplace=True)
    # Create an empty Signal column for short position
    sp_signals_df['Signal_Short'] = 0.0
    # Generate the trading signal 0 or 1, for short position strategy
    sp_signals_df['Signal_Short'] = np.where(
        sp_signals_df['SMA20'] < sp_signals_df['SMA50'], 1.0, 0.0
    )
    
    # Calculate the points in time at which a short position should be taken, 1 or -1
    sp_signals_df['Entry/Exit'] = sp_signals_df['Signal_Short'].diff()

    sp_entry = sp_signals_df[sp_signals_df['Entry/Exit'] == 1.0]['close'].hvplot.scatter(
        color='red',
        marker='^',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the exit positions relative to close price
    sp_exit = sp_signals_df[sp_signals_df['Entry/Exit'] == -1.0]['close'].hvplot.scatter(
        color='green',
        marker='v',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the close price for the investment
    sp_security_close = sp_signals_df[['close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize the moving averages
    sp_moving_avgs = sp_signals_df[['SMA20', 'SMA50']].hvplot(
        ylabel='Price in $',
        width=1000,
        height=400
    )
    
    # Overlay the plots
    sp_entry_exit_plot = sp_security_close * sp_moving_avgs * sp_entry * sp_exit
    sp_entry_exit_plot.opts(
        title='BIGB - Short-Position Dual Moving Average Trading Algorithm'
    )
    
    plt.savefig("images/BIGB - Short-Position Dual Moving Average Trading Algorithm.png")
    return sp_entry_exit_plot


def visualize_long_position_strategy(bigb_trading_df):
     
    # Prepare Signals dataframe for visualize the long position strategy using Signal_Long
    
    # create a new DataFrame for long postion strategy signal
    lp_signals_df = bigb_trading_df.copy()
    # Set the short window and long window
    lp_short_window = 20
    lp_long_window = 50
    # Generate the short and long moving averages 
    lp_signals_df['SMA20'] = lp_signals_df['close'].rolling(window=lp_short_window).mean()
    lp_signals_df['SMA50'] = lp_signals_df['close'].rolling(window=lp_long_window).mean()
    lp_signals_df.dropna(inplace=True)
    # Create an empty Signal column for long position
    lp_signals_df['Signal_Long'] = 0.0
    # Generate the trading signal 0 or 1, for long position strategy
    lp_signals_df['Signal_Long'] = np.where(
        lp_signals_df['SMA20'] > lp_signals_df['SMA50'], 1.0, 0.0
    )
    # Calculate the points in time at which a long position should be taken, 1 or -1
    lp_signals_df['Entry/Exit'] = lp_signals_df['Signal_Long'].diff()

    # Visualize the entry positions relative to close price
    lp_entry = lp_signals_df[lp_signals_df['Entry/Exit'] == 1.0]['close'].hvplot.scatter(
        color='yellow',
        marker='^',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the exit positions relative to close price
    lp_exit = lp_signals_df[lp_signals_df['Entry/Exit'] == -1.0]['close'].hvplot.scatter(
        color='purple',
        marker='v',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the close price for the investment
    lp_security_close = lp_signals_df[['close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize the long moving averages
    lp_moving_avgs = lp_signals_df[['SMA20', 'SMA50']].hvplot(
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Overlay the plots
    lp_entry_exit_plot = lp_security_close * lp_moving_avgs * lp_entry * lp_exit
    lp_entry_exit_plot.opts(
        title='BIGB - Long-Position Dual Moving Average Trading Algorithm'
    )
    plt.savefig("images/BIGB - Long-Position Dual Moving Average Trading Algorithm.png")
    
    return lp_entry_exit_plot