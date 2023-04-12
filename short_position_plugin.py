# Imports the required libraries
import pandas as pd
import hvplot.pandas
from bokeh.io import show
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Import other modules
import data_plugin
import questionnaire_plugin

# Prepare the data
def short_position_data_preparation():
    
    # create a new DataFrame with BIGB's data of past 3 years
    # Call the get_bigb_data() function to get the bigb_df, bigb_daily_returns, bigb_annual_return, bigb_annual_volatility
    bigb_df, bigb_daily_returns, bigb_annual_return, bigb_annual_volatility = data_plugin.get_bigb_data()
    # Add a daily return values column 'actual_return' to the DataFrame
    bigb_df['actual_returns'] = bigb_daily_returns
    # Drop all NaN values from the DataFrame
    bigb_df.dropna(inplace=True)
    # Calculate the date 3 year ago from today's date to get the past 3 years' data of BIGB
    start_date = datetime.now() - timedelta(days=1095)
    # Slice the past 3 year's DataFrame
    bigb_df_3y = bigb_df.loc[start_date:]
    
    
    # Generate a Dual Moving Average Crossover Trading Signal
    # Filter the date index and close, actual_returns columns
    sp_signals_df = bigb_df_3y.loc[:,['close','actual_returns']]
    
    # Set the short window and long window
    sp_short_window = 20
    sp_long_window = 50
    # Generate the short and long moving averages 
    sp_signals_df['SMA20'] = sp_signals_df['close'].rolling(window=sp_short_window).mean()
    sp_signals_df['SMA50'] = sp_signals_df['close'].rolling(window=sp_long_window).mean()
    sp_signals_df.dropna(inplace=True)
    # Create an empty Signal column
    sp_signals_df['Signal'] = 0.0
    # Generate the trading signal 0 or 1, for short position strategy
    sp_signals_df['Signal'] = np.where(
        sp_signals_df['SMA20'] < sp_signals_df['SMA50'], 1.0, 0.0
    )

    # Visualize the short position strategy using Singal
    # Calculate the points in time at which a position should be taken, 1 or -1
    sp_signals_df['Entry/Exit'] = sp_signals_df['Signal'].diff()
    # Visualize the entry positions relative to close price
    entry = sp_signals_df[sp_signals_df['Entry/Exit'] == 1.0]['close'].hvplot.scatter(
        color='red',
        marker='^',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the exit positions relative to close price
    exit = sp_signals_df[sp_signals_df['Entry/Exit'] == -1.0]['close'].hvplot.scatter(
        color='green',
        marker='v',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the close price for the investment
    security_close = sp_signals_df[['close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize the moving averages
    moving_avgs = sp_signals_df[['SMA20', 'SMA50']].hvplot(
        ylabel='Price in $',
        width=1000,
        height=400
    )
    
    # Overlay the plots
    sp_entry_exit_plot = security_close * moving_avgs * entry * exit
    sp_entry_exit_plot.opts(
        title='BIGB - Short-Position Dual Moving Average Trading Algorithm'
    )


    # Generate the trainning dataset and the testing dataset
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = sp_signals_df[['SMA20', 'SMA50']]
    # Create the target set selecting the Signal column and assiging it to y
    y = sp_signals_df['Signal']
    # Select the start and the end date of the training period
    training_begin = X.index.min()
    # Select the ending period for the training data with an offset of 2 years
    training_end = training_begin + DateOffset(years=2)
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

    return sp_entry_exit_plot, X_train_scaled, X_test_scaled, y_train, y_test, sp_signals_df

#Apply the original parameters to the performance of a first machine learning model-SCV, then backtest and evaluate the model. 
def short_position_svc_model(X_train_scaled, X_test_scaled, y_train, y_test, sp_signals_df):
    
    # From SVM, instantiate SVC classifier model instance
    sp_svc_model = svm.SVC()
    # Fit the model to the data using the training data
    sp_svc_model = sp_svc_model.fit(X_train_scaled, y_train)
    # Use the testing data to make the model predictions
    sp_svc_y_pred = sp_svc_model.predict(X_test_scaled)

    
    # Use a classification report to evaluate the SVC model using the predictions and testing data
    sp_svc_testing_report = classification_report(y_test, sp_svc_y_pred)


    # Create a new empty predictions DataFrame.
    sp_predictions_df_svc = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    sp_predictions_df_svc['predicted_signal'] = sp_svc_y_pred
    # Add the actual returns to the DataFrame
    sp_predictions_df_svc['actual_returns'] = sp_signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    sp_predictions_df_svc['strategy_returns'] = sp_predictions_df_svc['actual_returns'] * sp_predictions_df_svc['predicted_signal']

    
    # Plot the actual returns versus the strategy returns
    plot_actual_vs_strategy_returns_svc = (1 + sp_predictions_df_svc[["actual_returns", "strategy_returns"]]).cumprod().plot(
        title="Actual Returns vs Strategy Returns - SVM",
        xlabel="Date",
        ylabel="Cumulative Returns"
    )
    plot_actual_vs_strategy_returns_svc.get_figure().savefig("actual_vs_strategy_returns_svc.png")
    return sp_svc_y_pred, sp_predictions_df_svc, sp_svc_testing_report, plot_actual_vs_strategy_returns_svc




#Apply the original parameters to the performance of a second machine learning model-KNN. 
def short_position_knn_model(X_train_scaled, X_test_scaled, y_train, y_test, sp_signals_df):
    
    # Initiate the model instance
    sp_knn_model = KNeighborsClassifier()
    # Fit the model using the training data
    sp_knn_model = sp_knn_model.fit(X_train_scaled, y_train)
    # Use the testing dataset to generate the predictions for the new model
    sp_knn_y_pred = sp_knn_model.predict(X_test_scaled)
    

    # Use a classification report to evaluate the model using the predictions and testing data
    sp_knn_testing_report = classification_report(y_test, sp_knn_y_pred)
    
    # Create a new empty predictions DataFrame.
    # Create a predictions DataFrame
    sp_predictions_df_knn = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    sp_predictions_df_knn['predicted_signal'] = sp_knn_y_pred
    # Add the actual returns to the DataFrame
    sp_predictions_df_knn['actual_returns'] = sp_signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    sp_predictions_df_knn['strategy_returns'] = sp_predictions_df_knn['actual_returns'] * sp_predictions_df_knn['predicted_signal']
    
    
    # Plot the actual returns versus the strategy returns
    plot_actual_vs_strategy_returns_knn = (1 + sp_predictions_df_knn[["actual_returns", "strategy_returns"]]).cumprod().plot(
        title="Actual Returns vs Strategy Returns - KNN",
        xlabel="Date",
        ylabel="Cumulative Returns"
    )
    plot_actual_vs_strategy_returns_knn.get_figure().savefig("sp_actual_vs_strategy_returns_knn.png")

    return sp_knn_y_pred, sp_predictions_df_knn, sp_knn_testing_report, plot_actual_vs_strategy_returns_knn

   
def bigb_short_position_metric(holding_period, initial_capital, sp_predictions_df_svc):

    # Calculate the Sharpe ratio
    bigb_sp_annualized_returns = sp_predictions_df_svc['strategy_returns'].mean() * 252
    bigb_sp_annualized_volatility = sp_predictions_df_svc['strategy_returns'].std() * np.sqrt(252)
    bigb_sp_sharpe_ratio = bigb_sp_annualized_returns / bigb_sp_annualized_volatility
    
    
    # Calculate the Omega ratio
    threshold = 0
    sp_positive_returns = sp_predictions_df_svc['strategy_returns'][sp_predictions_df_svc['strategy_returns'] > threshold]
    sp_negative_returns = sp_predictions_df_svc['strategy_returns'][sp_predictions_df_svc['strategy_returns'] < threshold]

    sp_prob_positive_returns = len(sp_positive_returns) / len(sp_predictions_df_svc['strategy_returns'])
    sp_expected_positive_returns = sp_positive_returns.mean()
    sp_expected_negative_returns = sp_negative_returns.mean()
    bigb_sp_omega_ratio = (sp_prob_positive_returns - threshold) / (1 - threshold) * (sp_expected_positive_returns / abs(sp_expected_negative_returns))
      
    
    # Calculate the cumulative returns of the strategy
    sp_predictions_df_svc['cumulative_returns'] = (1 + sp_predictions_df_svc['strategy_returns']).cumprod()
    # Calculate the final value of the investment
    bigb_sp_final_capital = initial_capital * sp_predictions_df_svc['cumulative_returns'][-1]  
    # Calculate the total return and annualized return of the investment
    bigb_sp_total_return = bigb_sp_final_capital / initial_capital - 1
    bigb_sp_annualized_return = (1 + bigb_sp_total_return) ** (12 / holding_period) - 1
    
    return bigb_sp_sharpe_ratio, bigb_sp_omega_ratio, bigb_sp_final_capital, bigb_sp_total_return, bigb_sp_annualized_return


def sp500_short_position_metric(holding_period, initial_capital):
    
    # create a new DataFrame with sp500's data of past 3 years
    # Call the get_bigb_data() function to get the bigb_df, bigb_daily_returns, bigb_annual_return, bigb_annual_volatility
    sp500_df = data_plugin.get_sp500_data()
    # Add a daily return values column 'actual_return' to the DataFrame
    sp500_df['actual_returns'] = sp500_df['close'].pct_change()
    # Drop all NaN values from the DataFrame
    sp500_df.dropna(inplace=True)
    # Calculate the date 3 year ago from today's date to get the past 3 years' data of SP500
    start_date = datetime.now() - timedelta(days=1095)
    # Slice the past 3 year's DataFrame
    sp500_df_3y = sp500_df.loc[start_date:]
    
    
    # Generate a Dual Moving Average Crossover Trading Signal
    # Filter the date index and close, actual_returns columns
    sp_signals_df = sp500_df_3y.loc[:,['close','actual_returns']]
    # Set the short window and long window
    sp_short_window = 20
    sp_long_window = 50
    # Generate the short and long moving averages 
    sp_signals_df['SMA20'] = sp_signals_df['close'].rolling(window=sp_short_window).mean()
    sp_signals_df['SMA50'] = sp_signals_df['close'].rolling(window=sp_long_window).mean()
    sp_signals_df.dropna(inplace=True)
    # Create an empty Signal column
    sp_signals_df['Signal'] = 0.0
    # Generate the trading signal 0 or 1, for short position strategy
    sp_signals_df['Signal'] = np.where(
        sp_signals_df['SMA20'] < sp_signals_df['SMA50'], 1.0, 0.0
    )

    
    # Generate the trainning dataset and the testing dataset
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = sp_signals_df[['SMA20', 'SMA50']]
    # Create the target set selecting the Signal column and assiging it to y
    y = sp_signals_df['Signal']
    # Select the start and the end date of the training period
    training_begin = X.index.min()
    # Select the ending period for the training data with an offset of 2 years
    training_end = training_begin + DateOffset(years=2)
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


    # From SVM, instantiate SVC classifier model instance
    sp_svc_model = svm.SVC()
    # Fit the model to the data using the training data
    sp_svc_model = sp_svc_model.fit(X_train_scaled, y_train)
    # Use the testing data to make the model predictions
    sp_svc_y_pred = sp_svc_model.predict(X_test_scaled)


    # Create a new empty predictions DataFrame.
    sp_predictions_df_svc = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    sp_predictions_df_svc['predicted_signal'] = sp_svc_y_pred
    # Add the actual returns to the DataFrame
    sp_predictions_df_svc['actual_returns'] = sp_signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    sp_predictions_df_svc['strategy_returns'] = sp_predictions_df_svc['actual_returns'] * sp_predictions_df_svc['predicted_signal']

    
    # holding_period, initial_capital = questionnaire_plugin.questionnaire()
    
    # Calculate the Sharpe ratio
    sp500_sp_annualized_returns = sp_predictions_df_svc['strategy_returns'].mean() * 252
    sp500_sp_annualized_volatility = sp_predictions_df_svc['strategy_returns'].std() * np.sqrt(252)
    sp500_sp_sharpe_ratio = sp500_sp_annualized_returns / sp500_sp_annualized_volatility
    
    
    # Calculate the Omega ratio
    threshold = 0
    sp_positive_returns = sp_predictions_df_svc['strategy_returns'][sp_predictions_df_svc['strategy_returns'] > threshold]
    sp_negative_returns = sp_predictions_df_svc['strategy_returns'][sp_predictions_df_svc['strategy_returns'] < threshold]

    sp_prob_positive_returns = len(sp_positive_returns) / len(sp_predictions_df_svc['strategy_returns'])
    sp_expected_positive_returns = sp_positive_returns.mean()
    sp_expected_negative_returns = sp_negative_returns.mean()
    sp500_sp_omega_ratio = (sp_prob_positive_returns - threshold) / (1 - threshold) * (sp_expected_positive_returns / abs(sp_expected_negative_returns))
      
    
    # Calculate the cumulative returns of the strategy
    sp_predictions_df_svc['cumulative_returns'] = (1 + sp_predictions_df_svc['strategy_returns']).cumprod()
    # Calculate the final value of the investment
    sp500_sp_final_capital = initial_capital * sp_predictions_df_svc['cumulative_returns'][-1]  
    # Calculate the total return and annualized return of the investment
    sp500_sp_total_return = sp500_sp_final_capital / initial_capital - 1
    sp500_sp_annualized_return = (1 + sp500_sp_total_return) ** (12 / holding_period) - 1
    
    return sp500_sp_sharpe_ratio, sp500_sp_omega_ratio, sp500_sp_final_capital, sp500_sp_total_return, sp500_sp_annualized_return

 
    
    
     



    