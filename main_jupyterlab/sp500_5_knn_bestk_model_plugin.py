import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from warnings import filterwarnings
filterwarnings("ignore")

def sp500_knn_bestk_model():
    
    # set end_date as today and start_date as 10 years ago
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*20)).strftime("%Y-%m-%d")
    sp500_data = si.get_data('^GSPC', start_date=start_date, end_date=end_date)
    sp500_df= pd.DataFrame(sp500_data)
    
    # create a new dataframe for sp 500 which can be used for comparing with bigb in trading algorithm
    
    # extract the close column and calculate the daily returns
    sp500_daily_returns = sp500_df["close"].pct_change().dropna()
    # create a new dataframe with close and actual_return columns
    sp500_trading_df = pd.DataFrame({"close": sp500_df["close"],"actual_returns": sp500_daily_returns})
    # drop the first row since it contains a NaN value
    sp500_trading_df = sp500_trading_df.dropna()
     
    # create a new dataframe to store the feartures and signals
    sp500_signals_df = sp500_trading_df

    # Set the short window and long window
    short_window = 30
    long_window = 60

    # Generate the fast and slow simple moving averages
    sp500_signals_df['SMA_Fast'] = sp500_signals_df['close'].rolling(window=short_window).mean()
    sp500_signals_df['SMA_Slow'] = sp500_signals_df['close'].rolling(window=long_window).mean()

    sp500_signals_df = sp500_signals_df.dropna()
    
    # Initialize the new Signal column
    sp500_signals_df['Signal'] = 0.0
    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    sp500_signals_df.loc[(sp500_signals_df['actual_returns'] >= 0), 'Signal'] = 1
    # When Actual Returns are less than 0, generate signal to sell stock short
    sp500_signals_df.loc[(sp500_signals_df['actual_returns'] < 0), 'Signal'] = -1

    # Generate the trainning dataset and the testing dataset
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = sp500_signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna().copy()
    # Create the target set selecting the Signal column and assiging it to y
    y = sp500_signals_df['Signal']
    
    # Select the start and the end date of the training period
    training_begin = X.index.min()
    # Select the ending period for the training data with an offset of 15 (20*75%) years
    training_end = training_begin + DateOffset(years=15)
    # Generate the X_train and y_train DataFrames
    sp500_X_train = X.loc[training_begin:training_end]
    sp500_y_train = y.loc[training_begin:training_end]
    # Generate the X_test and y_test DataFrames
    sp500_X_test = X.loc[training_end+DateOffset(hours=1):]
    sp500_y_test = y.loc[training_end+DateOffset(hours=1):]

    # Scale the features DataFrames
    scaler = StandardScaler()
    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(sp500_X_train)
    # Transform the X_train and X_test DataFrames using the X_scaler
    sp500_X_train_scaled = X_scaler.transform(sp500_X_train)
    sp500_X_test_scaled = X_scaler.transform(sp500_X_test)

    # Define the range of values to search for k
    sp500_param_grid = {'n_neighbors': np.arange(1, 7)}

    # Create a KNN model
    sp500_knn_model = KNeighborsClassifier()

    # Create a grid search object to find the best value of k using 5-fold cross-validation
    sp500_grid_search = GridSearchCV(sp500_knn_model, sp500_param_grid, cv=5)

    # Fit the grid search object to the training data
    sp500_grid_search.fit(sp500_X_train_scaled, sp500_y_train)

    # Here you could remove the # to print the best value of k and its corresponding accuracy score
    # print("Best k:", sp500_grid_search.best_params_['n_neighbors'])
    # print("Best accuracy:", sp500_grid_search.best_score_)

    # Use the best value of k to fit the KNN model on the training data
    sp500_knn_model = KNeighborsClassifier(n_neighbors=sp500_grid_search.best_params_['n_neighbors'])
    sp500_knn_model.fit(sp500_X_train_scaled, sp500_y_train)

    # Make predictions on the test set using the trained model
    sp500_knn_y_pred = sp500_knn_model.predict(sp500_X_test_scaled)

    # Use a classification report to evaluate the model using the predictions and testing data
    sp500_knn_testing_report = classification_report(sp500_y_test, sp500_knn_y_pred)

    # Create a new empty predictions DataFrame.
    # Create a predictions DataFrame
    sp500_predictions_df_knn = pd.DataFrame(index=sp500_y_test.index)
    # Add the SVM model predictions to the DataFrame
    sp500_predictions_df_knn['predicted_signal'] = sp500_knn_y_pred
    # Add the actual returns to the DataFrame
    sp500_predictions_df_knn['actual_returns'] = sp500_signals_df.loc[sp500_y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    sp500_predictions_df_knn['strategy_returns'] = sp500_predictions_df_knn['actual_returns'] * sp500_predictions_df_knn['predicted_signal']

    # Plot the actual returns versus the strategy returns
    plot_sp500_actual_vs_strategy_returns_knn = (1 + sp500_predictions_df_knn[["actual_returns", "strategy_returns"]]).cumprod().plot(
        title="S&P500 - Actual Returns vs Strategy Returns - KNN",
        xlabel="Date",
        ylabel="Cumulative Returns"
    )
    plt.savefig("images/S&P500 - Actual Returns vs Strategy Returns - KNN.png")

    return sp500_knn_y_pred, sp500_predictions_df_knn, sp500_knn_testing_report, plot_sp500_actual_vs_strategy_returns_knn