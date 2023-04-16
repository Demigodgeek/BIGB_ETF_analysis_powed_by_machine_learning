import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from warnings import filterwarnings
filterwarnings("ignore")

# Apply the tuned X, y, and the examing period to the second machine learning model - Random Forest, then backtest and evaluate the model. 

def bigb_rf_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    
    # Instantiate a random forest classifier model
    bigb_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the model to the training data
    bigb_rf_model.fit(X_train_scaled, y_train)
    # Use the testing data to make predictions
    bigb_rf_y_pred = bigb_rf_model.predict(X_test_scaled)
    
    # Use a classification report to evaluate the model's performance
    bigb_rf_testing_report = classification_report(y_test, bigb_rf_y_pred)
    
    # Create a DataFrame to store the predicted signals, actual returns, and strategy returns
    bigb_predictions_df_rf = pd.DataFrame(index=y_test.index)
    # Add the predicted signals to the DataFrame
    bigb_predictions_df_rf['predicted_signal'] = bigb_rf_y_pred
    # Add the actual returns to the DataFrame
    bigb_predictions_df_rf['actual_returns'] = signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    bigb_predictions_df_rf['strategy_returns'] = bigb_predictions_df_rf['actual_returns'] * bigb_predictions_df_rf['predicted_signal']

    # Plot the actual returns versus the strategy returns
    actual_vs_strategy_returns_rf = (1 + bigb_predictions_df_rf[["actual_returns", "strategy_returns"]]).cumprod() 

    # Create a line plot of the cumulative returns
    plot_actual_vs_strategy_returns_rf, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual_vs_strategy_returns_rf)
    ax.legend(actual_vs_strategy_returns_rf.columns)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.set_title('Actual vs. Strategy Cumulative Returns')

    return bigb_rf_y_pred, bigb_predictions_df_rf, bigb_rf_testing_report, plot_actual_vs_strategy_returns_rf