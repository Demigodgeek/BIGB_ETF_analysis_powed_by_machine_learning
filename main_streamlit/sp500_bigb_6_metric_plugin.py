import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from tabulate import tabulate
# from sp500_knn_bestk_model_plugin import sp500_knn_bestk_model

# sp500_knn_y_pred, sp500_predictions_df_knn, sp500_knn_testing_report, plot_sp500_actual_vs_strategy_returns_knn = sp500_knn_bestk_model(sp500_trading_df)

def bigb_sp500_metric_table(holding_period, initial_capital, bigb_predictions_df_knn, sp500_predictions_df_knn):
    
    # Get the Big Bank's metric
    
    # Calculate the Sharpe ratio
    bigb_annualized_returns = bigb_predictions_df_knn['strategy_returns'].mean() * 252
    bigb_annualized_volatility = bigb_predictions_df_knn['strategy_returns'].std() * np.sqrt(252)
    bigb_sharpe_ratio = bigb_annualized_returns / bigb_annualized_volatility
    
    # Calculate the Omega ratio
    threshold = 0
    bigb_positive_returns = bigb_predictions_df_knn['strategy_returns'][bigb_predictions_df_knn['strategy_returns'] > threshold]
    bigb_negative_returns = bigb_predictions_df_knn['strategy_returns'][bigb_predictions_df_knn['strategy_returns'] < threshold]

    bigb_prob_positive_returns = len(bigb_positive_returns) / len(bigb_predictions_df_knn['strategy_returns'])
    bigb_expected_positive_returns = bigb_positive_returns.mean()
    bigb_expected_negative_returns = bigb_negative_returns.mean()
    
    bigb_omega_ratio = (bigb_prob_positive_returns - threshold) / (1 - threshold) * (bigb_expected_positive_returns / abs(bigb_expected_negative_returns))
      
    # Calculate the cumulative returns of the strategy
    bigb_predictions_df_knn['cumulative_returns'] = (1 +bigb_predictions_df_knn['strategy_returns']).cumprod()
    
    # Calculate the final value of the investment
    bigb_final_capital = initial_capital * bigb_predictions_df_knn['cumulative_returns'][-1]  
    
    # Calculate the total return and annualized return of the investment
    bigb_total_return = bigb_final_capital / initial_capital - 1
    bigb_annualized_return = (1 + bigb_total_return) ** (12 / holding_period) - 1
     
    
    # Get the S&P 500 Metric
    
    # Calculate the Sharpe ratio
    sp500_annualized_returns = sp500_predictions_df_knn['strategy_returns'].mean() * 252
    sp500_annualized_volatility = sp500_predictions_df_knn['strategy_returns'].std() * np.sqrt(252)
    sp500_sharpe_ratio = sp500_annualized_returns / sp500_annualized_volatility
    
    # Calculate the Omega ratio
    threshold = 0
    sp500_positive_returns = sp500_predictions_df_knn['strategy_returns'][sp500_predictions_df_knn['strategy_returns'] > threshold]
    sp500_negative_returns = sp500_predictions_df_knn['strategy_returns'][sp500_predictions_df_knn['strategy_returns'] < threshold]

    sp500_prob_positive_returns = len(sp500_positive_returns) / len(sp500_predictions_df_knn['strategy_returns'])
    sp500_expected_positive_returns = sp500_positive_returns.mean()
    sp500_expected_negative_returns = sp500_negative_returns.mean()
    
    sp500_omega_ratio = (sp500_prob_positive_returns - threshold) / (1 - threshold) * (sp500_expected_positive_returns / abs(sp500_expected_negative_returns))
      
    # Calculate the cumulative returns of the strategy
    sp500_predictions_df_knn['cumulative_returns'] = (1 + sp500_predictions_df_knn['strategy_returns']).cumprod()
    
    # Calculate the final value of the investment
    sp500_final_capital = initial_capital * sp500_predictions_df_knn['cumulative_returns'][-1]  
   
    # Calculate the total return and annualized return of the investment
    sp500_total_return = sp500_final_capital / initial_capital - 1
    sp500_annualized_return = (1 + sp500_total_return) ** (12 / holding_period) - 1
    

    # Create bigb_sp500_metric_table 
    
    table_data = [["Metric", "BIG BANK", "S&P500"],
              ["Sharpe Ratio", bigb_sharpe_ratio, sp500_sharpe_ratio],
              ["Omega Ratio", bigb_omega_ratio, sp500_omega_ratio],
              ["Final Capital", bigb_final_capital, sp500_final_capital],
              ["Total Return", bigb_total_return, sp500_total_return],
              ["Annualized Return", bigb_annualized_return, sp500_annualized_return]]

    bigb_sp500_metric_string= (tabulate(table_data, headers="firstrow", tablefmt="plain"))

    # Use io.StringIO to create a file-like object for the string
    bigb_sp500_metric_table = io.StringIO(bigb_sp500_metric_string)

    # Read the file into a pandas DataFrame
    bigb_sp500_df = pd.read_csv(bigb_sp500_metric_table, sep="|", skipinitialspace=True, engine='python')
    
    return bigb_sp500_df

