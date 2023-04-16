import streamlit as st
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import io
import logging
import hvplot.pandas
import matplotlib.pyplot as plt
import holoviews as hv
from bokeh.models import HoverTool
hv.extension('bokeh')
from tabulate import tabulate
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#############################
##import all modlues needed##
#############################

from _preparation_data_plugin import get_bigb_data, get_benchmark_cluster_data, get_bank_etf_cluster_data
from _x_y_tune_widget_plugin import bigb_model_data_preparation
from bigb_1_cluster_plugin import get_benchmark_cluster, get_bank_etf_cluster
from bigb_2_season_price_heatmap_plugin import plot_bigb_seasonal_pattern_heatmap
from bigb_3_visualize_short_long_position_strategy_plugin import visualize_long_position_strategy, visualize_short_position_strategy
from bigb_4_1_svc_model_plugin import bigb_svc_model
from bigb_4_2_rf_model_plugin import bigb_rf_model
from bigb_4_3_knn_bestk_model_plugin import bigb_knn_bestk_model
from sp500_5_knn_bestk_model_plugin import sp500_knn_bestk_model
from sp500_bigb_6_metric_plugin import bigb_sp500_metric_table

############################
##Call necessary functions##
##    get return values   ##
############################

# Data Preparation
bigb_df, bigb_trading_df, bigb_cluster_df = get_bigb_data()
bank_etf_cluster_df = get_bank_etf_cluster_data()
benchmark_cluster_df = get_benchmark_cluster_data()
 

# Trading algorithms, Model, Backtest, and Evaluation
benchmark_cluster_plot = get_benchmark_cluster(benchmark_cluster_df)
bank_etf_cluster_plot = get_bank_etf_cluster(bank_etf_cluster_df)

price_heatmap = plot_bigb_seasonal_pattern_heatmap(bigb_df)
sp_entry_exit_plot = visualize_short_position_strategy(bigb_trading_df)
lp_entry_exit_plot = visualize_long_position_strategy(bigb_trading_df)

X_train_scaled, X_test_scaled, y_train, y_test, signals_df = bigb_model_data_preparation(bigb_trading_df)
bigb_svc_y_pred, bigb_predictions_df_svc, bigb_svc_testing_report, plot_actual_vs_strategy_returns_svc = bigb_svc_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
bigb_rf_y_pred, bigb_predictions_df_rf, bigb_rf_testing_report, plot_actual_vs_strategy_returns_rf = bigb_rf_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
bigb_knn_y_pred, bigb_predictions_df_knn, bigb_knn_testing_report, plot_actual_vs_strategy_returns_knn = bigb_knn_bestk_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
sp500_knn_y_pred, sp500_predictions_df_knn, sp500_knn_testing_report, plot_sp500_actual_vs_strategy_returns_knn = sp500_knn_bestk_model()

######################
##Streamlit Page Set##
######################

# set up streamlit page
st.set_page_config(page_title="My Webpage", page_icon="tda", layout="wide")

# add background
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://raw.githubusercontent.com/Demigodgeek/from_zero_to_hero/main/background.jpeg");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()

# add left margin
def add_left_margin():
    st.markdown(
        """
        <style>
        .main {
            margin-right: 20%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
add_left_margin()


# Set sidebar 1/2 - questionnaire
st.sidebar.title("Is BIGB a wise choice for you?")
st.sidebar.write("This analysis is for informational purposes only and does not constitute investment advice.")

st.sidebar.write("---")

# create question list
holding_period = st.sidebar.selectbox("What period do you want to hold this stock for?", list(range(1, 37)), index=0, format_func=lambda x: f'{x} month(s)')
initial_capital = st.sidebar.slider("Please select your initial capital (in USD)", 10000, 100000, 10000, step=1000, format="$%.0f")
# from the question list above get the user input as parameters to get the bigb_sp500_df value
bigb_sp500_df = bigb_sp500_metric_table(holding_period, initial_capital, bigb_predictions_df_knn, sp500_predictions_df_knn)

st.write("---")
st.write("---")
st.sidebar.subheader("Click one or more check boxes to read the visulization report of Big Bank")

# Set sidebar 2/2 - choice list
analysis_choices = ["Reward/Risk Comparison", "Seasonal Pattern in Price", "Visualization of Short/Long Position Strategy", 
                    "Compare Strategy Return with 3 Models", "Compare Strategy Return with S&P500"]

# create check boxes
reward_risk_comparison = st.sidebar.checkbox("Reward/Risk Comparison")
seasonal_pattern = st.sidebar.checkbox("Seasonal Pattern in Price")
position_strategy = st.sidebar.checkbox("Visualization of Short/Long Position Strategy")
compare_3_models = st.sidebar.checkbox("Compare Strategy Return of 3 Models")
compare_strategy_return = st.sidebar.checkbox("Compare Strategy Return: BIG Bank vs S&P500")
compare_financial_metrics = st.sidebar.checkbox("Compare Financial Metrics: BIG Bank vs S&P 500")

st.write("---")

# process selected choices
if reward_risk_comparison:
    st.subheader("Reward/Risk Comparison")

    st.write("---")

    st.write("Here is the benchmark cluster plot:")
    st.write("---")
    st.bokeh_chart(benchmark_cluster_plot)
    st.write("---")
    st.write("Here is the return bank ETF cluster plot:")
    st.write("---")
    st.bokeh_chart(bank_etf_cluster_plot)

    st.write("---")

if seasonal_pattern:
    st.subheader("Seasonal Pattern in Price")

    st.write("---")

    st.write("As shown in the following figure, ")
    st.write("the price of the stock does not show an obvious seasonality,")
    st.write("but it seems to have a cycle of 16 years.")
    st.write("---")
    st.bokeh_chart(price_heatmap)

    st.write("---")

if position_strategy:
    st.subheader("Visualization of Short/Long Position Strategy")
    st.write("---")
    st.write("Here is the short position entry/exit plot:")
    st.write("---")
    st.bokeh_chart(sp_entry_exit_plot)
    st.write("---")
    st.write("Here is the long position entry/exit plot:")
    st.write("---")
    st.bokeh_chart(lp_entry_exit_plot)

    st.write("---")

if compare_3_models:
    st.subheader("Compare Strategy Return with 3 Models")
    st.write("---")
    st.write("Here is the actual vs strategy returns plot of SVM:")
    st.pyplot(plot_actual_vs_strategy_returns_svc)
    st.write("---")
    st.write("Here is the actual vs strategy returns plot of Random Forest:")
    st.pyplot(plot_actual_vs_strategy_returns_rf)
    st.write("---") 
    st.write("Here is the actual vs strategy returns plot of KNN:")
    st.pyplot(plot_actual_vs_strategy_returns_knn)

if compare_strategy_return:
    st.subheader("Compare Strategy Return with 3 Models")
    st.write("Here is BIG Bank'S actual vs strategy returns plot for SVM:")
    st.pyplot(plot_actual_vs_strategy_returns_knn)
    st.write("Here is S&P500's actual vs strategy returns plot for KNN:")
    st.pyplot(plot_sp500_actual_vs_strategy_returns_knn)

if compare_financial_metrics:
    st.subheader("Compare Important Financial Metrics")
    st.write(bigb_sp500_df)