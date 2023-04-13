#!/usr/bin/env python
# coding: utf-8

# time_series.py


"""


This code is split into several functions that use time series models to analyze and make predictions about the stock prices of the BIGB asset.

plot_quarterly_heatmap: This function creates a heatmap that shows the quarterly close prices of the BIGB asset per year. It takes in the data as input, and returns the heatmap plot.

train_prophet_model: This function fits a Prophet model using the training data and returns the fitted model.

predict_prophet: This function makes predictions using the fitted Prophet model and returns the predicted values.

train_sarima_model: This function fits a SARIMA model using the training data and returns the fitted model.

predict_sarima: This function makes predictions using the fitted SARIMA model and returns the predicted values.

calculate_rmse: This function calculates the RMSE (root mean squared error) of the predicted values compared to the actual values.

The code imports the necessary modules such as pandas, numpy, holoviews, Prophet, SARIMAX, mean_squared_error, and sqrt. It also imports the required data from the data_plugin module.

Then, the code defines the plot_quarterly_heatmap function to create a heatmap plot that shows the quarterly close prices of the BIGB asset per year.

Next, the code defines the train_prophet_model function, which fits a Prophet model using the training data and returns the fitted model. Then, the code defines the predict_prophet function, which makes predictions using the fitted Prophet model and returns the predicted values.

The code also defines the train_sarima_model function, which fits a SARIMA model using the training data and returns the fitted model. Then, the code defines the predict_sarima function, which makes predictions using the fitted SARIMA model and returns the predicted values.

Finally, the code defines the calculate_rmse function, which calculates the RMSE of the predicted values compared to the actual values. It then creates a plot that shows the actual vs. predicted close prices of the BIGB asset, and calculates the RMSE for both models.


"""


# Import the necessary modules
import pandas as pd
import numpy as np
import holoviews as hv
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

# Import the required data from the data_plugin module
from data_plugin import bigb_data, train_data, test_data

def plot_quarterly_heatmap(quarterly_close_prices):
    quarterly_close_prices = quarterly_close_prices.resample('Q').mean()
    quarterly_close_prices['Year'] = quarterly_close_prices.index.year
    quarterly_close_prices['Quarter'] = quarterly_close_prices.index.quarter
    
    heatmap = quarterly_close_prices.hvplot.heatmap(
        x="Year",
        y="Quarter",
        C="close",
        cmap="blues",
        width=800,
        height=400,
        colorbar=True,
        title="BIGB Quarterly Close Price per Year Heatmap"
    )

    # Save the plot as a PNG file in the "images" folder
    hv.save(heatmap, file_path, fmt='png')
    
    return heatmap

# Plot the heatmap for BIGB data
file_path = 'images/heatmap.png'
heatmap = plot_quarterly_heatmap(bigb_data)
heatmap


# Plot the heatmap for BIGB data
file_path = 'images/heatmap.png'
heatmap = plot_quarterly_heatmap(bigb_data)
heatmap


# Define the train_prophet_model function
hv.extension('bokeh')
renderer = hv.renderer('bokeh')

# existing code to get the data
bigb_data = get_bigb_data()
bigb_daily_returns = bigb_data["close"].pct_change().tail(756)
bigb_annual_return = np.mean(bigb_daily_returns) * 252
bigb_annual_volatility = np.std(bigb_daily_returns) * np.sqrt(252)

# Prepare the data for Prophet
prophet_data = bigb_data.reset_index()[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})

# Split the data into train and test sets
train_data = prophet_data.iloc[:-365]
test_data = prophet_data.iloc[-365:]

# Fit the Prophet model using the train data
prophet_model = Prophet()
prophet_model.fit(train_data)

# Make predictions using the test data
prophet_future = prophet_model.make_future_dataframe(periods=365, freq='D')
prophet_forecast = prophet_model.predict(prophet_future)
# Plot the Prophet forecast
file_path = 'images/prophet_predicted_plot.png'
prophet_predicted_plot = plot(prophet_model, prophet_forecast)

# SARIMA

# Fit the SARIMA model using the train data
sarima_model = SARIMAX(train_data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_results = sarima_model.fit()

# Make predictions using the test data
sarima_predicted_close = sarima_results.get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
sarima_predicted_close_df = sarima_predicted_close.conf_int()
sarima_predicted_close_df['yhat'] = (sarima_predicted_close_df['lower y'] + sarima_predicted_close_df['upper y']) / 2
sarima_predicted_close_df['ds'] = test_data['ds']  # Add 'ds' column for plotting

# Combine the actual and predicted data
combined_data = test_data.copy()
combined_data['yhat'] = sarima_predicted_close_df['yhat']

# Plot the actual and predicted data
sarima_predicted_plot = combined_data.hvplot(x='ds', y=['y', 'yhat'], ylabel='Close Price', legend=True)

sarima_predicted_plot

# Calculate the RMSE for both models

# Calculate the RMSE for both models
prophet_predicted_close = prophet_forecast.iloc[-365:]['yhat']
actual_close = test_data['y']
prophet_mse = mean_squared_error(actual_close, prophet_predicted_close)
prophet_rmse = sqrt(prophet_mse)

sarima_predicted_close = sarima_predicted_close_df.iloc[-365:]['yhat']
sarima_mse = mean_squared_error(actual_close, sarima_predicted_close)
sarima_rmse = sqrt(sarima_mse)

print("Prophet RMSE:", prophet_rmse)
print("SARIMA RMSE:", sarima_rmse)

# Add the Prophet predictions to the combined_data DataFrame
combined_data['prophet_yhat'] = prophet_forecast.iloc[-365:]['yhat'].values

# Plot the actual data and predictions from both models
actual_vs_predicted_plot = combined_data.hvplot(x='ds', y=['y', 'yhat', 'prophet_yhat'], ylabel='Close Price', legend=True)

actual_vs_predicted_plot

# In[ ]:




