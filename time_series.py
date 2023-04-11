#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[26]:


# Import Libraries and dependancies 
import pandas as pd
import numpy as np
import os
import hvplot.pandas
import holoviews as hv
import yahoo_fin.stock_info as si
import yfinance as yf
import warnings 
warnings.filterwarnings("ignore")
from math import sqrt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


# In[27]:



# # Time Series Forecast

# In[32]:


# Time Series Forecast
def plot_quarterly_heatmap(data):
    data = data.resample('Q').mean()
    data['Year'] = data.index.year
    data['Quarter'] = data.index.quarter
    data = data.reset_index().melt(id_vars=['Year', 'Quarter'], value_vars=['close'], value_name='Close')
    
    heatmap = hv.HeatMap(data, 
                         kdims=['Year', 'Quarter'], 
                         vdims=['Close']).opts(cmap="coolwarm", 
                                               colorbar=True, toolbar="above", 
                                               width=800, height=400,
                                                                               
                                               title="BIGB Quarterly Close Price per Year Heatmap", 
                                               xrotation=90)
    # Save the plot as a PNG file in the "images" folder
    hv.save(heatmap, file_path, fmt='png')
    return heatmap

# Plot the heatmap for BIGB data
file_path = 'images/heatmap.png'
heatmap = plot_quarterly_heatmap(bigb_data)


# In[33]:


heatmap


# # Prophet

# In[34]:


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


# In[35]:


from prophet.plot import plot

# Plot the Prophet forecast
prophet_predicted_plot = plot(prophet_model, prophet_forecast)


# In[36]:


# prophet_predicted_plot.show()
# # Make predictions using the test data
# prophet_future = prophet_model.make_future_dataframe(periods=365, freq='D')
# prophet_forecast = prophet_model.predict(prophet_future)


# # SARIMA

# In[37]:


# Fit the SARIMA model using the train data
sarima_model = SARIMAX(train_data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_results = sarima_model.fit()

# Find the row numbers for the start and end dates
start_row = test_data.index[0]
end_row = test_data.index[-1]

# Make predictions using the test data
sarima_predicted_close = sarima_results.get_prediction(start=start_row, end=end_row, dynamic=False)
sarima_predicted_close_df = sarima_predicted_close.conf_int()
sarima_predicted_close_df['yhat'] = (sarima_predicted_close_df['lower y'] + sarima_predicted_close_df['upper y']) / 2
sarima_predicted_close_df['ds'] = test_data['ds']  # Add 'ds' column for plotting


# In[38]:


import hvplot.pandas

# Plot the predicted values
sarima_predicted_close_df.hvplot(
    x='ds',
    y='yhat',
    title='SARIMA Predicted Close Prices',
    xlabel='Date',
    ylabel='Close Price'
)


# # Calculate the RMSE for both models

# In[39]:


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

# Create a plot of actual vs. predicted close prices
actual_vs_predicted = pd.DataFrame({'ds': actual_close.index, 'actual': actual_close.values, 
                                     'prophet_predicted': prophet_predicted_close.values,
                                     'sarima_predicted': sarima_predicted_close.values})
actual_vs_predicted.set_index('ds', inplace=True)

actual_vs_predicted.hvplot(title='Actual vs. Predicted Close Prices', xlabel='Date', ylabel='Price')


# In[ ]:

# In[ ]:




