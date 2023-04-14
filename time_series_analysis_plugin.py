#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Import the necessary modules
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

from data_plugin_for_julio import get_bigb_data

bigb_df = get_bigb_data()[0]


# In[36]:


bigb_df


# In[37]:


print(bigb_df.columns)


# In[38]:


print(bigb_df.index.name)


# # Seasonal Closing Price

# In[39]:


quarterly_close_prices = bigb_df.resample('Q').mean()
quarterly_close_prices['Year'] = quarterly_close_prices.index.year
quarterly_close_prices['Quarter'] = quarterly_close_prices.index.quarter

# Plot the heatmap
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
# Save the elbow plot as a PNG file in the "images" directory
file_path = 'images/heatmap.png'
hvplot.save(heatmap, file_path, fmt='png')
heatmap


# # Prophet 

# In[40]:


# Prepare the data for Prophet
prophet_data = bigb_df.reset_index()[['date', 'close']]
prophet_data.columns = ['ds', 'y']


# In[65]:


# Fit the model
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Make predictions
future = prophet_model.make_future_dataframe(periods=365)
forecast = prophet_model.predict(future)

# Prophet plot
prophet_fig = prophet_model.plot(forecast)
prophet_ax = prophet_fig.gca()

# Save the Prophet plot
prophet_fig.savefig('images/prophet_plot.png')


# # Prepare the data for SARIMA and ARIMA

# In[49]:


# Prepare the data for SARIMA and ARIMA
sarima_data = bigb_df['close']


# # SARIMA

# In[70]:


# Fit the model
sarima_model = SARIMAX(sarima_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_results = sarima_model.fit()

# Make predictions
sarima_pred = sarima_results.get_prediction(start=pd.to_datetime('2023-04-13'), dynamic=False)
sarima_pred_conf = sarima_pred.conf_int()

# SARIMA plot
sarima_plot = sarima_data.plot(label='Observed')
sarima_pred.predicted_mean.plot(ax=sarima_plot, label='Predicted', alpha=0.7)
sarima_plot.fill_between(sarima_pred_conf.index,
                         sarima_pred_conf.iloc[:, 0],
                         sarima_pred_conf.iloc[:, 1], color='k', alpha=0.2)
sarima_plot.legend()

# Save the SARIMA plot
sarima_plot.figure.savefig('images/sarima_plot.png')


# # ARIMA

# In[71]:


# Fit the model
arima_model = ARIMA(sarima_data, order=(1, 1, 1))
arima_results = arima_model.fit()

# Make predictions
arima_pred = arima_results.get_prediction(start=pd.to_datetime('2023-04-13'), dynamic=False)
arima_pred_conf = arima_pred.conf_int()

# Plot the predictions
arima_plot = sarima_data.plot(label='Observed')
arima_pred.predicted_mean.plot(ax=arima_plot, label='Predicted', alpha=0.7)
arima_plot.fill_between(arima_pred_conf.index,
                        arima_pred_conf.iloc[:, 0],
                        arima_pred_conf.iloc[:, 1], color='k', alpha=0.2)
arima_plot.legend()

# Save the ARIMA plot
arima_plot.figure.savefig('images/arima_plot.png')


# In[72]:


# Convert Prophet predictions to DataFrame
prophet_pred_df = forecast.set_index('ds')[['yhat']].rename(columns={'yhat': 'Prophet'})

# Convert SARIMA and ARIMA predictions to DataFrame
sarima_pred_df = sarima_pred.predicted_mean.to_frame().rename(columns={0: "SARIMA"})
arima_pred_df = arima_pred.predicted_mean.to_frame().rename(columns={0: "ARIMA"})

# Merge predictions DataFrames
predictions_df = pd.concat([prophet_pred_df, sarima_pred_df, arima_pred_df], axis=1)

# Plot the actual values, Prophet, SARIMA, and ARIMA predictions
fig, ax = plt.subplots(figsize=(15, 8))
bigb_df['adjclose'].plot(ax=ax, label='BIGB: Actual', linewidth=2)
predictions_df['Prophet'].plot(ax=ax, label='Prophet', linestyle='--', linewidth=2)
predictions_df['SARIMA'].plot(ax=ax, label='SARIMA', linestyle=':', linewidth=2)
predictions_df['ARIMA'].plot(ax=ax, label='ARIMA', linestyle='-.', linewidth=2)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.title('BIGB: Actual vs. Predicted Prices')
plt.show()
# Save Plot
fig.savefig('images/combined_plot.png')


# In[ ]:




