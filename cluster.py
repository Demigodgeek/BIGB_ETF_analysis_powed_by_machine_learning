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

# # Clustering

# In[29]:


# Perform clustering and find the appropriate cluster for BIGB
def find_bigb_cluster(data, bigb_annual_return, bigb_annual_volatility, n_clusters=5):
    # Preprocess the data
    data = data.dropna()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Fit the KMeans model and find the optimal cluster number
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_scaled)
    
    # Assign the cluster number to each data point
    data["cluster"] = kmeans.labels_
    
    # Add BIGB data point to the dataset
    bigb_point = pd.DataFrame({"annual_return": [bigb_annual_return], "annual_volatility": [bigb_annual_volatility]})
    bigb_point_scaled = scaler.transform(bigb_point)
    
    # Find the cluster for BIGB
    bigb_cluster = kmeans.predict(bigb_point_scaled)[0]
    
    return data, bigb_cluster

bigb_data = get_bigb_data()
benchmark_data = get_benchmark_cluster_data()
bank_etf_data = get_bank_etf_cluster_data()

bigb_daily_returns = bigb_data["close"].pct_change().tail(756)
bigb_annual_return = np.mean(bigb_daily_returns) * 252
bigb_annual_volatility = np.std(bigb_daily_returns) * np.sqrt(252)


benchmark_data_clustered, bigb_benchmark_cluster = find_bigb_cluster(benchmark_data, bigb_annual_return, bigb_annual_volatility)
bank_etf_data_clustered, bigb_bank_etf_cluster = find_bigb_cluster(bank_etf_data, bigb_annual_return, bigb_annual_volatility)

# Plot the clusters
def plot_clusters(data, bigb_cluster, title):
    scatter = hv.Scatter(data, kdims=['annual_return', 'annual_volatility'], vdims=['cluster']).opts(color='cluster', cmap='Category10', size=10, tools=['hover'], title=title)
    bigb_scatter = hv.Scatter((bigb_annual_return, bigb_annual_volatility)).opts(color='red', marker='x', size=20)
    bigb_label = hv.Text(bigb_annual_return, bigb_annual_volatility, "BIGB").opts(text_color='black', text_font_size='10pt')
    plot = (scatter * bigb_scatter * bigb_label).opts(legend_position='top', width=800, height=400, xlabel='Annual Return', ylabel='Annual Volatility')
    
    # Save the plot as a PNG file in the "images" folder
    hv.save(plot, file_path, fmt='png')
    
    return plot


# In[30]:


# # Save the plot as a PNG file in the "images" folder
# hv.save(plot, file_path, fmt='png')
file_path = 'images/benchmark_plot.png'
benchmark_plot = plot_clusters(benchmark_data_clustered, bigb_benchmark_cluster, "Benchmark Clustering")
benchmark_plot


# In[31]:


# # Save the plot as a PNG file in the "images" folder
# hv.save(plot, file_path, fmt='png')
file_path = 'images/bank_etf_plot.png'
bank_etf_plot = plot_clusters(bank_etf_data_clustered, bigb_bank_etf_cluster, "Bank ETF Clustering")
bank_etf_plot




# In[ ]:




