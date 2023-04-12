#!/usr/bin/env python
# coding: utf-8

# # Clustering


"""

The necessary modules are imported, which include pandas, numpy, Holoviews, KMeans and StandardScaler.

Required data for BIGB, benchmark, and bank ETFs is imported from the data_plugin module.

The find_bigb_cluster function is defined, which takes in all_assets_data, bigb_annual_return, bigb_annual_volatility, and n_clusters as inputs. The function preprocesses the all_assets_data by removing missing values, replacing infinite values with NaN, and filling NaN with column mean values. The function also standardizes the all_assets_data, fits the KMeans model, and assigns the cluster number to each asset in the dataset. The function adds BIGB's data point to the dataset and finds the cluster for BIGB using the KMeans model. The function returns the clustered asset data and BIGB's cluster label.

The find_bigb_cluster function is called twice with benchmark_data and bank_etf_data as all_assets_data, and bigb_annual_return and bigb_annual_volatility as the other inputs. 

**** The function returns two sets of variables, benchmark_data_clustered and bigb_benchmark_cluster, and bank_etf_data_clustered and bigb_bank_etf_cluster.****

The plot_clusters function is defined, which takes in data, bigb_cluster, and title as inputs. The function creates a scatter plot of the data, with the colors representing the clusters. The function also adds a BIGB scatter point, BIGB text label, and customizes the plot by adding a title, legend, and axis labels.

The plot_clusters function is called twice with benchmark_data_clustered and bigb_benchmark_cluster, and bank_etf_data_clustered and bigb_bank_etf_cluster, respectively. The function returns a Holoviews plot, which is saved as a PNG file in the "images" folder.



"""


# cluster.py

# Import the necessary modules
import pandas as pd
import numpy as np
import holoviews as hv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import the required data from the data_plugin module
from data_plugin import bigb_annual_return, bigb_annual_volatility, benchmark_data, bank_etf_data

# Define the find_bigb_cluster function
# Perform clustering and find the appropriate cluster for BIGB
def find_bigb_cluster(all_assets_data, bigb_annual_return, bigb_annual_volatility, n_clusters=5):
    # Preprocess the all_assets_data (remove missing values, replace infinite values, fill NaN with mean)
    all_assets_data = all_assets_data.dropna()
    all_assets_data = all_assets_data.replace([np.inf, -np.inf], np.nan)
    all_assets_data = all_assets_data.fillna(all_assets_data.mean())

    # Standardize the all_assets_data
    scaler = StandardScaler()
    all_assets_data_scaled = scaler.fit_transform(all_assets_data)
    
    # Fit the KMeans model and find the optimal cluster number
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_assets_data_scaled)
    
    # Assign the cluster number to each asset in the dataset
    all_assets_data["cluster"] = kmeans.labels_
    
    # Add BIGB data point to the dataset (annual return and annual volatility)
    bigb_point = pd.DataFrame({"annual_return": [bigb_annual_return], "annual_volatility":[bigb_annual_volatility]})
    bigb_point_scaled = scaler.transform(bigb_point)
    
    # Find the cluster for BIGB using the KMeans model
    bigb_cluster = kmeans.predict(bigb_point_scaled)[0]
    
    return all_assets_data, bigb_cluster
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
# In[ ]:




"""
HOW IS THE DATA BEING SPLIT? - 

The find_bigb_cluster function is used to cluster benchmark_data and bank_etf_data separately using KMeans clustering. bigb_annual_return and bigb_annual_volatility are also passed as arguments to the function, and are used to find the appropriate cluster for the BIGB asset. The function returns the clustered asset data and BIGB's cluster label.

The output of find_bigb_cluster is then assigned to benchmark_data_clustered and bigb_benchmark_cluster for the benchmark data, and bank_etf_data_clustered and bigb_bank_etf_cluster for the bank ETF data.

Finally, the plot_clusters function is called to plot the clusters for both the benchmark data and bank ETF data. The bigb_cluster label is passed as an argument to the function to highlight the cluster that BIGB belongs to in each plot.

"""