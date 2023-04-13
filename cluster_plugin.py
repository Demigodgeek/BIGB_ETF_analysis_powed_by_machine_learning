#!/usr/bin/env python
# coding: utf-8

# cluster_plugin.py


"""

The module imports various necessary Python modules such as pandas, numpy, sklearn.cluster, and holoviews. 
It also imports data from the "data_plugin_for_julio" module.
The main function of the module is "bigb_cluster()", which performs clustering on the benchmark and bank ETF data to find the appropriate cluster for the BIGB stock. 
It first finds the optimal number of clusters for the benchmark and bank ETF data using the elbow method. 
Then, it performs clustering on the benchmark and bank ETF data using the KMeans algorithm with the optimal number of clusters. 
The function also calls the "plot_clusters()" function to visualize the clusters and save the plots as PNG images.
The "optimal_clusters_elbow_method()" function finds the optimal number of clusters for a given dataset using the elbow method. 
The "find_bigb_cluster()" function performs clustering on a given dataset using the KMeans algorithm with a specified number of clusters, 
and finds the appropriate cluster for the BIGB stock using the annual return and annual volatility data.


"""


# cluster_plugin.py

import pandas as pd
import numpy as np
import holoviews as hv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_plugin_for_julio import get_bigb_data, get_benchmark_cluster_data, get_bank_etf_cluster_data

# Load the Holoviews Bokeh extension
hv.extension('bokeh')

bigb_df, bigb_trading_df, bigb_cluster_df = get_bigb_data()
bank_etf_cluster_df = get_bank_etf_cluster_data()
benchmark_cluster_df = get_benchmark_cluster_data()

# Get BIGB annual return and annual volatility from bigb_cluster_df
bigb_annual_return = bigb_cluster_df["bigb_annual_return"].iloc[0]
bigb_annual_volatility = bigb_cluster_df["bigb_annual_volatility"].iloc[0]

def bigb_cluster():
    # Find the optimal number of clusters using the Elbow Method
    benchmark_elbow = optimal_clusters_elbow_method(benchmark_cluster_df)
    bank_etf_elbow = optimal_clusters_elbow_method(bank_etf_cluster_df)

    # You can choose the optimal number of clusters visually from the elbow curve or use an algorithmic approach
    optimal_clusters = 5

    # Perform clustering and find the appropriate cluster for BIGB
    benchmark_data_clustered, bigb_benchmark_cluster = find_bigb_cluster(benchmark_cluster_df, optimal_clusters)
    bank_etf_data_clustered, bigb_bank_etf_cluster = find_bigb_cluster(bank_etf_cluster_df, optimal_clusters)

    # Call the plot_clusters function and save the plots as PNG images
    plot_cluster_bigb_benchmark = plot_clusters(benchmark_data_clustered, bigb_benchmark_cluster, "Benchmark Clusters", "images/benchmark_clusters.png")
    plot_cluster_bigb_bank_etf = plot_clusters(bank_etf_data_clustered, bigb_bank_etf_cluster, "Bank ETF Clusters", "images/bank_etf_clusters.png")

    return plot_cluster_bigb_benchmark, plot_cluster_bigb_bank_etf

# Find the optimal number of clusters using the Elbow Method
def optimal_clusters_elbow_method(df, max_clusters=10):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    sum_of_squared_distances = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_scaled)
        sum_of_squared_distances.append(kmeans.inertia_)

    return sum_of_squared_distances

# Perform clustering and find the appropriate cluster for BIGB
def find_bigb_cluster(df, n_clusters):
    # Standardize the dataframe
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Fit the KMeans model and find the optimal cluster number
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_scaled)

    # Assign the cluster number to each asset in the dataset
    df["cluster"] = kmeans.labels_

    # Add BIGB data point to the dataset (annual return and annual volatility)
    bigb_point = pd.DataFrame({"annual_return": [bigb_annual_return], "annual_volatility": [bigb_annual_volatility]})
    bigb_point_scaled = scaler.transform(bigb_point)

    # Find the cluster for BIGB using the KMeans model
    bigb_cluster = kmeans.predict(bigb_point_scaled)[0]

    return df, bigb_df
