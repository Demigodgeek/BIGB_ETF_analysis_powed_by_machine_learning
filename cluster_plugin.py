#!/usr/bin/env python
# coding: utf-8

# cluster_plugin.py


"""

The necessary modules are imported, which include pandas, numpy, Holoviews, KMeans and StandardScaler.

Required data for BIGB, benchmark, and bank ETFs is imported from the data_plugin module.

The find_bigb_cluster function is defined, which takes in all_assets_data, bigb_annual_return, bigb_annual_volatility, and n_clusters as inputs. The function preprocesses the all_assets_data by removing missing values, replacing infinite values with NaN, and filling NaN with column mean values. The function also standardizes the all_assets_data, fits the KMeans model, and assigns the cluster number to each asset in the dataset. The function adds BIGB's data point to the dataset and finds the cluster for BIGB using the KMeans model. The function returns the clustered asset data and BIGB's cluster label.

The find_bigb_cluster function is called twice with benchmark_data and bank_etf_data as all_assets_data, and bigb_annual_return and bigb_annual_volatility as the other inputs. 

**** The function returns two sets of variables, benchmark_data_clustered and bigb_benchmark_cluster, and bank_etf_data_clustered and bigb_bank_etf_cluster.****

The plot_clusters function is defined, which takes in data, bigb_cluster, and title as inputs. The function creates a scatter plot of the data, with the colors representing the clusters. The function also adds a BIGB scatter point, BIGB text label, and customizes the plot by adding a title, legend, and axis labels.

The plot_clusters function is called twice with benchmark_data_clustered and bigb_benchmark_cluster, and bank_etf_data_clustered and bigb_bank_etf_cluster, respectively. The function returns a Holoviews plot, which is saved as a PNG file in the "images" folder.



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
