# import necessary libraries
import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
import holoviews as hv
from bokeh.models import HoverTool
from sklearn.cluster import KMeans
hv.extension('bokeh')

def get_benchmark_cluster(benchmark_cluster_df):
    
    # Initialize the K-Means model
    model = KMeans(n_clusters=4)
    
    # Fit the model
    model.fit(benchmark_cluster_df)
    
    # Predict clusters
    benchmark_classes = model.predict(benchmark_cluster_df)
    
    # Create a copy of the original DataFrame
    benchmark_cluster_df_predictions = benchmark_cluster_df.copy()

    # Create a new column in the DataFrame with the predicted clusters
    benchmark_cluster_df_predictions["benchmark_classes"] = benchmark_classes
    
    # Plotting the 2D-Scatter with x="Annual Income" and y="Spending Score"
    benchmark_cluster_scarter = benchmark_cluster_df_predictions.hvplot.scatter(x="annual_return", y="annual_volatility", by="benchmark_classes")
    
    benchmark_cluster_plot = hv.render(benchmark_cluster_scarter, backend='bokeh')
    
    return benchmark_cluster_plot
     
def get_bank_etf_cluster(bank_etf_cluster_df):
    
    # Initialize the K-Means model
    model = KMeans(n_clusters=4)
    
    # Fit the model
    model.fit(bank_etf_cluster_df)
    # Predict clusters
    bank_etf_classes = model.predict(bank_etf_cluster_df)
    
    # Create a copy of the original DataFrame
    bank_etf_cluster_df_predictions = bank_etf_cluster_df.copy()

    # Create a new column in the DataFrame with the predicted clusters
    bank_etf_cluster_df_predictions["bank_etf_classes"] = bank_etf_classes
    
    # Plotting the 2D-Scatter with x="Annual Income" and y="Spending Score"
    bank_etf_cluster_scarter = bank_etf_cluster_df_predictions.hvplot.scatter(x="annual_return", y="annual_volatility", by="bank_etf_classes")
    
    bank_etf_cluster_plot = hv.render(bank_etf_cluster_scarter, backend='bokeh')
    
    return bank_etf_cluster_plot