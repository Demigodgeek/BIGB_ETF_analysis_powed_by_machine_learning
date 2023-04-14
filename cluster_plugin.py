  #!/usr/bin/env python
# coding: utf-8

# In[161]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_plugin_for_julio import get_bigb_data, get_benchmark_cluster_data, get_bank_etf_cluster_data

bigb_df, bigb_trading_df, bigb_cluster_df = get_bigb_data()
bank_etf_cluster_df = get_bank_etf_cluster_data()
benchmark_cluster_df = get_benchmark_cluster_data()


# # BIGB Only

# In[162]:


# Remove missing values, replace infinite values, and fill NaN with mean
bigb_df = bigb_df.replace([np.inf, -np.inf], np.nan).fillna(bigb_df.mean())

# Create a list of k values and calculate inertia for each k
inertia = []
k = list(range(1, 11))
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(bigb_df)
    inertia.append(model.inertia_)

# Create elbow plot
bigb_elbow_data = {
    'k': k,
    'inertia': inertia
}
bigb_elbow_data_df = pd.DataFrame(bigb_elbow_data)
bigb_elbow_plot = bigb_elbow_data_df.hvplot.line(
    x='k',
    y='inertia',
    title='BIGB Elbow Curve',
    xticks=k
)

print(bigb_elbow_data_df)
print(inertia)
bigb_elbow_plot = bigb_elbow_plot.opts(width=600, height=400, show_grid=True)

# Save the elbow plot as a PNG file in the "images" directory
file_path = 'images/bigb_elbow_plot.png'
hvplot.save(bigb_elbow_plot, file_path, fmt='png')
bigb_elbow_plot


# # Benchmark Cluster

# In[163]:


# Concatenate BIGB and Benchmark DataFrames into a single DataFrame
benchmark_cluster = pd.concat([bigb_cluster_df, benchmark_cluster_df])

# Preprocess the data
benchmark_cluster = benchmark_cluster.replace([np.inf, -np.inf], np.nan).fillna(benchmark_cluster.mean())

# Apply the KMeans algorithm to find clusters
# Choose the optimal number of clusters 
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
benchmark_cluster['cluster'] = kmeans.fit_predict(benchmark_cluster[['annual_return', 'annual_volatility']])

import hvplot.pandas

# Create a dictionary to map cluster numbers to their labels
cluster_labels = {
    0: 'BIGB',
    1: 'Benchmark'
}

# Map cluster numbers to their labels
benchmark_cluster['cluster_label'] = benchmark_cluster['cluster'].map(cluster_labels)

# Create a scatter plot using hvplot
benchmark_scatter_plot = benchmark_cluster.hvplot.scatter(
    x='annual_return',
    y='annual_volatility',
    by='cluster_label',
    cmap='Category10',
    legend='top_left',
    title='Benchmark',
    xlabel='Annual Return',
    ylabel='Annual Volatility',
    alpha=0.7,
    size=100,
    hover=True
)
file_path = 'images/benchmark_scatter_plot.png'
hvplot.save(benchmark_scatter_plot, file_path, fmt='png')
benchmark_scatter_plot.opts(width=800, height=500, show_grid=True)


# # Bank ETF Cluster

# In[164]:


# Concatenate BIGB and Benchmark DataFrames into a single DataFrame
bank_etf_cluster = pd.concat([bigb_cluster_df, bank_etf_cluster_df])

# Preprocess the data
bank_etf_cluster = bank_etf_cluster.replace([np.inf, -np.inf], np.nan).fillna(bank_etf_cluster.mean())

# Apply the KMeans algorithm to find clusters
# Choose the optimal number of clusters
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
bank_etf_cluster['cluster'] = kmeans.fit_predict(bank_etf_cluster[['annual_return', 'annual_volatility']])

import hvplot.pandas

# Create a dictionary to map cluster numbers to their labels
cluster_labels = {
    0: 'BIGB',
    1: 'Bank_ETF'
}

# Map cluster numbers to their labels
bank_etf_cluster['cluster_label'] = bank_etf_cluster['cluster'].map(cluster_labels)

# Create a scatter plot using hvplot
bank_etf_scatter_plot = bank_etf_cluster.hvplot.scatter(
    x='annual_return',
    y='annual_volatility',
    by='cluster_label',
    cmap='Category10',
    legend='top_left',
    title='Bank ETF',
    xlabel='Annual Return',
    ylabel='Annual Volatility',
    alpha=0.7,
    size=100,
    hover=True
)
file_path = 'images/bank_etf_scatter_plot.png'
hvplot.save(bank_etf_scatter_plot, file_path, fmt='png')
bank_etf_scatter_plot.opts(width=800, height=500, show_grid=True)


# In[ ]:





# In[ ]:




