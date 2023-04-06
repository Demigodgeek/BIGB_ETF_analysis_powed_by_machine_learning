# Import Tools and libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from prophet import Prophet

# Fetch data using API.
# Data listed below is for example only.
data = {
    'Ticker': ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5'],
    'Annual_Return': [0.1, 0.11, 0.22, 0.33, 0.44],
    'Annual_Volalitlity': [0.23, 0.43, 0.35, 0.97, o.2],
    'ETF_Segment': ['Tech', 'Finance', 'Healthcare', 'Energy', 'Market']
}

# Create a DataFrame from the example data 
df = pd.DataFrame(data)
df.head()