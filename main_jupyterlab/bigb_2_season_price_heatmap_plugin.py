# Import the necessary libraries
import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt

# Create a heatmap to check if there is a seasonal pattern in the price change of the Big Bank
def plot_bigb_seasonal_pattern_heatmap(bigb_df):
    
    # Create new columns for heatmap
    bigb_df.index = pd.to_datetime(bigb_df.index)
    quarterly_close_prices = bigb_df.resample('Q').mean()
    quarterly_close_prices['Year'] = quarterly_close_prices.index.year
    quarterly_close_prices['Quarter'] = quarterly_close_prices.index.quarter

    # Plot the heatmap
    price_heatmap = quarterly_close_prices.hvplot.heatmap(
        x="Year",
        y="Quarter",
        C="close",
        cmap="blues",
        width=800,
        height=400,
        colorbar=True,
        title="BIGB Quarterly Close Price per Year Heatmap"
    )
    
    plt.savefig("images/BIGB Quarterly Close Price per Year Heatmap.png")
    
    return price_heatmap

 