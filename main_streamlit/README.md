# Understanding BIGB with Machine Learing

---

## Instruction

## libraries you should install

+ streamlit
+ yahoo_fin
+ pandas
+ numpy
+ hvplot
+ matplotlib
+ holoviews
+ bokeh
+ tabulate
+ scikit-learn

### For Jupyter Lab 
1. Download the BIGB_analysis_with_machine_learning folder and save it to your desktop.
2. Open your terminal or Git Bash and navigate to the BIGB_analysis_with_machine_learning folder using the command "cd desktop/BIGB_analysis_with_machine_learning/main_jupyterlab"
3. Create a new conda environment for the project by entering the command "conda create -n trading python=3.x.xx"(your pythong version) and answering "y" to all prompts.
4. Activate the new environment by entering the command "conda activate trading".
5. Install all the required libraries by entering the command "pip install -r requirements.txt";
   Or you can install these libraries by running the following command in your terminal or Git Bash:
   `pip install streamlit yahoo_fin pandas numpy hvplot matplotlib holoviews bokeh tabulate scikit-learn`
   Note that you need to activate the virtual environment "trading" first before installing these libraries. 
   Also, you need to ensure that you have the correct version of Python installed. 
   (Should be `3.7+`, Code was tested on versions`3.8.10`& `3.9.16`)
6. After installation, you can run the _main.py file in the main_jupyterlab folder using the command "python _main.py".
7. The program will start running and you will be prompted to enter the investment period and the initial capital.
8. After entering, the program will gather the portfolio data, perform technical analysis, and generate trading signals, and compare metrics with those of S&P500
9. All results using the interactive charts and tables that will be displayed.
10. The program will also output conclusions describing the performance of the trading strategy using different machine learning models.

### For Streamlit
1. Download the BIGB_analysis_with_machine_learning folder and save it to your desktop.
2. Open your terminal or Git Bash and navigate to the BIGB_analysis_with_machine_learning folder using the command "cd desktop/BIGB_analysis_with_machine_learning/main_streamlit".
3. Create a new conda environment for the project by entering the command "conda create -n trading python=3.x.xx"(your python version) and answering "y" to all prompts.
4. Activate the new environment by entering the command "conda activate trading".
5. Install all the required libraries by entering the command "pip install -r requirements.txt";
   Or you can install these libraries by running the following command in your terminal or Git Bash:
   `pip install streamlit yahoo_fin pandas numpy hvplot matplotlib holoviews bokeh tabulate scikit-learn`
   Note that you need to activate the virtual environment "trading" first before installing these libraries. 
   Also, you need to ensure that you have the correct version of Python installed. 
   (Should be `3.7+`, Code was tested on versions`3.8.10`& `3.9.16`)
6. After installation, run the Streamlit app by navigating to the main_streamlit folder and entering the command "streamlit run _main.py".
7. The program will start running and you will be prompted to choose the investment period and the initial capital.
8. Then you can choose one or more check boxes from "Reward/Risk Comparison", "Seasonal Pattern in Price", "Visualization of Short/Long Position Strategy", "Compare Strategy Return with 3 Models", "Compare Strategy Return with S&P500"
8. After checking, the program will gather the portfolio data, perform technical analysis, and generate trading signals, and compare metrics with those of S&P500.
9. All results using interactive charts and tables will be displayed on the Streamlit app.
10. The program will also output conclusions describing the performance of the trading strategy using different machine learning models.

**Note: This program uses historical stock data to generate trading signals and is for educational purposes only. It is not intended to be used as a trading tool for real-world investments. Please consult with a financial advisor before making any investment decisions.

---

## About the Project

### Project Object 
* The objective of this project is to leverage the power of machine learning to help users better understand the BIG Bank portfolio.


### Project strcucture
* The reserch of this project is divided into three main parts:

1. Unsupervised machine learning using clustering to classify the BIGB.
2. Time series analysis to check the seasonal pattern of portfolio's price
3. Supervised machine learning using SVM/KNN/RANDOM FOREST with trading algorithm bot to predict the performance of BIGB, and then compare the metrics to those of S&P 500

---

## Files Navigation

The main branch of the repository has 2 foleders and 4 seperate files

### `main_jupyterlab` folder:
There are 2 folders, 13 modules in the `main_jupyterlab` folder
+ `_main.ipynb`: Main code, run to get all plots or reports.
+ `csv` : Saved bigb's ohlcv data, which is a .csv file
+ `images`: saved plots for future analysis
+ `_preparation_data_plugin` : prepare data for cluster, trading algorithms and time series analysis using 
+ `_questionnaire_plugin` : get user input which are inisial capital and holding period
+ `_x_y_tune_widget_plugin` : here you can tune the window, examing period to find a better value.
+ `bigb_1_cluster_plugin` : cluster and plot to see where the bigb fall on the benchamark's pool and bank industry etf's pool.
+ `bigb_2_season_price_heatmap_plugin` : plot to see whether there is a seasanal pattern in price
+ `bigb_3_visualize_short_long_position_strategy_plugin` : plog to see the short/long position strategy
+ `bigb_4_1_svc_model_plugin` : apply SVC to bigb to generate the trading signal and predict.
+ `bigb_4_2_rf_model_plugin` : apply RANDOM FOREST to bigb to generate the trading signal and predict.
+ `bigb_4_3_knn_bestk_model_plugin` : apply KNN to bigb to generate the trading signal and predict.
+ `sp500_5_knn_bestk_model_plugin` : apply KNN to SP500 to generate the trading signal and predict.
+ `sp500_bigb_6_metric_plugin import bigb_sp500_metric_table` : compare the performance to that of benchmark


### `streamlit_deployment` folder:
There are 11 modules in the main_code
+ `_main.py`: Main code, run to prompt the streamlit interface
+ `_preparation_data_plugin` : prepare data for cluster, trading algorithms and time series analysis using 
+ `_x_y_tune_widget_plugin` : here you can tune the window, examing period to find a better value.
+ `bigb_1_cluster_plugin` : cluster and plot to see where the bigb fall on the benchamark's pool and bank industry etf's pool.
+ `bigb_2_season_price_heatmap_plugin` : plot to see whether there is a seasanal pattern in price
+ `bigb_3_visualize_short_long_position_strategy_plugin` : plog to see the short/long position strategy
+ `bigb_4_1_svc_model_plugin` : apply SVC to bigb to generate the trading signal and predict.
+ `bigb_4_2_rf_model_plugin` : apply RANDOM FOREST to bigb to generate the trading signal and predict.
+ `bigb_4_3_knn_bestk_model_plugin` : apply KNN to bigb to generate the trading signal and predict.
+ `sp500_5_knn_bestk_model_plugin` : apply KNN to SP500 to generate the trading signal and predict.
+ `sp500_bigb_6_metric_plugin import bigb_sp500_metric_table` : compare the performance to that of benchmark

    
### Other Files:
> `project_proposal.doc`
> `presentation.pdf`
> `README.md`
> `requirements.txt`
> `.gitignore`

---

## Limitation

+ In this project, we utilize a  data set based on the performance of the six stocks included in the BIGB portfolio and use Yahoo Finance as the primary data source for this project. The data didn't consider the dividend and thus affect the model performance.

+ We didn't have enough time to test if there is linear relationships between all features and target, so we only choose models which are good at handling non-linear data.

+ Although we set a separate module for tune the windows or train/test dataset, we got no enough time to find out a better one with higher accuracy, more overlay line plot and higher strategy return.

+ Although we didn't observe an obviously seasonal pattern of the price, we still tried Prophet, Sarima and Arima, none got a good result, and for some Technichal reasons, the output is pretty long and abnormal.

---

## Summary

+ According to the cluster results, the BIGB has a higher annual return and relatively moderate annual volatility comparing to the index ETFs or other bank industry ETFs.

+ From the heatmap we can see that there is no obviously seasonal pattern in term of price of Big Bank.

+ All three 3 models need to be tune or optimaze, but the KNN with best k has a higher accuracy, and higher annual return.

+ Compare 5 important metrics of Big Bank to those of S&P 500, the BigB is a goode choice for investors who have a relatively high tolarence of risk.

---

## Contributors

* Demi
* Julio
* Cary

---

## License
This program is licensed under the MIT License.
