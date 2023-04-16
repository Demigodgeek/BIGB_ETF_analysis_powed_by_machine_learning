import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from warnings import filterwarnings
filterwarnings("ignore")

# Apply the tuned X, y, and examing period to first machine learning model-SCV, then backtest and evaluate the model. 
def bigb_svc_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    
    # From SVM, instantiate SVC classifier model instance
    bigb_svc_model = svm.SVC()
    # Fit the model to the data using the training data
    bigb_svc_model = bigb_svc_model.fit(X_train_scaled, y_train)
    # Use the testing data to make the model predictions
    bigb_svc_y_pred = bigb_svc_model.predict(X_test_scaled)

    
    # Use a classification report to evaluate the SVC model using the predictions and testing data
    bigb_svc_testing_report = classification_report(y_test, bigb_svc_y_pred)


    # Create a new empty predictions DataFrame.
    bigb_predictions_df_svc = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    bigb_predictions_df_svc['predicted_signal'] = bigb_svc_y_pred
    # Add the actual returns to the DataFrame
    bigb_predictions_df_svc['actual_returns'] = signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    bigb_predictions_df_svc['strategy_returns'] = bigb_predictions_df_svc['actual_returns'] * bigb_predictions_df_svc['predicted_signal']

    
    # Plot the actual returns versus the strategy returns
    plot_actual_vs_strategy_returns_svc = (1 + bigb_predictions_df_svc[["actual_returns", "strategy_returns"]]).cumprod().plot(
        title="BIG BANK - Actual Returns vs Strategy Returns - SVM",
        xlabel="Date",
        ylabel="Cumulative Returns"
    )

    plt.savefig("images/BIG BANK - Actual Returns vs Strategy Returns - SVM.png")
    
    return bigb_svc_y_pred, bigb_predictions_df_svc, bigb_svc_testing_report, plot_actual_vs_strategy_returns_svc


