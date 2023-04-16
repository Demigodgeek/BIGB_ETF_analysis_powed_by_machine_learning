import pandas as pd
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from warnings import filterwarnings
filterwarnings("ignore")

# Apply the tuned X, y, and examing period to third machine learning model-KNN, then backtest and evaluate the model. 

def bigb_knn_bestk_model(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    # Define the range of values to search for k
    bigb_param_grid = {'n_neighbors': np.arange(1, 7)}

    # Create a KNN model
    bigb_knn_model = KNeighborsClassifier()

    # Create a grid search object to find the best value of k using 5-fold cross-validation
    bigb_grid_search = GridSearchCV(bigb_knn_model, bigb_param_grid, cv=5)

    # Fit the grid search object to the training data
    bigb_grid_search.fit(X_train_scaled, y_train)

    # Here you could remove the # to print the best value of k and its corresponding accuracy score
    # print("Best k:", bigb_grid_search.best_params_['n_neighbors'])
    # print("Best accuracy:", bigb_grid_search.best_score_)

    # Use the best value of k to fit the KNN model on the training data
    bigb_knn_model = KNeighborsClassifier(n_neighbors=bigb_grid_search.best_params_['n_neighbors'])
    bigb_knn_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set using the trained model
    bigb_knn_y_pred = bigb_knn_model.predict(X_test_scaled)

    # Use a classification report to evaluate the model using the predictions and testing data
    bigb_knn_testing_report = classification_report(y_test, bigb_knn_y_pred)

    # Create a new empty predictions DataFrame.
    # Create a predictions DataFrame
    bigb_predictions_df_knn = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    bigb_predictions_df_knn['predicted_signal'] = bigb_knn_y_pred
    # Add the actual returns to the DataFrame
    bigb_predictions_df_knn['actual_returns'] = signals_df.loc[y_test.index]['actual_returns']
    # Add the strategy returns to the DataFrame
    bigb_predictions_df_knn['strategy_returns'] = bigb_predictions_df_knn['actual_returns'] * bigb_predictions_df_knn['predicted_signal']

    # Plot the actual returns versus the strategy returns
    plot_bigb_actual_vs_strategy_returns_knn = (1 + bigb_predictions_df_knn[["actual_returns", "strategy_returns"]]).cumprod().plot(
        title="BIG BANK - Actual Returns vs Strategy Returns - KNN",
        xlabel="Date",
        ylabel="Cumulative Returns"
    )
    plt.savefig("images/BIG BANK - Actual Returns vs Strategy Returns - KNN.png")
    return bigb_knn_y_pred, bigb_predictions_df_knn, bigb_knn_testing_report, plot_bigb_actual_vs_strategy_returns_knn