import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier


def knn(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    plot_decision_boundary: bool = False,
    n_neighbors: int = 3
):

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100

    if plot_decision_boundary and X_train.shape[1] == 2:
        plot_decision_regions(X_train, y_train, knn_model)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend(loc="upper left")
        plt.show()

    return f"Test set accuracy: {accuracy:.2f}%"
