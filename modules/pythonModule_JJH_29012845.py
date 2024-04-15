from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, mean_squared_error, rand_score, v_measure_score
import numpy as np


def train_model(dataset, test_size=0.2, random_state=42, model=LogisticRegression()):
    """
    Trains a logistic regression model on the given dataset.

    Parameters:
    - dataset: pandas DataFrame
        The dataset to train the model on.
    - test_size: float, optional
        The proportion of the dataset to use for testing. Default is 0.2.
    - random_state: int, optional
        The random seed for reproducibility. Default is 42.
    - model: object, optional
        The model to train. Default is LogisticRegression.

    Returns:
    - model: LogisticRegression
        The trained logistic regression model.
    - X_train: pandas DataFrame
        The training features.
    - X_test: pandas DataFrame
        The testing features.
    - y_train: pandas Series
        The training labels.
    - y_test: pandas Series
        The testing labels.
    """
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, dataset, cv_num=5):
    """
    Evaluates the given model on the given test set.

    Parameters:
    - model: sklearn model
        The trained model.
    - X_test: pandas DataFrame
        The testing features.
    - y_test: pandas Series
        The testing labels.

    Returns:
    - dict
        A dictionary of evaluation metrics.
        - Contains the accuracy, precision, recall, F1 score, confusion matrix, cross-validation scores, and cross-validation mean, mean squared error, root mean squared error, rand score, and v measure score.
        - MSE and RMSE are only useful for regression models, but are included for completeness.
        - Rand score and v measure score are only useful for clustering models, but are included for completeness.
    """

    predictions = model.predict(X_test)

    cv_score = cross_val_score(
        model, dataset.iloc[:, :-1], dataset.iloc[:, -1], cv=cv_num)
    cv_mean = cv_score.mean()

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return {"accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "cross_validation": cv_score,
            "cross_validation_mean": cv_mean,
            "mse": mse,
            "rmse": rmse,
            "rand_score": rand_score(y_test, predictions),
            "v_measure_score": v_measure_score(y_test, predictions)}
