import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix(confusion_matrix, ax=None):
    """
    Plotting Confusion Matrix

    Parameters: 
        - confusion_matrix (array) :  The confusion matrix to plot
        - ax (matplotlib Axes) : The axes to plot confusion matrix on, uses if statement for if it not provided a new figure is created

    Returns: 
        - None

    """
    if ax is None:
        plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def cross_validation(cv_score, cv_mean, ax=None):
    """
    Plotting cross-validation scores.

    Parameters:
        - cv_score (array): The cross-validation scores.
        - cv_mean (float): The mean cross-validation score.
        - ax (matplotlib Axes): The axes to plot confusion matrix on, uses if statement for if it not provided a new figure is created

    Returns:
        - None

    """
    if ax is None:
        plt.figure(figsize=(6, 6))
    ax.plot(range(1, len(cv_score) + 1), cv_score, marker='o', linestyle='-')
    ax.axhline(y=cv_mean, color='r', linestyle='--',
               label='Mean Cross-validation Mean Score')
    ax.set_title('Trained Model Cross-validation Scores')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.legend()
    print(cv_score)


def performance_metrics(accuracy, precision, recall, f1, ax=None):
    """
    Plotting the performance metrics.

    Parameters:
        - accuracy (float): The accuracy score.
        - precision (float): The precision score.
        - recall (float): The recall score.
        - f1 (float): The F1 score.
        - ax (matplotlib Axes): The axes to plot confusion matrix on, uses if statement for if it not provided a new figure is created

    Returns:
        - None

    """
    if ax is None:
        plt.figure(figsize=(8, 6))
    metrics_value_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    ax.bar(metrics_value_names, metrics_values, color='skyblue')
    ax.set_title('Trained Model Performance Metrics')
    ax.set_ylabel('Score')


def error_metrics(mse, rmse, ax=None):
    """
    Plotting the error metrics.

    Parameters:
        - mse (float): The mean squared error.
        - rmse (float): The root mean squared error.
        - ax (matplotlib Axes): The axes to plot confusion matrix on, uses if statement for if it not provided a new figure is created

    Returns:
        - None

    """
    if ax is None:
        plt.figure(figsize=(6, 6))
    ax.bar(['MSE', 'RMSE'], [mse, rmse], color='blue')
    ax.set_title('Mean Squared Error and Root Mean Squared Error')
    ax.set_ylabel('Error')
