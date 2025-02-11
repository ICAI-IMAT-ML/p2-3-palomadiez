# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import *


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # TODO: Train linear regression model with only one coefficient
        meanX = np.mean(X)
        meanY = np.mean(y)
        n = len(X)
        s_xy = (np.sum([X[i]*y[i] for i in range(n)])/n)-(meanX*meanY)
        s_x = (np.sum([X[i]*X[i] for i in range(n)])/n)-(meanX*meanX)

        self.coefficients = s_xy/s_x
        self.intercept = meanY - meanX*self.coefficients

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # TODO: Train linear regression model with multiple coefficients
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])
        w = np.dot(np.linalg.inv((np.dot(np.transpose(X),X))), np.dot(np.transpose(X), y))
        self.intercept = w[0]
        self.coefficients = w[1:]

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.intercept + self.coefficients*X
        else:
            # TODO: Predict when X is more than one variable
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
            w = np.hstack([self.intercept, self.coefficients])
            predictions = np.dot(X,w) 
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    N = len(y_true)
    # R^2 Score
    # TODO: Calculate R^2
    n = len(y_true)
    rss = np.sum([(y_true[i]-y_pred[i])**2 for i in range(n)])
    tss = np.sum([(y_true[i]-np.mean(y_true))**2 for i in range(n)])
    r_squared = 1-(rss/tss)

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt((1/N)*np.sum([(y_true[i]-y_pred[i])**2 for i in range(N)]))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = (1/N)*np.sum([np.abs(y_true[i]-y_pred[i]) for i in range(N)])

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    # Importado arriba

    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    x_reshaped = x.reshape(-1, 1)

    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")
    
    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = ["I", "II", "III", "IV"]

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        data = anscombe[anscombe["dataset"]==dataset]

        # Create a linear regression model
        # TODO
        model = None

        # Fit the model
        # TODO
        X = None  # Predictor, make it 1D for your custom model
        y = None  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred = None

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return results


# Go to the notebook to visualize the results