
import numpy as np

class MyLinearRegression:
    
    def __init__(self, iters=1000):
        self.iters = iters
        self.intercept = None
        self.beta_hat = None


    def fit(self, X, y):

        # We are using the closed-form method to obtain the
        # optimal beta_hat we will use for prediction

        XTX = np.dot(X.T, X)
        xTy = np.dot(X.T, y)

        self.beta_hat = np.dot(np.linalg.inv(XTX), xTy)


    def predict(self, X_test):
        """
        Makes a prediction y given a group of features x
        inputs: x_test - A set of features from a dataset test split
        returns: y_new_pred - A model prediction 
        """

        y_new_pred = np.dot(X_test, self.beta_hat)
        return y_new_pred


    def get_mse(self, y_test, predictions):
        return np.mean((y_test - predictions)**2)
