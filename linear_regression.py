
import numpy as np

class MyLinearRegression:
    
    def __init__(self, iters=1000):
        self.iters = iters
        self.intercept = None
        self.last_beta = None


    def fit(self, X, y):

        # We are using the closed-form method to obtain the
        # which permits us to get beta immediately
        # get different values of betas after many iterations

        betas = []
        for i in range(self.iters):
            XTX = np.dot(X.T, X)
            xTy = np.dot(X.T, y)

            beta_hat = np.dot(np.linalg.inv(XTX), xTy)

            betas.append(beta_hat)

        self.last_beta = betas[-1]


    def predict(self, X_test):
        """
        Makes a prediction y given a group of features x
        inputs: x_test - A set of features from a dataset test split
        returns: y_new_pred - A model prediction 
        """

        y_new_pred = np.dot(X_test, self.last_beta)
        return y_new_pred


    def get_mse(self, y_test, predictions):
        return np.mean((y_test - predictions)**2)
