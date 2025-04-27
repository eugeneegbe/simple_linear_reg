
import numpy as np

class MyLinearRegression:
    
    def __init__(self, learning_rate=0.01, iters=6):
        self.learning_rate = learning_rate
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights

        for iteration in range(self.iters):
            y_pred = np.dot(X, self.weights)  # Predictions
            dw = (1/n_samples) * X.T.dot(y_pred - y)  # Gradient
            self.weights -= self.learning_rate * dw  # Update weights


    def predict(self, x_test):
        """
        Makes a prediction y given a group of features x
        inputs: x_test - A set of features from a dataset test split
        returns: y_new_pred - A model prediction 
        """
        y_new_pred = x_test.dot(self.weights)

        return y_new_pred

    def get_mse(self, y_test, predictions):
        return np.mean((y_test - predictions)**2)
