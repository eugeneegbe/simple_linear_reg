
import numpy as np

class MyLinearRegression:
    
    def __init__(self, learning_rate, iters=10):
        self.learning_rate = learning_rate
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # get the number of features from the training set 
        # initialize our wights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iters):
            # predicting for all samples at once as a matrice
            # np.dot gives us a summation already
            y_pred = np.dot(X, self.weights) + self.bias

            # we compute the change in the bias and
            db = (1/n_samples) * np.sum(y_pred - y)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.learning_rate * dw

            # update the weight and bias
            self.bias = self.bias = (self.learning_rate * db)

    def predict(self, x_test):
        """
        Makes a prediction y given a group of features x
        inputs: x_test - A set of features from a dataset test split
        returns: y_new_pred - A model prediction 
        """
        y_new_pred = np.dot(x_test, self.weights) + self.bias

        return y_new_pred

    def get_mse(self, y_test, predictions):
        return np.mean((y_test - predictions)**2)
