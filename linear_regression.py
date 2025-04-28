
import numpy as np

class MyLinearRegression:
    
    def __init__(self, lr = 0.01, iters=1000):
        self.iters = iters
        self.weight = 0
        self.bias = 0
        self.loss = 0
        self.lr = lr

    def fit(self, X, y):
        n = len(X)

        for i in range(self.iters):
            y_pred = self.predict(X)
            loss = self.get_mse(y, y_pred)

            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            # We are using gradient descent method
            self.weight = self.weight - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            print(y_pred)
            self.loss = loss


    def predict(self, X):
        y_new_pred = (self.weight * X) + self.bias 
        return y_new_pred


    def get_mse(self, y, y_pred):
        return np.mean((y - y_pred)**2)
