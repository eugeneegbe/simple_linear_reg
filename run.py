from linear_regression import MyLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import Data


# Make a call to the Data class to set the data and convert to df
project_data = Data(fetch_california_housing())
df = project_data.get_dataframe()

print('dataset information')

# We remove the target from the x_train 
X_train, X_test, y_train, y_test = project_data.split_data(df.drop(columns=["target"], axis=1),
                                                           df["target"], test_size=0.2)

print('sdataset hape:', df.shape)
print('train:', X_train.shape)
print('test', X_test.shape)
print('--------------------------')
# Apply our custom model to the train set and report the error

lr_model = MyLinearRegression()
weight_bias = lr_model.fit(X_train, y_train)

# predict on the training set
train_predictions = lr_model.predict(X_train)

# Make prediction on our model
test_predictions = lr_model.predict(X_test)


# train and test dataset rmse
lr_train_model_mse = lr_model.get_mse(y_train, train_predictions)

lr_test_model_mse = lr_model.get_mse(y_test, test_predictions)

print('Custom MSE: Train\t Test \n')
print(lr_train_model_mse, lr_test_model_mse)
print('----------------------------------')

# Implementing the Linear regression from scikit learn
sk_lr_model = LinearRegression()

# we run the linear model on our test set
sk_lr_model.fit(X_train, y_train)

sk_train_pred = sk_lr_model.predict(X_train)
sk_test_pred = sk_lr_model.predict(X_test)

sk_train_mse  = (np.mean((y_train - sk_train_pred)**2))**1/2
sk_test_mse  = (np.mean((y_test - sk_test_pred)**2))**1/2

print('Sklearn MSE: Train\t Test')
print(sk_train_mse, sk_test_mse)