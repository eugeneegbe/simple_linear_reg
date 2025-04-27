import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class Data:

    def __init__(self, dataset):
        self.dataset = dataset
 
    def get_dataframe(self):
        """
        Construct a dataframe from the dataset on an instance
        """

        ds = self.dataset
        df = pd.DataFrame(data= np.c_[ds['data'], ds['target']],
                     columns= ds['feature_names'] + ['target'])
        
        return df


    def split_data(self, dataframe, y_column, test_size):
        """
        Splits a dataframe into train, test and validation sets
        Input: A pandas dataframe, y_column 
        Output, train set and test set splits
        """

        # Note that we don't include the column y
        X_train, X_test, y_train, y_test = train_test_split(dataframe, y_column, test_size=test_size)
        return X_train, X_test, y_train, y_test


