import pandas as pd
import numpy as np
from dataset import DataSetPreparer
from sklearn.model_selection import train_test_split

class DataSplitter:

    def __init__(self):
        self.validation = 0.20
        self.test = 0.10
        self.random_state = 42

        dataPreparer = DataSetPreparer()
        self.x_raw, self.y_raw = dataPreparer.prepareDataSet()

    def splitData(self):
        
        remain_size = self.validation + self.test

        x_train, x_remain, y_train, y_remain = train_test_split(self.x_raw, self.y_raw, test_size=remain_size, random_state=self.random_state)

        test_size = self.test / (self.validation + self.test)

        x_validation, x_test, y_validation, y_test = train_test_split(x_remain, y_remain, test_size=test_size, random_state=self.random_state)

        return x_train, x_validation, x_test, y_train, y_validation, y_test