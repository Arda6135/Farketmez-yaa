import pandas as pd
import numpy as np
from dataset import DataSetPreparer
from sklearn.model_selection import train_test_split

class DataSplitter:

    def __init__(self, validation = 0.15, test = 0.15):
        self.validation = validation
        self.test = test
        self.random_state = 42

        self.preparer = DataSetPreparer()

    def splitData(self, mode = 'both'):

        x, y = self.preparer.prepareDataSet(mode)
        
        remain_size = self.validation + self.test

        x_train, x_remain, y_train, y_remain = train_test_split(x, y, test_size=remain_size, random_state=self.random_state, stratify=self.y_raw)

        test_size = self.test / (self.validation + self.test)

        x_val, x_test, y_val, y_test = train_test_split(x_remain, y_remain, test_size=test_size, random_state=self.random_state, stratify=y_remain)

        return x_train, x_val, x_test, y_train, y_val, y_test