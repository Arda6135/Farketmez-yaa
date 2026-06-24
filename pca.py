import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class PcaTransformer:
    def __init__(self,n_components=50, random_state=42):

        self.n_components=n_components
        self.random_state=random_state
        self.pca=None

    def fit_transform_train(self, x_train_extracted):

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        
        x_train_pca = self.pca.fit_transform(x_train_extracted)

        return x_train_pca
    
    def transform_unseen(self, x_val_test_extracted):

        x_val_test_pca = self.pca.transform(x_val_test_extracted)

        return x_val_test_pca
        