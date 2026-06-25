import torch
from collections import Counter
import numpy as np
from scipy.sparse import issparse

class PyTorchKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    def _to_tensor(self, data, is_label=False):
        if issparse(data):
            data = data.toarray()
        elif isinstance(data, np.ndarray):
            pass
        elif hasattr(data, 'to_numpy'):
            data = data.to_numpy()

        if is_label:
            return torch.tensor(data, dtype=torch.long)
        return torch.tensor(data, dtype=torch.float32)
    
    def fit(self, X_train, y_train):
        self.X_train = self._to_tensor(X_train)
        self.y_train = self._to_tensor(y_train, is_label=True)

    def predict(self, X_test):
        X_test_tensor = self._to_tensor(X_test)
        predictions = []

        for test_row in X_test_tensor:
            distances = torch.sqrt(torch.sum((self.X_train - test_row) ** 2, dim=1))
            topk_indices = torch.topk(distances, self.k, largest=False).indices
            k_nearest_labels = self.y_train[topk_indices].tolist()
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
            
        return np.array(predictions)

    def evaluate(self, X_eval, y_eval):
        y_true = self._to_tensor(y_eval, is_label=True)
        y_pred = self.predict(X_eval)

        correct = torch.sum(y_true == torch.tensor(y_pred)).item()
        accuracy = correct / len(y_true)

        mse = torch.mean((y_true.float() - torch.tensor(y_pred).float()) ** 2).item()

        Labels = [0, 1, 2]
        f1_scores = []

        for c in Labels:
            tp = torch.sum((y_true == c) & (torch.tensor(y_pred) == c)).item()
            fp = torch.sum((y_true != c) & (torch.tensor(y_pred) == c)).item()
            fn = torch.sum((y_true == c) & (torch.tensor(y_pred) != c)).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1_score)
        macro_f1 = sum(f1_scores) / len(Labels)

        return mse, accuracy, macro_f1    

