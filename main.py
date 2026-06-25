from dataSplitter import DataSplitter
from featureExtractor import FeatureExtractor
from pca import PcaTransformer

from KNN import PyTorchKNNClassifier

def main():
    
    #Optimnization
    F_values = [100,200]
    P_values = [5,10]
    K_values = [5,10]

    # Split dataset
    splitter = DataSplitter()
    x_train, x_val, x_test, y_train, y_val, y_test = splitter.splitData()
    
    best_mse = float('inf')
    best_accuracy = 0.0
    best_f1 = 0.0
    
    OutputTxt = "Results.txt"
    with open(OutputTxt, "w") as f:
        f.write(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}\n")

    # Vectorize text data using TF-IDF
    for F_max_features in F_values:
        Extractor = FeatureExtractor(max_features=F_max_features)
        x_train_extracted = Extractor.fit_transform_train(x_train)
        x_val_extracted = Extractor.transform_unseen(x_val)
        for P_components in P_values:
            # Apply PCA for dimensionality reduction
            if(P_components > F_max_features):
                continue
            transformer = PcaTransformer(n_components=P_components)
            x_train_pca = transformer.fit_transform_train(x_train_extracted)
            x_val_pca = transformer.transform_unseen(x_val_extracted)
            for K_neighbors in K_values:
                # Initialize KNN
                knn = PyTorchKNNClassifier(k=K_neighbors)
                knn.fit(x_train_pca, y_train)
                # Evaluate on validation set(Model Gecerleme)
                mse, accuracy, f1 = knn.evaluate(x_val_pca, y_val)

                if mse < best_mse:
                    best_mse = mse
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if f1 > best_f1:
                    best_f1 = f1
                with open(OutputTxt, "a") as f:
                    f.write(f"F: {F_max_features}, P: {P_components}, K: {K_neighbors}\n")
                    f.write(f"MSE: {mse}, Accuracy: {accuracy}, F1: {f1}\n")
                    f.write("--------------------------------------------------\n")
    with open(OutputTxt, "a") as f:
        f.write(f"Lowest MSE: {best_mse}, Highest Accuracy: {best_accuracy}, Highest F1: {best_f1}\n")
if __name__ == "__main__":
    main()
