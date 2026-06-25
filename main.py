from dataSplitter import DataSplitter
from featureExtractor import FeatureExtractor
from pca import PcaTransformer

from KNN import PyTorchKNNClassifier

def main():
    
    #Optimnization
    F_max_features = 2000
    P_components = 50
    K_neighbors = 5

    # Split dataset
    splitter = DataSplitter()
    x_train, x_val, x_test, y_train, y_val, y_test = splitter.splitData()
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    # Vectorize text data using TF-IDF
    Extractor = FeatureExtractor(max_features=F_max_features)
    x_train_extracted = Extractor.fit_transform_train(x_train)
    x_val_extracted = Extractor.transform_unseen(x_val)

    # Apply PCA for dimensionality reduction
    transformer = PcaTransformer(n_components=P_components)
    x_train_pca = transformer.fit_transform_train(x_train_extracted)
    x_val_pca = transformer.transform_unseen(x_val_extracted)

    # Initialize KNN
    knn = PyTorchKNNClassifier(k=K_neighbors)
    knn.fit(x_train_pca, y_train)

    # Evaluate on validation set(Model Gecerleme)
    mse, accuracy, f1 = knn.evaluate(x_val_pca, y_val)

    print("Feature Vector Dimension (F) : ", F_max_features)
    print("PCA Output Vector Dim (P)    : ", P_components)
    print("KNN Neighbors (K)            : ", K_neighbors)
    print("Mean Squared Error (MSE)     : ", mse)
    print("Accuracy   : ", accuracy)
    print("F1-Score   : ", f1)

if __name__ == "__main__":
    main()
