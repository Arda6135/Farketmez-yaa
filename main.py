from dataSplitter import DataSplitter
from featureExtractor import FeatureExtractor
from pca import PcaTransformer

from KNN import PyTorchKNNClassifier

def main():
    
    #Optimnization
    F_values = [2000, 4000, 6000, 8000, 13000, 17000, 23000, 27000]
    P_values = [25,35,45,55,65,75,85,95]
    K_values = [15,19,26,30,34,40,50,63,78,98]

    datasets = ['imdb', 'tweets']
    OutputTxt = "Results.txt"
    
    for dataset in datasets:
        splitter = DataSplitter()
        x_train, x_val, x_test, y_train, y_val, y_test = splitter.splitData(mode=dataset)
        
        best_mse = float('inf')
        best_accuracy = 0.0
        best_f1 = 0.0

        best_F = None
        best_P = None
        best_K = None
        with open(OutputTxt, "w") as f:
            f.write(f"{dataset} dataset:")
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
                    knn = PyTorchKNNClassifier(k=K_neighbors)
                    knn.fit(x_train_pca, y_train)
                    mse, accuracy, f1 = knn.evaluate(x_val_pca, y_val)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_mse = mse
                        best_f1 = f1
                        best_F = F_max_features
                        best_K = K_neighbors
                        best_P = P_components
                    with open(OutputTxt, "a") as f:
                        f.write(f"F: {F_max_features}, P: {P_components}, K: {K_neighbors}\n")
                        f.write(f"MSE: {mse}, Accuracy: {accuracy}, F1: {f1}\n")
                        f.write(f"{'-'*40}\n")
    with open(OutputTxt, "a") as f:
        f.write(f"Best F,P,K values: F:{F_max_features}, P:{P_components}, K:{K_neighbors}\n")
        f.write(f"MSE: {best_mse}, Accuracy: {best_accuracy}, F1: {best_f1}\n")
        f.write(f"{'-'*40}\n")

    #Test
    BestExtractor = FeatureExtractor(max_features=best_F)
    x_train_best_ext = BestExtractor.fit_transform_train(x_train)
    x_test_best_ext = BestExtractor.transform_unseen(x_test)

    BestTransformer = PcaTransformer(n_components=best_P)
    x_train_best_pca = BestTransformer.fit_transform_train(x_train_best_ext)
    x_test_best_pca = BestTransformer.transform_unseen(x_test_best_ext)

    FinalKNN = PyTorchKNNClassifier(k=best_K)
    FinalKNN.fit(x_train_best_pca, y_train)

    test_mse, test_accuracy, test_f1 = FinalKNN.evaluate(x_test_best_pca,y_test)
    with open(OutputTxt, "a") as f:
        f.write(f"TEST MSE      : {test_mse}\n")
        f.write(f"TEST ACCURACY : {test_accuracy}\n")
        f.write(f"TEST F1-SCORE : {test_f1}\n")
        f.write(f"{'='*40}\n\n")
if __name__ == "__main__":
    main()
