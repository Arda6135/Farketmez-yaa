from dataSplitter import DataSplitter
from featureExtractor import FeatureExtractor
from pca import PcaTransformer

from KNN import PyTorchKNNClassifier

def main():
    
    #Optimnization
    F_values = [2000]
    P_values = [25]
    K_values = [15]

    datasets = ['imdb', 'tweets']
    OutputTxt = "Results.txt"

    with open(OutputTxt, "w") as f:
        f.write("Previous text cleared. \n\n")

    for dataset in datasets:
        splitter = DataSplitter()
        x_train, x_val, x_test, y_train, y_val, y_test = splitter.splitData(mode=dataset)
        
        best_mse = float('inf')
        best_accuracy = 0.0
        best_f1 = 0.0

        best_F = None
        best_P = None
        best_K = None
        with open(OutputTxt, "a") as f:
            f.write(f"{dataset} dataset:\n")
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
            f.write(f"Best F,P,K values: F:{best_F}, P:{best_P}, K:{best_K}\n")
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
        
        cm = FinalKNN.advanced_metrics(x_test_best_pca, y_test)
        
        report = {}
        num_class = len(cm)   
        for i in range(num_class):
            tp = cm[i][i]
            fn = sum(cm[i]) - tp
            fp = sum(cm[row][i] for row in range(num_class)) - tp

            precision = tp/(tp+fp) if(tp+fp) > 0 else 0.0
            recall = tp/(tp+fn) if(tp+fn) > 0 else 0.0
            f1 = 2*(precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            report[i] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "Size": sum(cm[i])
            }
        with open(OutputTxt, "a") as f:
            f.write(f"TEST MSE      : {test_mse}\n")
            f.write(f"TEST ACCURACY : {test_accuracy}\n")
            f.write(f"TEST F1-SCORE : {test_f1}\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"{'Class':<10}{'Precision':<18}{'Recall':<18}{'F1-Score':<12}{'Size':<12}\n")
            label_names = {0: "Negative", 1: "Positive"} if dataset == 'imdb' else {0: "Negative", 1: "Neutral", 2: "Positive"}
            for class_id, metrics in report.items():
                name = label_names.get(class_id, f"Class {class_id}")
                f.write(f"{name:<10}{metrics['precision']:<18.4f}{metrics['recall']:<18.4f}{metrics['f1-score']:<12.4f}{metrics['Size']:<12}\n")
            f.write(f"{'-' *40}\n\n")
            f.write(f"Confusion Matrix:\n\n")
            
            header_str = "          " + "".join([f"{label_names[i]:<12}" for i in range(len(cm))])
            f.write(header_str + "\n")
            
            for i, row in enumerate(cm):
                row_str = f"{label_names[i]:<10}" + "".join([f"{val:<12}" for val in row])
                f.write(row_str + "\n")
                
            f.write(f"\n{'='*75}\n\n")    
if __name__ == "__main__":
    main()
