import os
import joblib
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeEEGPredictor:
    def __init__(self, model_dir):
        """
        Loads saved artifacts (models, transformer) from disk.
        The `model_dir` parameter should point to the versioned directory containing the saved models.
        """
        self.model_dir = model_dir

        try:
            self.transformer = joblib.load(os.path.join(model_dir, "power_transformer.pkl"))
            logging.info("Transformer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading transformer: {e}")
            raise e

        try:
            self.lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
            logging.info("Logistic Regression model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Logistic Regression model: {e}")
            raise e

        try:
            self.rf_model = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
            logging.info("Random Forest model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Random Forest model: {e}")
            raise e

        try:
            self.svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
            logging.info("SVM model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SVM model: {e}")
            raise e

    def load_features_from_hdf5(self, filename):
        X, y, runs, epoch_nums = [], [], [], []
        with h5py.File(filename, 'r') as h5f:
            for class_key in h5f.keys():
                subject_code = class_key.split('_', 1)[-1]
                class_group = h5f[class_key]
                for run_key in class_group.keys():
                    run_group = class_group[run_key]
                    for epoch_key in run_group.keys():
                        epoch_group = run_group[epoch_key]
                        feats = epoch_group['features'][()]
                        X.append(feats)
                        y.append(subject_code)
                        runs.append(run_key)
                        ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                        epoch_nums.append(ep_idx)
        return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

    def preprocess(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        X_transformed = self.transformer.transform(X_flat)
        X_processed = X_transformed.reshape(X.shape[0], 32, 28)
        X_flat_final = X_processed.reshape(X_processed.shape[0], -1)
        return X_flat_final

    def predict_all(self, X):
        X_flat_final = self.preprocess(X)
        predictions = {}
        predictions['LogisticRegression'] = self.lr_model.predict(X_flat_final)
        predictions['RandomForest'] = self.rf_model.predict(X_flat_final)
        predictions['SVM'] = self.svm_model.predict(X_flat_final)
        return predictions

    def evaluate_all(self, y_true, predictions):
        results = {}
        for model_name, preds in predictions.items():
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
            cm = confusion_matrix(y_true, preds)
            results[model_name] = {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}
        return results

    def predict_and_evaluate_from_file(self, filename):
        X, y_loaded, runs, epoch_nums = self.load_features_from_hdf5(filename)
        predictions = self.predict_all(X)
        y_true = np.array(["03"] * len(y_loaded))
        metrics = self.evaluate_all(y_true, predictions)
        return predictions, metrics

if __name__ == "__main__":
    versioned_dir = "models/run_20250404_161351"  # Replace with your chosen version folder
    predictor = RealTimeEEGPredictor(model_dir=versioned_dir)
    
    new_h5_file = "chetan_eyes_close_hierarchical.h5"  # Update as needed
    predictions, metrics = predictor.predict_and_evaluate_from_file(new_h5_file)
    
    for model_name in predictions:
        print(f"--- {model_name} ---")
        print("Predictions:", predictions[model_name])
        print("Evaluation Metrics:", metrics[model_name])
