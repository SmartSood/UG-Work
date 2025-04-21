import os
import joblib
import h5py
import numpy as np
import logging
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeEEGCNNPredictor:
    def __init__(self, model_dir):
        """
        Loads the 1D CNN model, transformer, and label encoder.
        """
        self.model_dir = model_dir

        try:
            self.transformer = joblib.load(os.path.join(model_dir, "power_transformer.pkl"))
            logging.info("Transformer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading transformer: {e}")
            raise e

        try:
            self.cnn_model = load_model(os.path.join(model_dir, "best_1d_cnn_model.h5"))
            logging.info("1D CNN model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading 1D CNN model: {e}")
            raise e

        try:
            self.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
            logging.info("Label encoder loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading label encoder: {e}")
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
        X_cnn = X_processed.reshape(X_processed.shape[0], -1, 1)
        return X_cnn

    def predict(self, X):
        X_cnn = self.preprocess(X)
        cnn_probs = self.cnn_model.predict(X_cnn)
        cnn_preds = self.le.inverse_transform(np.argmax(cnn_probs, axis=1))
        return cnn_preds

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}

    def predict_and_evaluate_from_file(self, filename):
        X, y_loaded, runs, epoch_nums = self.load_features_from_hdf5(filename)
        y_pred = self.predict(X)
        y_true = np.array(["03"] * len(y_loaded))  # Set your expected label here
        metrics = self.evaluate(y_true, y_pred)
        return y_pred, metrics

if __name__ == "__main__":
    versioned_dir = "CNN_models"  # Update with your model folder path
    predictor = RealTimeEEGCNNPredictor(model_dir=versioned_dir)

    new_h5_file = "chetan_eyes_close_hierarchical.h5"  # Update with your HDF5 input
    predictions, metrics = predictor.predict_and_evaluate_from_file(new_h5_file)

    print("--- 1D CNN Predictions ---")
    print("Predictions:", predictions)
    print("Evaluation Metrics:", metrics)
