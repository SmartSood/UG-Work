#!/usr/bin/env python
import os
import joblib
import logging

# Configure logging for more detailed output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ModelHyperparameterInspector:
    def __init__(self, model_dir):
        """
        Loads the machine learning models from the given model directory.
        Expected files:
          - logistic_regression_model.pkl
          - random_forest_model.pkl
          - svm_model.pkl
        """
        self.model_dir = model_dir

        # Load Logistic Regression model
        try:
            self.lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
            logging.info("Logistic Regression model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Logistic Regression model: {e}")
            raise e

        # Load Random Forest model
        try:
            self.rf_model = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
            logging.info("Random Forest model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Random Forest model: {e}")
            raise e

        # Load SVM model
        try:
            self.svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
            logging.info("SVM model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SVM model: {e}")
            raise e

    def print_hyperparameters(self):
        """
        Prints the hyperparameters of each loaded model.
        """
        print("Logistic Regression Hyperparameters:")
        print(self.lr_model.get_params())

        print("\nRandom Forest Hyperparameters:")
        print(self.rf_model.get_params())

        print("\nSVM Hyperparameters:")
        print(self.svm_model.get_params())

if __name__ == "__main__":
    # Update the path to your versioned model directory as needed
    model_directory = "models/run_20250404_161351"
    
    # Create an instance of the inspector and print hyperparameters
    inspector = ModelHyperparameterInspector(model_directory)
    inspector.print_hyperparameters()
