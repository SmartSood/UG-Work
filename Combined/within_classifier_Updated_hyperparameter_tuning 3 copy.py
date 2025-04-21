import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc, roc_auc_score, precision_score, recall_score)
from sklearn.preprocessing import PowerTransformer, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import joblib

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# 1. Data Loading
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in 32x32 format.
    Returns:
      X: (n_samples, 32, 32)
      y: subject labels (e.g., "01", "02", etc.)
      runs: run labels (e.g., "Run_1", "Run_2")
      epoch_nums: epoch indices
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]  # shape (32,32)
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Basic Data Inspection and Plotting
###############################################################################
def inspect_data_variance(X, y, save_path="data_variance.png"):
    subjects = np.unique(y)
    var_stats = {}
    for subj in subjects:
        X_subj = X[y == subj]
        X_subj_flat = X_subj.reshape(X_subj.shape[0], -1)
        feat_stds = np.std(X_subj_flat, axis=0)
        var_stats[subj] = np.mean(feat_stds)

    logging.info("=== Data Variance Inspection ===")
    for subj in subjects:
        logging.info(f"Subject {subj}: mean STD = {var_stats[subj]:.4f}")

    plt.figure(figsize=(8,4))
    subj_list = list(var_stats.keys())
    mean_std_vals = [var_stats[s] for s in subj_list]
    plt.bar(subj_list, mean_std_vals)
    plt.xlabel("Subject")
    plt.ylabel("Mean STD of Features")
    plt.title("Data Variance per Subject")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, subject_list, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(subject_list))
    plt.xticks(tick_marks, subject_list, rotation=45)
    plt.yticks(tick_marks, subject_list)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregated_roc(scores_matrix, true_labels, subjects, save_path, title="Aggregated ROC Curve"):
    y_true_bin = label_binarize(true_labels, classes=subjects)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"Micro-averaged ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test, save_path, csv_name):
    df_accuracy = pd.DataFrame({
        'subject': true_labels,
        'epoch': epoch_nums_test,
        'correct': (true_labels == np.array(predictions)).astype(int)
    })
    heatmap_data = df_accuracy.pivot(index='subject', columns='epoch', values='correct')
    heatmap_data = heatmap_data.fillna(np.nan)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray')
    plt.xlabel("Epoch Number")
    plt.ylabel("Subject")
    plt.title("Epoch-wise Classification Accuracy (1 = Correct, 0 = Incorrect)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    heatmap_data.to_csv(csv_name)
    logging.info(f"[INFO] Saved epoch-wise classification matrix to {csv_name}")

###############################################################################
# 3. Utility: Evaluation Function for ML Classifiers
###############################################################################
def evaluate_predictions(y_true, predictions, scores_matrix, subjects):
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=subjects)
    return accuracy, f1, cm

###############################################################################
# 4. Across-Run Analysis for ML Classifiers with Hyperparameter Tuning
###############################################################################
def across_run_analysis_ml(classifier, X, y, runs, epoch_nums, classifier_name="Classifier",
                             debug=False, hyperparameter_tuning=False, param_grid=None, cv=3):
    """
    Performs across-run analysis for a scikit-learn based classifier.
    Optionally applies hyperparameter tuning using GridSearchCV.
    - Train on data from Run_1.
    - Test on data from Run_2.
    Returns:
      Dictionary with predictions, true labels, evaluation metrics, and the fitted classifier.
    """
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Please check your run labels.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Hyperparameter tuning step
    if hyperparameter_tuning and param_grid is not None:
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train_flat, y_train)
        classifier = grid_search.best_estimator_
        if debug:
            logging.info(f"[DEBUG] Best hyperparameters for {classifier_name}: {grid_search.best_params_}")

    classifier.fit(X_train_flat, y_train)
    predictions = classifier.predict(X_test_flat)

    subjects_unique = np.unique(y_train)
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_test_flat)
    elif hasattr(classifier, "decision_function"):
        decision_vals = classifier.decision_function(X_test_flat)
        if decision_vals.ndim == 1:
            probs = np.vstack([1 - decision_vals, decision_vals]).T
        else:
            probs = decision_vals
    else:
        probs = np.zeros((X_test_flat.shape[0], len(subjects_unique)))
        for idx, pred in enumerate(predictions):
            probs[idx, list(subjects_unique).index(pred)] = 1

    scores_matrix = probs
    accuracy, f1, cm = evaluate_predictions(y_test, predictions, scores_matrix, subjects_unique)

    results = {
        "classifier_name": classifier_name,
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums_test,
        "scores_matrix": scores_matrix,
        "subjects": subjects_unique,
        "fitted_classifier": classifier  # Return the fitted model here
    }

    if debug:
        logging.info(f"[DEBUG] {classifier_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
    return results

###############################################################################
# 5. Utility Functions for Saving Artifacts and Metadata
###############################################################################
def save_artifact(artifact, filepath, save_method="joblib"):
    """
    Saves an artifact using the specified method with error handling.
    """
    try:
        if save_method == "joblib":
            joblib.dump(artifact, filepath)
        else:
            raise ValueError("Unsupported save method specified.")
        logging.info(f"Saved artifact to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save artifact at {filepath}: {e}")

def save_metadata(model_dir, metadata):
    """
    Saves experiment metadata to a JSON file.
    """
    metadata_filepath = os.path.join(model_dir, "metadata.json")
    try:
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved metadata to {metadata_filepath}")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")

def create_versioned_dir(base_dir="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir

###############################################################################
# 6. Main Script: Run Experiments, Save Comparison CSVs, and Save Models
###############################################################################
def main():
    # Create directories for saving plots and models
    ml_plots_folder = "ml_plots"
    os.makedirs(ml_plots_folder, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create a versioned directory (e.g., models/run_20250403_153000)
    versioned_model_dir = create_versioned_dir()

    # Load data from HDF5 file
    filename = "all_subjects_merged_new_full_epochs.h5"  # Adjust as needed
    X, y, runs, epoch_nums = load_features_from_hdf5(filename)

    # Inspect data variance and save the plot
    inspect_data_variance(X, y, save_path="data_variance.png")

    # Preprocess the data using a PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    # Reshape to (n_samples, 32, 28) as required by the pipeline
    X = X_tf.reshape(X.shape[0], 32, 28)

    # Save the transformer for real-time use
    save_artifact(transformer, os.path.join(versioned_model_dir, "power_transformer.pkl"), save_method="joblib")

    comparison_results = []

    ##############################
    # Train and evaluate scikit-learn models with Hyperparameter Tuning
    ##############################
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],         # Only l2 is supported by 'lbfgs'
        'solver': ['lbfgs']
    }
    logging.info("=== Across-Run Analysis: Logistic Regression with Tuning ===")
    results_lr = across_run_analysis_ml(lr, X, y, runs, epoch_nums,
                                        classifier_name="LogisticRegression",
                                        debug=True, hyperparameter_tuning=True, param_grid=lr_param_grid, cv=3)
    df_results_lr = pd.DataFrame({"True_Label": results_lr["true_labels"],
                                  "Predicted_Label": results_lr["predictions"]})
    lr_csv_name = os.path.join(ml_plots_folder, "ml_results_LogisticRegression.csv")
    df_results_lr.to_csv(lr_csv_name, index=False)
    logging.info(f"[INFO] Logistic Regression results saved to {lr_csv_name}")
    comparison_results.append({
        "Classifier": "LogisticRegression",
        "Accuracy": results_lr["accuracy"],
        "F1_Score": results_lr["f1"]
    })
    # Save the fitted Logistic Regression model
    save_artifact(results_lr["fitted_classifier"], os.path.join(versioned_model_dir, "logistic_regression_model.pkl"), save_method="joblib")
    plot_confusion_matrix(results_lr["confusion_matrix"], results_lr["subjects"],
                          save_path=os.path.join(ml_plots_folder, "confusion_matrix_LogisticRegression.png"),
                          title="Logistic Regression Confusion Matrix")
    plot_aggregated_roc(results_lr["scores_matrix"], results_lr["true_labels"], results_lr["subjects"],
                        save_path=os.path.join(ml_plots_folder, "aggregated_roc_LogisticRegression.png"),
                        title="Logistic Regression ROC Curve")
    lr_epoch_csv = os.path.join(ml_plots_folder, "epoch_accuracy_LogisticRegression.csv")
    plot_epoch_accuracy_heatmap(results_lr["true_labels"], results_lr["predictions"], results_lr["epoch_nums_test"],
                                save_path=os.path.join(ml_plots_folder, "epoch_accuracy_heatmap_LogisticRegression.png"),
                                csv_name=lr_epoch_csv)

    # Random Forest
    rf = RandomForestClassifier()
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    logging.info("=== Across-Run Analysis: Random Forest with Tuning ===")
    results_rf = across_run_analysis_ml(rf, X, y, runs, epoch_nums,
                                        classifier_name="RandomForest",
                                        debug=True, hyperparameter_tuning=True, param_grid=rf_param_grid, cv=3)
    df_results_rf = pd.DataFrame({"True_Label": results_rf["true_labels"],
                                  "Predicted_Label": results_rf["predictions"]})
    rf_csv_name = os.path.join(ml_plots_folder, "ml_results_RandomForest.csv")
    df_results_rf.to_csv(rf_csv_name, index=False)
    logging.info(f"[INFO] Random Forest results saved to {rf_csv_name}")
    comparison_results.append({
        "Classifier": "RandomForest",
        "Accuracy": results_rf["accuracy"],
        "F1_Score": results_rf["f1"]
    })
    save_artifact(results_rf["fitted_classifier"], os.path.join(versioned_model_dir, "random_forest_model.pkl"), save_method="joblib")
    plot_confusion_matrix(results_rf["confusion_matrix"], results_rf["subjects"],
                          save_path=os.path.join(ml_plots_folder, "confusion_matrix_RandomForest.png"),
                          title="Random Forest Confusion Matrix")
    plot_aggregated_roc(results_rf["scores_matrix"], results_rf["true_labels"], results_rf["subjects"],
                        save_path=os.path.join(ml_plots_folder, "aggregated_roc_RandomForest.png"),
                        title="Random Forest ROC Curve")
    rf_epoch_csv = os.path.join(ml_plots_folder, "epoch_accuracy_RandomForest.csv")
    plot_epoch_accuracy_heatmap(results_rf["true_labels"], results_rf["predictions"], results_rf["epoch_nums_test"],
                                save_path=os.path.join(ml_plots_folder, "epoch_accuracy_heatmap_RandomForest.png"),
                                csv_name=rf_epoch_csv)

    # SVM
    svm = SVC(probability=True)
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    logging.info("=== Across-Run Analysis: SVM with Tuning ===")
    results_svm = across_run_analysis_ml(svm, X, y, runs, epoch_nums,
                                         classifier_name="SVM",
                                         debug=True, hyperparameter_tuning=True, param_grid=svm_param_grid, cv=3)
    df_results_svm = pd.DataFrame({"True_Label": results_svm["true_labels"],
                                   "Predicted_Label": results_svm["predictions"]})
    svm_csv_name = os.path.join(ml_plots_folder, "ml_results_SVM.csv")
    df_results_svm.to_csv(svm_csv_name, index=False)
    logging.info(f"[INFO] SVM results saved to {svm_csv_name}")
    comparison_results.append({
        "Classifier": "SVM",
        "Accuracy": results_svm["accuracy"],
        "F1_Score": results_svm["f1"]
    })
    save_artifact(results_svm["fitted_classifier"], os.path.join(versioned_model_dir, "svm_model.pkl"), save_method="joblib")
    plot_confusion_matrix(results_svm["confusion_matrix"], results_svm["subjects"],
                          save_path=os.path.join(ml_plots_folder, "confusion_matrix_SVM.png"),
                          title="SVM Confusion Matrix")
    plot_aggregated_roc(results_svm["scores_matrix"], results_svm["true_labels"], results_svm["subjects"],
                        save_path=os.path.join(ml_plots_folder, "aggregated_roc_SVM.png"),
                        title="SVM ROC Curve")
    svm_epoch_csv = os.path.join(ml_plots_folder, "epoch_accuracy_SVM.csv")
    plot_epoch_accuracy_heatmap(results_svm["true_labels"], results_svm["predictions"], results_svm["epoch_nums_test"],
                                save_path=os.path.join(ml_plots_folder, "epoch_accuracy_heatmap_SVM.png"),
                                csv_name=svm_epoch_csv)

    # Save overall comparison CSV
    df_comparison = pd.DataFrame(comparison_results)
    comparison_csv = os.path.join(ml_plots_folder, "ml_classifiers_comparison.csv")
    df_comparison.to_csv(comparison_csv, index=False)
    logging.info(f"[INFO] Comparison metrics saved to {comparison_csv}")

    # Save experiment metadata for reproducibility
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "LogisticRegression": lr_param_grid,
            "RandomForest": rf_param_grid,
            "SVM": svm_param_grid
        },
        "data_shape": {
            "X": list(X.shape),
            "y": list(y.shape)
        }
    }
    save_metadata(versioned_model_dir, metadata)

if __name__ == "__main__":
    main()
