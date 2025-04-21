import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, pinv, cond
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             precision_score, recall_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import PowerTransformer, label_binarize

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
def inspect_data_variance(X, y):
    subjects = np.unique(y)
    var_stats = {}
    for subj in subjects:
        X_subj = X[y == subj]
        X_subj_flat = X_subj.reshape(X_subj.shape[0], -1)
        feat_stds = np.std(X_subj_flat, axis=0)
        var_stats[subj] = np.mean(feat_stds)

    print("=== Data Variance Inspection ===")
    for subj in subjects:
        print(f"Subject {subj}: mean STD = {var_stats[subj]:.4f}")

    plt.figure(figsize=(8,4))
    subj_list = list(var_stats.keys())
    mean_std_vals = [var_stats[s] for s in subj_list]
    plt.bar(subj_list, mean_std_vals)
    plt.xlabel("Subject")
    plt.ylabel("Mean STD of Features")
    plt.title("Data Variance per Subject")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, subject_list):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(subject_list))
    plt.xticks(tick_marks, subject_list, rotation=45)
    plt.yticks(tick_marks, subject_list)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_distance_histograms(genuine_distances, imposter_distances):
    plt.figure(figsize=(8,6))
    plt.hist(genuine_distances, bins=30, alpha=0.7, label='Genuine Distances')
    plt.hist(imposter_distances, bins=30, alpha=0.7, label='Imposter Distances')
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Genuine and Imposter Distances")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_far_frr(thresholds, far_array, frr_array, eer):
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, far_array, label="FAR")
    plt.plot(thresholds, frr_array, label="FRR")
    tau_eer = thresholds[np.argmin(np.abs(np.array(far_array) - np.array(frr_array)))]
    plt.axvline(x=tau_eer, color='gray', linestyle='--', label=f"EER Threshold ({tau_eer:.4f})")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title(f"FAR and FRR Curves (EER = {eer*100:.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

###############################################################################
# 3. Covariance, Template, and Prediction Functions
###############################################################################
def build_subject_covariances_flat(X_data, y_data, alpha=1e-6, debug=False):
    """
    For each subject in y_data, compute the mean template and the regularized
    covariance matrix (plus its inverse) for the flattened data X_data.
    """
    subjects = np.unique(y_data)
    templates = {}
    cov_dict = {}
    for idx, subj in enumerate(subjects):
        subj_data = X_data[y_data == subj]
        template = np.mean(subj_data, axis=0)
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = pinv(cov_matrix)
        templates[subj] = template
        cov_dict[subj] = (cov_matrix, inv_cov)
        if debug and idx == 0:
            print(f"[DEBUG] Subject {subj}:")
            print("  Template (first 5):", template[:5])
            print("  Cov diag (first 5):", np.diag(cov_matrix)[:5])
            print("  Condition number:", cond(cov_matrix))
    return templates, cov_dict

def predict_mahalanobis_flat(sample, templates, cov_dict):
    """
    Predict the subject label for the flattened sample based on the smallest
    Mahalanobis distance to the subject templates.
    """
    best_subj, best_dist = None, float('inf')
    # For recording scores for ROC, store negative distances for every subject.
    scores = {}
    for subj, tmpl in templates.items():
        _, inv_cov = cov_dict[subj]
        dist = mahalanobis(sample, tmpl, inv_cov)
        scores[subj] = -dist  # negative distance as the "score"
        if dist < best_dist:
            best_dist = dist
            best_subj = subj
    return best_subj, scores

def compute_biometric_metrics(genuine_distances, imposter_distances, n_thresholds=1000):
    genuine_distances = np.array(genuine_distances)
    imposter_distances = np.array(imposter_distances)
    all_distances = np.concatenate([genuine_distances, imposter_distances])
    min_d, max_d = all_distances.min(), all_distances.max()
    thresholds = np.linspace(min_d, max_d, n_thresholds)
    far_array, frr_array = [], []
    for tau in thresholds:
        fr = np.mean(genuine_distances > tau)
        fa = np.mean(imposter_distances < tau)
        far_array.append(fa)
        frr_array.append(fr)
    far_array = np.array(far_array)
    frr_array = np.array(frr_array)
    diff = np.abs(far_array - frr_array)
    idx_eer = np.argmin(diff)
    eer = (far_array[idx_eer] + frr_array[idx_eer]) / 2.0
    return eer, thresholds, far_array, frr_array

###############################################################################
# 4. Across-Run Analysis: Train on Run_1 and Test on Run_2
###############################################################################
def across_run_analysis(X, y, runs, epoch_nums, alpha=1e-4, debug=False):
    """
    Performs across-run analysis:
      - Train on data from Run_1.
      - Test on data from Run_2.
      - Build subject templates and covariance matrices using training data.
      - Evaluate predictions on test data using Mahalanobis distance.
      - Compute biometric metrics (EER) from genuine and imposter distances.
      - Record negative distances for ROC curve computation.
    """
    # Identify training and testing indices based on run labels
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Please check your run labels.")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]
    
    # Flatten data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Build templates and covariance matrices from training data
    templates, cov_dict = build_subject_covariances_flat(X_train_flat, y_train, alpha=alpha, debug=debug)
    
    if debug:
        example_subj = list(templates.keys())[0]
        print(f"[DEBUG] Example Template for Subject {example_subj}: {templates[example_subj][:5]}")
    
    predictions = []
    genuine_distances = []
    imposter_distances = []
    # For ROC: we will record a scores matrix (n_samples x n_subjects) for test samples.
    scores_matrix = []
    subjects = sorted(templates.keys())
    
    # Evaluate each test sample
    for i, sample in enumerate(X_test_flat):
        true_subj = y_test[i]
        pred_subj, scores = predict_mahalanobis_flat(sample, templates, cov_dict)
        predictions.append(pred_subj)
        # Record scores in the order of subjects
        scores_vector = [scores[subj] for subj in subjects]
        scores_matrix.append(scores_vector)
        
        # Genuine distance (from sample to its own subject's template)
        _, inv_cov_true = cov_dict[true_subj]
        d_genuine = mahalanobis(sample, templates[true_subj], inv_cov_true)
        genuine_distances.append(d_genuine)
        
        # Imposter distances (sample to every other subject's template)
        for subj in templates:
            if subj == true_subj:
                continue
            _, inv_cov_other = cov_dict[subj]
            d_imposter = mahalanobis(sample, templates[subj], inv_cov_other)
            imposter_distances.append(d_imposter)
            
        if debug and i < 5:
            print(f"[DEBUG] Test sample {i} (True: {true_subj}) | Predicted: {pred_subj} | Genuine distance: {d_genuine:.4f}")
    
    scores_matrix = np.array(scores_matrix)  # shape: (n_test_samples, n_subjects)
    
    # Compute overall metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=subjects)
    eer, thresholds, far_array, frr_array = compute_biometric_metrics(genuine_distances, imposter_distances)
    
    results = {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "eer": eer,
        "genuine_distances": genuine_distances,
        "imposter_distances": imposter_distances,
        "predictions": predictions,
        "true_labels": y_test,
        "thresholds": thresholds,
        "far_array": far_array,
        "frr_array": frr_array,
        "epoch_nums_test": epoch_nums_test,
        "scores_matrix": scores_matrix,
        "subjects": subjects  # ordered list of subjects
    }
    return results

###############################################################################
# 5. Per-Patient Metrics Calculation
###############################################################################
def compute_per_patient_metrics(true_labels, predictions, subjects):
    """
    Compute per-subject metrics: accuracy, precision, recall, F1-score, specificity.
    Specificity is computed by treating each subject as positive in a one-vs-all setting.
    """
    metrics = []
    for subj in subjects:
        # Binary indicators: 1 if sample belongs to current subject, else 0.
        y_true_binary = (true_labels == subj).astype(int)
        y_pred_binary = (np.array(predictions) == subj).astype(int)
        
        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1_val = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # For specificity, we need TN and FP.
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.append({
            "Subject": subj,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1_val,
            "Specificity": specificity
        })
    
    return pd.DataFrame(metrics)

###############################################################################
# 6. ROC Curve Computation and Plotting (Aggregated)
###############################################################################
def plot_aggregated_roc(scores_matrix, true_labels, subjects):
    """
    Computes and plots the micro-averaged ROC curve over all subjects.
    - scores_matrix: shape (n_samples, n_subjects) with negative distances as scores.
    - true_labels: array of true labels for test samples.
    - subjects: list of subject labels (in same order as scores_matrix columns).
    """
    # Binarize the true labels (one-vs-rest)
    y_true_bin = label_binarize(true_labels, classes=subjects)
    
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"Micro-averaged ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregated Micro-averaged ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

###############################################################################
# 7. Epoch-wise Classification Accuracy Heatmap
###############################################################################
def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test):
    """
    Plot and save a heatmap of epoch-wise classification accuracy.
    Here, each test sample's epoch is mapped with a binary indicator (1 = correct, 0 = incorrect).
    """
    df_accuracy = pd.DataFrame({
        'subject': true_labels,
        'epoch': epoch_nums_test,
        'correct': (true_labels == np.array(predictions)).astype(int)
    })
    # Pivot the DataFrame to form a matrix with subjects as rows and epochs as columns.
    heatmap_data = df_accuracy.pivot(index='subject', columns='epoch', values='correct')
    # Fill missing epochs with NaN
    heatmap_data = heatmap_data.fillna(np.nan)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray')
    plt.xlabel("Epoch Number")
    plt.ylabel("Subject")
    plt.title("Epoch-wise Classification Accuracy (1 = Correct, 0 = Incorrect)")
    plt.tight_layout()
    plt.show()
    
    # Save to CSV
    heatmap_data.to_csv("Eyes_open_epoch_wise_classification_accuracy.csv")
    print("[INFO] Saved epoch-wise classification matrix to epoch_wise_classification_accuracy.csv")

###############################################################################
# 8. Main Script
###############################################################################
def main():
    # Update the filename as needed.
    filename = "all_subjects_merged_new_full_epochs.h5"
    X, y, runs, epoch_nums = load_features_from_hdf5(filename)
    
    # Inspect data variance
    inspect_data_variance(X, y)
    
    # Optionally apply Yeo-Johnson power transformation to flatten out distributions.
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    X = X_tf.reshape(X.shape[0], 32, 28)
    
    alpha = 1e-4
    debug = True
    
    print("=== Across-Run Analysis: Train on Run_1 and Test on Run_2 ===")
    results = across_run_analysis(X, y, runs, epoch_nums, alpha=alpha, debug=debug)
    
    print("\n=== Overall Metrics ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"EER: {results['eer']*100:.2f}%")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    
    # Plot confusion matrix
    plot_confusion_matrix(results["confusion_matrix"], results["subjects"])
    # Plot histograms of genuine and imposter distances
    plot_distance_histograms(results["genuine_distances"], results["imposter_distances"])
    # Plot FAR and FRR curves
    plot_far_frr(results["thresholds"], results["far_array"], results["frr_array"], results["eer"])
    # Plot aggregated ROC curve
    plot_aggregated_roc(results["scores_matrix"], results["true_labels"], results["subjects"])
    
    # Compute and save per-patient metrics
    df_per_patient = compute_per_patient_metrics(results["true_labels"], results["predictions"], results["subjects"])
    df_per_patient.to_csv("Eyes_open_per_patient_performance_metrics.csv", index=False)
    print("[INFO] Per-patient performance metrics saved to per_patient_performance_metrics.csv")
    
    # Plot epoch-wise classification accuracy heatmap and save CSV
    plot_epoch_accuracy_heatmap(results["true_labels"], results["predictions"], results["epoch_nums_test"])
    
    # Optionally, save overall test predictions to CSV for further analysis
    df_results = pd.DataFrame({
        "True_Label": results["true_labels"],
        "Predicted_Label": results["predictions"]
    })
    df_results.to_csv("Eyes_open_epochwise_across_run_results.csv", index=False)
    print("[INFO] Across-run results saved to across_run_results.csv")

if __name__ == "__main__":
    main()
