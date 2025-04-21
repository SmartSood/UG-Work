import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the data
file_path = r"G:\Softwares\Results_S8\S8_no_norm_across_run_results.csv"  # Change the path if needed
data = pd.read_csv(file_path)

# Step 2: Extract labels
y_true = data['True_Label']
y_pred = data['Predicted_Label']

# Step 3: Compute Multiclass Performance Metrics
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)

# Macro and Weighted metrics
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1 = report['weighted avg']['f1-score']

# Combine into a dictionary
multiclass_metrics = {
    'Accuracy': accuracy,
    'Macro Precision': macro_precision,
    'Macro Recall': macro_recall,
    'Macro F1 Score': macro_f1,
    'Weighted Precision': weighted_precision,
    'Weighted Recall': weighted_recall,
    'Weighted F1 Score': weighted_f1,
}

# Convert to DataFrame for saving
multiclass_metrics_df = pd.DataFrame.from_dict(multiclass_metrics, orient='index', columns=['Value'])

# Step 4: FAR, FRR, and EER Calculation per class (One-vs-All)
unique_classes = y_true.unique()
far_frr_eer_per_class = {}

for cls in unique_classes:
    binary_true = (y_true == cls).astype(int)
    binary_pred = (y_pred == cls).astype(int)

    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()

    # Calculate FAR, FRR, EER
    far = fp / (fp + tn) if (fp + tn) != 0 else 0
    frr = fn / (fn + tp) if (fn + tp) != 0 else 0
    eer = (far + frr) / 2

    far_frr_eer_per_class[cls] = {
        'FAR': far,
        'FRR': frr,
        'EER': eer
    }

# Convert to DataFrame
far_frr_eer_df = pd.DataFrame.from_dict(far_frr_eer_per_class, orient='index')
far_frr_eer_df.index.name = 'Class'

# Step 5: Save both metrics to files
multiclass_metrics_df.to_csv(r'G:\Softwares\Results_S8\S8_Multiclass_Performance_Metrics.csv', index=True)
far_frr_eer_df.to_csv(r'G:\Softwares\Results_S8\S8_FAR_FRR_EER_per_class.csv', index=True)

print("Saved 'Multiclass_Performance_Metrics.csv' and 'FAR_FRR_EER_per_class.csv'")
