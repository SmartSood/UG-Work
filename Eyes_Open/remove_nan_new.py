import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv, pinv, cond
from sklearn.preprocessing import PowerTransformer

###############################################################################
# 1. Data Loading: Combined EEG and Connectivity Features
###############################################################################
def load_combined_features_from_hdf5(filename, exclude_diagonal=True):
    """
    Loads both EEG features and connectivity features from an HDF5 file.
    
    For each epoch:
      - EEG features are loaded from the 'features' dataset and flattened.
      - Connectivity features are loaded from the 'connectivity' dataset, and only the lower
        triangular elements (excluding the diagonal if desired) are extracted and flattened.
    
    If the connectivity dataset is missing, that epoch is skipped.
    
    Returns:
      feature_matrices: a list of dicts, where each dict contains:
          'features': combined feature vector (1D numpy array)
          'class_label': subject label (str)
          'epoch': epoch index (int)
          'run_id': run label (str)
    """
    feature_matrices = []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():  # e.g., "class_01", "class_02", ...
            # Extract subject code (assumes naming "class_<subject_code>")
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():  # e.g., "Run_1", "Run_2"
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():  # e.g., "epoch_0", "epoch_1", ...
                    epoch_group = run_group[epoch_key]
                    
                    # Ensure connectivity dataset exists
                    if 'connectivity' not in epoch_group:
                        print(f"[WARNING] Connectivity dataset not found for {epoch_key} in {run_key} for subject {subject_code}. Skipping epoch.")
                        continue
                    
                    # Load EEG features and flatten (assumes shape: [n_channels, n_features])
                    eeg_features = epoch_group['features'][()]
                    eeg_flat = eeg_features.flatten()
                    
                    # Load connectivity matrix (assumed symmetric) and extract lower triangular part
                    conn = epoch_group['connectivity'][()]
                    k_val = -1 if exclude_diagonal else 0
                    tril_indices = np.tril_indices(conn.shape[0], k=k_val)
                    conn_flat = conn[tril_indices]
                    
                    # Concatenate EEG and connectivity features
                    combined = np.concatenate([eeg_flat, conn_flat])
                    
                    # Retrieve metadata
                    label = epoch_group.attrs['class_label']
                    epoch_idx = epoch_group.attrs['epoch']
                    run_id = epoch_group.attrs['run']
                    
                    feature_matrices.append({
                        'features': combined,
                        'class_label': label,
                        'epoch': epoch_idx,
                        'run_id': run_id
                    })
    return feature_matrices

###############################################################################
# 2. Remove Near-Constant Features
###############################################################################
def remove_near_constant_features(feature_matrices, var_threshold=1e-10):
    """
    Remove feature columns (across all epochs) that have near-zero variance.
    
    Returns the cleaned feature_matrices and an index list of kept columns.
    """
    # 1) Stack all epochs into one array of shape (n_epochs, n_features)
    all_feats = [fm['features'] for fm in feature_matrices]
    big_array = np.stack(all_feats, axis=0)
    n_epochs, n_features = big_array.shape
    
    # 2) Compute variance over epochs for each feature
    var_vals = np.var(big_array, axis=0)
    
    # 3) Determine which columns to keep
    keep_mask = var_vals > var_threshold
    kept_indices = np.where(keep_mask)[0]
    
    # 4) Remove columns from each epoch's feature vector
    for fm in feature_matrices:
        fm['features'] = fm['features'][keep_mask]
    
    return feature_matrices, kept_indices

###############################################################################
# 3. Save Features by Class (Hierarchical)
###############################################################################
def save_features_by_class(feature_matrices, filename='combined_features_processed.h5'):
    """
    Save feature matrices into an HDF5 file, organized by:
      - Class Label (subject)
      - Run (e.g., 'Run_1' or 'Run_2')
      - Epoch number
    """
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as h5f:
        for data in feature_matrices:
            class_label = data['class_label']
            epoch_number = data['epoch']
            run_id = data.get('run_id', 'Unknown')
            
            class_group = h5f.require_group(f'class_{class_label}')
            run_group = class_group.require_group(f'Run_{run_id}')
            
            # Create a group for the epoch and store the feature vector
            epoch_group = run_group.create_group(f'epoch_{epoch_number}')
            epoch_group.create_dataset('features', data=data['features'])
            epoch_group.attrs['class_label'] = class_label
            epoch_group.attrs['epoch'] = epoch_number
            epoch_group.attrs['run'] = run_id

###############################################################################
# 4. (Optional) Data Inspection / Plotting Functions
###############################################################################
def inspect_data_variance(X, y):
    subjects = np.unique(y)
    var_stats = {}
    for subj in subjects:
        X_subj = X[y == subj]
        feat_stds = np.std(X_subj, axis=0)
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

###############################################################################
# 5. Main Script: Load, Process, and Save Combined Features
###############################################################################
def main():
    # File paths (update as needed)
    raw_hdf5 = "connectivity_all_subject_eeg_features_EyesOpen_with_ICA.h5"  # Input file
    output_hdf5 = "Removed_NAN_combined_features_processed.h5"              # Output file

    # 1) Load combined features (EEG + connectivity)
    feature_matrices = load_combined_features_from_hdf5(raw_hdf5, exclude_diagonal=True)
    print(f"Loaded {len(feature_matrices)} epochs from {raw_hdf5}")

    # 2) Check total number of features before filtering (using first epoch as example)
    n_features_before = feature_matrices[0]['features'].shape[0]
    print(f"Total features per epoch before filtering: {n_features_before}")

    # 3) Remove near-constant features
    feature_matrices, kept_indices = remove_near_constant_features(feature_matrices, var_threshold=1e-10)
    print(f"Kept {len(kept_indices)} features after removing near-constant columns.")
    
    # 4) (Optional) Apply a power transformation if needed
    # Flatten all feature vectors for transformation
    all_feats = np.stack([fm['features'] for fm in feature_matrices], axis=0)
    transformer = PowerTransformer(method='yeo-johnson')
    all_feats_transformed = transformer.fit_transform(all_feats)
    # Update each epoch's feature vector with the transformed data
    for i, fm in enumerate(feature_matrices):
        fm['features'] = all_feats_transformed[i]
    
    # 5) Save processed features to new HDF5 file
    save_features_by_class(feature_matrices, output_hdf5)
    print(f"Features processed and saved to {output_hdf5}")

if __name__ == "__main__":
    main()
