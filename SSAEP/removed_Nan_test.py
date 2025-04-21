import h5py
import numpy as np

def load_feature_matrices_from_hdf5(filename):
    """
    Load feature matrices from a hierarchical HDF5 file.
    
    Returns a list of dicts, where each dict has:
      'features': (n_channels, n_features) array
      'class_label': str
      'epoch': int
      'run_id': str
    """
    feature_matrices = []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():  # e.g., "class_01", "class_02", ...
            class_group = h5f[class_key]
            for run_key in class_group.keys():  # e.g., "Run_1", "Run_2"
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():  # e.g., "epoch_0", "epoch_1", ...
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]  # (n_channels, n_features)
                    
                    # Retrieve metadata
                    label = epoch_group.attrs['class_label']
                    epoch_idx = epoch_group.attrs['epoch']
                    run_id = epoch_group.attrs['run']
                    
                    feature_matrices.append({
                        'features': feats,
                        'class_label': label,
                        'epoch': epoch_idx,
                        'run_id': run_id
                    })
    return feature_matrices


def remove_near_constant_features(feature_matrices, var_threshold=1e-10):
    """
    Remove feature columns (across all channels) that have near-zero variance.
    Returns the cleaned feature_matrices and an index list of kept columns.
    """
    # 1) Stack all epochs to shape (n_epochs, n_channels, n_features)
    all_feats = [fm['features'] for fm in feature_matrices]
    big_array = np.stack(all_feats, axis=0)
    n_epochs, n_channels, n_features = big_array.shape
    
    # 2) Compute variance across (n_epochs, n_channels) for each feature dimension
    #    We'll flatten n_epochs x n_channels into one dimension => shape (n_epochs * n_channels, n_features)
    flattened = big_array.reshape(n_epochs * n_channels, n_features)
    var_vals = np.var(flattened, axis=0)
    
    # 3) Determine which columns to keep
    keep_mask = var_vals > var_threshold
    kept_indices = np.where(keep_mask)[0]
    
    # 4) Remove columns from each epoch
    for fm in feature_matrices:
        fm['features'] = fm['features'][:, keep_mask]
    
    return feature_matrices, kept_indices


def save_features_by_class(feature_matrices, filename='eeg_features_S6_normalized.h5'):
    """
    Save feature matrices into an HDF5 file, organized by class_label, run_id, epoch.
    """
    import os

    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as h5f:
        for data in feature_matrices:
            class_label = data['class_label']
            epoch_number = data['epoch']
            run_id = data.get('run_id', 'Unknown')
            
            class_group = h5f.require_group(f'class_{class_label}')
            run_group = class_group.require_group(f'Run_{run_id}')
            
            epoch_group = run_group.create_group(f'epoch_{epoch_number}')
            epoch_group.create_dataset('features', data=data['features'])
            epoch_group.attrs['class_label'] = class_label
            epoch_group.attrs['epoch'] = epoch_number
            epoch_group.attrs['run'] = run_id
            

def main():
    raw_hdf5 = "all_subject_eeg_features_S11_hierarchical.h5"      # Raw features file
    output_hdf5 = "Removed_NAN_all_subject_eeg_features_S11_hierarchical.h5"  # Output file

    # 1) Load raw features
    feature_matrices = load_feature_matrices_from_hdf5(raw_hdf5)
    print(f"Loaded {len(feature_matrices)} epochs from {raw_hdf5}")

    # 2) Get total number of features before filtering (assume first epoch represents all)
    n_features_before = feature_matrices[0]['features'].shape[1]

    # 3) Remove near-constant features
    feature_matrices, kept_indices = remove_near_constant_features(feature_matrices, var_threshold=1e-10)
    print(f"Kept {len(kept_indices)} features after removing near-constant columns.")

    # 4) Compute dropped indices
    all_indices = np.arange(n_features_before)
    dropped_indices = np.setdiff1d(all_indices, kept_indices)

    print(f"Dropped {len(dropped_indices)} features (indices): {dropped_indices.tolist()}")

    # 5) Save (optional, if needed in your pipeline)
    save_features_by_class(feature_matrices, output_hdf5)

    print(f"Features processed and saved to {output_hdf5}")


if __name__ == "__main__":
    main()


