import h5py
import os
import pandas as pd

# Define the input files
input_files = [
    'all_subject_eeg_features_S3_hierarchical.h5',
    'all_subject_eeg_features_S4_hierarchical.h5',
    'raw_all_subject_eeg_features_S5_hierarchical.h5',
    'all_subject_eeg_features_S6_hierarchical.h5',
    'all_subject_eeg_features_S7_hierarchical.h5',
    'all_subject_eeg_features_S8_hierarchical.h5',
    'all_subject_eeg_features_S9_hierarchical.h5',
    'all_subject_eeg_features_S10_hierarchical.h5',
    'all_subject_eeg_features_S11_hierarchical.h5',
    'eyes_close_hierarchical.h5',
    'EyesOpen_with_ICA.h5'    
]

# Define output path
output_path = 'all_subjects_merged_full_epochs.h5'

# Delete existing output if it exists
if os.path.exists(output_path):
    os.remove(output_path)

# Function to merge epoch groups, avoiding overwrites by renaming group
def merge_epochs_groups(src_group, dst_group):
    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            if name not in dst_group:
                dst_group.copy(item, name)
            else:
                # Rename group if it already exists
                i = 1
                new_name = f"{name}_{i}"
                while new_name in dst_group:
                    i += 1
                    new_name = f"{name}_{i}"
                dst_group.copy(item, new_name)

# Merge the files
with h5py.File(output_path, 'w') as f_out:
    for file_path in input_files:
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f_in:
                for class_key in f_in:
                    if class_key not in f_out:
                        f_out.create_group(class_key)
                    class_group_in = f_in[class_key]
                    class_group_out = f_out[class_key]

                    for run_key in class_group_in:
                        if run_key not in class_group_out:
                            class_group_out.create_group(run_key)
                        run_group_in = class_group_in[run_key]
                        run_group_out = class_group_out[run_key]

                        merge_epochs_groups(run_group_in, run_group_out)
        else:
            print(f"⚠️ File not found: {file_path}")

# Optional: summarize the merged file
def summarize_epochs(file_path):
    summary = []
    with h5py.File(file_path, 'r') as f:
        for class_key in f.keys():
            class_group = f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                epoch_count = sum(1 for name in run_group if name.startswith("epoch_"))
                summary.append({
                    "Subject": class_key,
                    "Run": run_key,
                    "Epoch_Count": epoch_count
                })
    return pd.DataFrame(summary)

# Example usage to view summary
summary_df = summarize_epochs(output_path)
print(summary_df)
