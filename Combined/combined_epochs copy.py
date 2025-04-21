import h5py
import os
import pandas as pd

# === Step 1: Setup paths ===
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
    'all_sub_eyes_close_hierarchical.h5',
    'All_sub_EyesOpen_with_ICA.h5'
]

output_path = 'all_subjects_merged_new_full_epochs.h5'

if os.path.exists(output_path):
    os.remove(output_path)

# === Step 2: Merge with Reindexed Epochs ===

# Epoch counters to track the next index per (subject, run)
epoch_counters = {}

def merge_and_reindex_epochs(src_group, dst_group, subject, run):
    key = (subject, run)
    if key not in epoch_counters:
        epoch_counters[key] = 0
    for name in sorted(src_group.keys()):
        item = src_group[name]
        if isinstance(item, h5py.Group) and name.startswith("epoch"):
            new_epoch_name = f"epoch_{epoch_counters[key]}"
            dst_group.copy(item, new_epoch_name)
            epoch_counters[key] += 1

# Perform merging
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

                        merge_and_reindex_epochs(run_group_in, run_group_out, class_key, run_key)

# === Step 3: Summary function ===
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

# === Step 4: Verification function for a specific subject/run ===
def count_epochs_in_input_files(subject, run, files):
    total = 0
    for file_path in files:
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                if subject in f and run in f[subject]:
                    run_group = f[subject][run]
                    count = sum(1 for name in run_group if name.startswith("epoch_"))
                    total += count
    return total

def count_epochs_in_merged(subject, run, file_path):
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            if subject in f and run in f[subject]:
                run_group = f[subject][run]
                return sum(1 for name in run_group if name.startswith("epoch_"))
    return 0

# === Step 5: Set subject/run to check ===
subject_to_check = "class_01"
run_to_check = "Run_1"

expected_epochs = count_epochs_in_input_files(subject_to_check, run_to_check, input_files)
merged_epochs = count_epochs_in_merged(subject_to_check, run_to_check, output_path)

# === Step 6: Report the check ===
print(f"\n📊 Epoch Check for {subject_to_check} / {run_to_check}")
print(f"Expected from input files: {expected_epochs}")
print(f"Found in merged file    : {merged_epochs}")

if expected_epochs == merged_epochs:
    print("✅ All epochs correctly merged and reindexed!")
else:
    print("❌ Mismatch: Some epochs may be missing!")

# === Optional: Show summary table ===
summary_df = summarize_epochs(output_path)
print("\n🔍 Merged Epoch Summary (first few rows):")
print(summary_df.head())
