import mne
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import shutil

# Data Loading Functions
def load_eeg_data(file_path):
    try:
        if file_path.endswith('.edf'):
            return mne.io.read_raw_edf(file_path, preload=True)
        elif file_path.endswith('.vhdr'):
            return mne.io.read_raw_brainvision(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading EEG data: {e}")

def get_annotations(raw_data):
    annotations = raw_data.annotations
    return pd.DataFrame({
        "Description": annotations.description,
        "Onset (s)": annotations.onset,
        "Duration (s)": annotations.duration
    })

# Function to extract segments and prepare a summary with marker info
def extract_segments_with_summary(raw, annotations_df):
    segments = []
    segment_summary = {}

    # Identify start indices that contain "5" (fix for marker format issues)
    start_indices = np.where(annotations_df["Description"].str.contains("5", regex=False))[0]
    # Identify indices for end marker "Stimulus/S 14"
    end_indices = np.where(annotations_df["Description"] == "Stimulus/S 14")[0]

    print(f"Found {len(start_indices)} occurrences of 'Stimulus/S 5' (or similar)")
    print(f"Found {len(end_indices)} occurrences of 'Stimulus/S 14'")

    for i, start_idx in enumerate(start_indices):
        start_onset = annotations_df.iloc[start_idx]["Onset (s)"]
        marker_type = "S5"

        valid_end_indices = end_indices[end_indices > start_idx]
        if valid_end_indices.size > 0:
            end_idx = valid_end_indices[0]
            end_onset = annotations_df.iloc[end_idx]["Onset (s)"]

            print(f"Extracting segment {i+1}: Start at {start_onset}s, End at {end_onset}s")

            segment = raw.copy().crop(tmin=start_onset, tmax=end_onset)
            segments.append(segment)

            segment_summary[f"Segment {i + 1}"] = {
                "Start (s)": start_onset,
                "End (s)": end_onset,
                "Marker": marker_type
            }

    return segments, segment_summary




# Function to extract baseline
def extract_baseline(raw, annotations_df, baseline_stimulus="Stimulus/S  2"):
    baseline_indices = annotations_df[annotations_df["Description"] == baseline_stimulus].index.tolist()
    if len(baseline_indices) < 2:
        raise ValueError(f"Insufficient markers for {baseline_stimulus} to define a baseline period.")

    start_time = annotations_df.loc[baseline_indices[0], "Onset (s)"]
    end_time = annotations_df.loc[baseline_indices[1], "Onset (s)"]

    return raw.copy().crop(tmin=start_time, tmax=end_time)

# Data Export Function
def save_as_brainvision(raw, output_dir, subject_code, segment_label):
    output_dir = Path(output_dir)
    vhdr_filename = f"{subject_code}_{segment_label}_raw.vhdr"
    vhdr_path = output_dir / vhdr_filename
    try:
        mne.export.export_raw(vhdr_path, raw, fmt='brainvision', overwrite=True)
        print(f"Successfully exported BrainVision files: {vhdr_path}")
    except Exception as e:
        print(f"Error saving BrainVision files: {e}")

# Main Workflow for processing a single EEG file
def process_eeg(file_path, output_dir, subject_code):
    raw = load_eeg_data(file_path)
    annotations_df = get_annotations(raw)
    print(f"Total annotations found: {len(annotations_df)}")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a subject-specific subfolder named "{subject_code}_Memory" inside output_dir
    subject_output_dir = output_dir / f"{subject_code}_Memory"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Save baseline in the same subject-specific subfolder
    try:
        baseline_raw = extract_baseline(raw, annotations_df)
        baseline_filename = f"{subject_code}_S__Baseline_raw.fif"
        baseline_output_path = subject_output_dir / baseline_filename
        baseline_raw.save(baseline_output_path, overwrite=True)
        print(f"Successfully saved baseline data as FIF: {baseline_output_path}")
        save_as_brainvision(baseline_raw, subject_output_dir, subject_code, "Baseline")
    except ValueError as e:
        print(e)

    # Extract segments using the updated function with summary and marker info
    segments, segment_summary = extract_segments_with_summary(raw, annotations_df)

    # Save each segment with marker info in the file name
    for i, segment in enumerate(segments):
        summary = segment_summary[f"Segment {i + 1}"]
        marker = summary["Marker"]

        segment_filename = f"{subject_code}_{marker}_segment_{i+1}_raw.fif"
        fif_output_path = subject_output_dir / segment_filename

        segment.save(fif_output_path, overwrite=True)
        print(f"Successfully saved segment {i+1} as FIF: {fif_output_path}")

        save_as_brainvision(segment, subject_output_dir, subject_code, f"{marker}_segment_{i+1}")

    print("Segment Summary:")
    for seg, info in segment_summary.items():
        print(f"{seg}: Start {info['Start (s)']}s, End {info['End (s)']}s, Marker: {info['Marker']}")

# Batch Processing
def batch_process_all_subjects(root_dir):
    subjects_path = Path(root_dir)
    missing_files = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code_match = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code_match.group(1) if subject_code_match else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            segmented_folder = run_folder / "Segmented"
            Memory_files = list(segmented_folder.glob("*Memory.vhdr")) + list(segmented_folder.glob("*memory.vhdr"))

            if Memory_files:
                for Memory_file in Memory_files:
                    print(f"Processing {Memory_file}")
                    process_eeg(str(Memory_file), str(run_folder / "Preprocessed"), subject_code)
            else:
                missing_files.append(f"Missing Memory.vhdr in {run_folder}")

def delete_previous_subject_outputs(root_dir):
    subjects_path = Path(root_dir)
    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code = subject_folder.name.split("_")[0]
        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            output_dir = run_folder / "Preprocessed" / f"{subject_code}_Memory"
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"Deleted: {output_dir}")

if __name__ == "__main__":
    root_directory = r"G:\Raw Data\Test_folder_3"
    delete_previous_subject_outputs(root_directory)
    batch_process_all_subjects(root_directory)
