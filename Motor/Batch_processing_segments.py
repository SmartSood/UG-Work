import mne
import pandas as pd
from pathlib import Path
from mne.export import export_raw
import os
import re

# Data Loading Function
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

# Function to extract baseline (unchanged)
def extract_baseline(raw, annotations_df, baseline_stimulus="Stimulus/S  2"):
    baseline_indices = annotations_df[annotations_df["Description"] == baseline_stimulus].index.tolist()
    if len(baseline_indices) < 2:
        raise ValueError(f"Insufficient markers for {baseline_stimulus} to define a baseline period.")
    start_time = annotations_df.loc[baseline_indices[0], "Onset (s)"]
    end_time = annotations_df.loc[baseline_indices[1], "Onset (s)"]
    return raw.copy().crop(tmin=start_time, tmax=end_time)

# Function to extract 7500 msec segments for Stimulus 3 and Stimulus 4
def extract_segments_motor(raw, annotations_df, stimulus_label, segment_duration=7.5):
    segments = []
    # Filter annotations for the given stimulus label
    stimulus_markers = annotations_df[annotations_df["Description"] == stimulus_label]
    for idx, row in stimulus_markers.iterrows():
        start = row["Onset (s)"]
        end = start + segment_duration
        if end > raw.times.max():
            print(f"Skipping segment for {stimulus_label}: End time ({end:.4f}s) exceeds max time ({raw.times.max():.4f}s)")
            continue
        segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=True)
        segments.append((segment, start, end))
        print(f"Segmented {stimulus_label}: Start = {start:.4f}s, End = {end:.4f}s, Duration = {segment_duration:.2f}s")
    return segments

# Data Export Function
def save_as_brainvision(raw, output_path, subject_code, suffix):
    # Construct the BrainVision header file name with the given suffix
    vhdr_path = output_path.parent / f"{subject_code}_Motor_{suffix}_raw.vhdr"
    try:
        export_raw(vhdr_path, raw, fmt='brainvision', overwrite=True)
        print(f"Successfully exported BrainVision file: {vhdr_path}")
    except Exception as e:
        print(f"Error saving BrainVision file: {e}")

# Main Workflow for Processing a Single EEG File
def process_eeg(file_path, output_dir, subject_code):
    raw = load_eeg_data(file_path)
    annotations_df = get_annotations(raw)
    print(f"Total annotations found: {len(annotations_df)}")

    # Ensure output directory exists (Run_folder/Preprocessed/{subject_code}_Motor)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract and save baseline
    try:
        baseline_raw = extract_baseline(raw, annotations_df)
        baseline_fif = output_path / f"{subject_code}_Motor_Baseline_raw.fif"
        baseline_raw.save(baseline_fif, overwrite=True)
        print(f"Successfully saved baseline data as FIF: {baseline_fif}")
        save_as_brainvision(baseline_raw, baseline_fif, subject_code, "Baseline")
    except ValueError as e:
        print(e)

    # Process segments for Stimulus 3 and Stimulus 4
    for stim_label, suffix in [("Stimulus/S  3", "S3"), ("Stimulus/S  4", "S4")]:
        segments = extract_segments_motor(raw, annotations_df, stim_label, segment_duration=7.5)
        if segments:
            for idx, (segment, start, end) in enumerate(segments):
                segment_filename = f"{subject_code}_Motor_{suffix}_segment_{idx+1}_raw.fif"
                segment_fif = output_path / segment_filename
                segment.save(segment_fif, overwrite=True)
                print(f"Successfully saved segment {idx+1} for {stim_label} as FIF: {segment_fif}")
                save_as_brainvision(segment, segment_fif, subject_code, f"{suffix}_seg_{idx+1}")
        else:
            print(f"No segments processed for {stim_label}.")

# Batch Processing for All Subjects and Runs (updated search pattern and output directory)
def batch_process_all_subjects(root_dir):
    subjects_path = Path(root_dir)
    missing_files = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_match = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_match.group(1) if subject_match else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            # Search in the Segmented folder for files ending with motor.vhdr (case insensitive)
            segmented_folder = run_folder / "Segmented"
            motor_files = list(segmented_folder.glob("*[Mm]otor.vhdr"))
            if motor_files:
                # Define output directory: run_folder/Preprocessed/{subject_code}_Motor
                preprocessed_motor_dir = run_folder / "Preprocessed" / f"{subject_code}_Motor"
                for motor_file in motor_files:
                    print(f"Processing {motor_file}")
                    process_eeg(
                        file_path=str(motor_file),
                        output_dir=str(preprocessed_motor_dir),
                        subject_code=subject_code
                    )
            else:
                missing_files.append(f"Missing motor.vhdr in {run_folder}")

    if missing_files:
        print("\nSummary of Missing motor.vhdr Files:")
        for missing in missing_files:
            print(missing)
    else:
        print("\nAll subjects and runs contain the required motor files.")

# Example Usage
if __name__ == "__main__":
    root_directory = r"G:\Raw Data\Test_folder_3"
    batch_process_all_subjects(root_directory)
