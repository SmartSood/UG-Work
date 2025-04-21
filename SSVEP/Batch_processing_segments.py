import mne
import pandas as pd
from pathlib import Path
from mne import concatenate_raws
from mne.export import export_raw
import os
import re
import shutil

# Function to delete previous combined_Stimulus directories
def delete_previous_combined_stimulus(root_dir):
    subjects_path = Path(root_dir)
    deleted_folders = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code.group(1) if subject_code else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            combined_stimulus_path = run_folder / "Preprocessed" / f"{subject_code}_combined_Stimulus"

            if combined_stimulus_path.exists():
                try:
                    shutil.rmtree(combined_stimulus_path)
                    deleted_folders.append(str(combined_stimulus_path))
                    print(f"Deleted folder: {combined_stimulus_path}")
                except Exception as e:
                    print(f"Error deleting folder {combined_stimulus_path}: {e}")
            else:
                print(f"No {subject_code}_combined_Stimulus folder found in {run_folder}")

    if deleted_folders:
        print("\nSummary of Deleted Folders:")
        for folder in deleted_folders:
            print(folder)
    else:
        print("\nNo folders were deleted. combined_Stimulus folders may be missing.")

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
def extract_segments_of_stimulus(raw, annotations_df, stimulus):
    segments = []
    segment_times = []
    max_time = raw.times.max()

    # Get indices of annotations corresponding to the stimulus
    stim_indices = annotations_df[annotations_df["Description"] == stimulus].index.tolist()
    print(f"Stimulus: {stimulus}, Markers Found: {len(stim_indices)}")

    while len(stim_indices) > 1:
        start_idx = stim_indices.pop(0)
        end_idx = stim_indices.pop(0)

        start = annotations_df.loc[start_idx, "Onset (s)"]
        end = annotations_df.loc[end_idx, "Onset (s)"]

        if end > max_time:
            print(f"Skipping segment: End time ({end:.4f}s) exceeds max time ({max_time:.4f}s)")
            continue

        # Crop the raw data to obtain an individual segment
        segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=True)
        segments.append(segment)
        segment_times.append({"Start (s)": start, "End (s)": end})
        duration = end - start
        print(f"Segmented {stimulus}: Start = {start:.4f}s, End = {end:.4f}s, Duration = {duration:.2f}s")

    if segments:
        print(f"Extracted {len(segments)} segments for {stimulus}.")
    else:
        print(f"No valid segments found for stimulus: {stimulus}")
        
    return segments, segment_times

# Function to extract baseline
def extract_baseline(raw, annotations_df, baseline_stimulus="Stimulus/S  2"):
    baseline_indices = annotations_df[annotations_df["Description"] == baseline_stimulus].index.tolist()
    if len(baseline_indices) < 2:
        raise ValueError(f"Insufficient markers for {baseline_stimulus} to define a baseline period.")

    start_time = annotations_df.loc[baseline_indices[0], "Onset (s)"]
    end_time = annotations_df.loc[baseline_indices[1], "Onset (s)"]

    return raw.copy().crop(tmin=start_time, tmax=end_time)

# Data Export Function
def save_as_brainvision(raw, output_path, subject_code, stimulus):
    vhdr_path = output_path.parent / f"{subject_code}_S__{stimulus.split()[-1]}_raw.vhdr"
    try:
        export_raw(vhdr_path, raw, fmt='brainvision', overwrite=True)
        print(f"Successfully exported BrainVision files: {vhdr_path}")
    except Exception as e:
        print(f"Error saving BrainVision files: {e}")

# Main Workflow
def process_eeg(file_path, stimuli, output_dir, subject_code):
    raw = load_eeg_data(file_path)
    annotations_df = get_annotations(raw)
    print(f"Total annotations found: {len(annotations_df)}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract and save baseline
    try:
        baseline_raw = extract_baseline(raw, annotations_df)
        baseline_output_path = Path(output_dir) / f"{subject_code}_S__Baseline_raw.fif"
        baseline_raw.save(baseline_output_path, overwrite=True)
        print(f"Successfully saved baseline data as FIF: {baseline_output_path}")
        save_as_brainvision(baseline_raw, baseline_output_path, subject_code, "Baseline")
    except ValueError as e:
        print(e)

    for stim in stimuli:
        segments, segment_times = extract_segments_of_stimulus(raw, annotations_df, stim)
        if segments:
            for idx, segment in enumerate(segments):
                # Create unique file names for each segment
                segment_filename = f"{subject_code}_{stim.replace(' ', '_')}_segment_{idx+1}_raw.fif"
                fif_output_path = Path(output_dir) / segment_filename
                fif_output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save each segment as a FIF file
                segment.save(fif_output_path, overwrite=True)
                print(f"Successfully saved segment {idx+1} as FIF: {fif_output_path}")

                # Also export in BrainVision format (if needed)
                save_as_brainvision(segment, fif_output_path, subject_code, f"{stim}_seg_{idx+1}")
        else:
            print(f"No segments processed for stimulus {stim}.")

# Batch Processing for All Subjects and Runs
def batch_process_all_subjects(root_dir):
    delete_previous_combined_stimulus(root_dir)

    stimuli = ["Stimulus/S  6", "Stimulus/S  7", "Stimulus/S  8", "Stimulus/S  9"]
    subjects_path = Path(root_dir)
    missing_files = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code.group(1) if subject_code else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            segmented_folder = run_folder / "Segmented"
            preprocessed_path = run_folder / "Preprocessed"

            # Find all files ending with SSVEP.vhdr
            ssvep_files = list(segmented_folder.glob("*SSVEP.vhdr"))

            if ssvep_files:
                for ssvep_file in ssvep_files:
                    print(f"Processing {ssvep_file}")
                    process_eeg(
                        file_path=str(ssvep_file),
                        stimuli=stimuli,
                        output_dir=str(preprocessed_path),
                        subject_code=subject_code
                    )
            else:
                missing_files.append(f"Missing SSVEP.vhdr in {run_folder}")

    if missing_files:
        print("\nSummary of Missing SSVEP Files:")
        for missing in missing_files:
            print(missing)
    else:
        print("\nAll subjects and runs contain the required SSVEP files.")

# Example Usage
if __name__ == "__main__":
    root_directory = r"G:\Raw Data\Test_folder_3"
    batch_process_all_subjects(root_directory)
