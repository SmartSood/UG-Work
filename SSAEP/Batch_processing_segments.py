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
    """
    Extracts EEG segments starting at stimulus 10 or 11 and ending at stimulus 15,
    and prepares a summary of their time points along with the marker type (S10 or S11).
    
    Parameters:
    - raw: mne.io.Raw object
        The raw EEG data.
    - annotations_df: pd.DataFrame
        DataFrame containing annotation descriptions and onsets.
    
    Returns:
    - segments: list of mne.io.Raw objects
        List of extracted EEG segments.
    - segment_summary: dict
        Dictionary containing segment details with absolute start and end times and marker type.
    """
    segments = []
    segment_summary = {}

    # Identify start indices that contain "Stimulus/S 10" or "Stimulus/S 11"
    start_indices = np.where(annotations_df["Description"].str.contains("Stimulus/S 10|Stimulus/S 11"))[0]
    # Identify indices for end marker "Stimulus/S 15"
    end_indices = np.where(annotations_df["Description"] == "Stimulus/S 15")[0]

    # Find pairs of start and end indices
    for i, start_idx in enumerate(start_indices):
        start_onset = annotations_df.iloc[start_idx]["Onset (s)"]
        start_desc = annotations_df.iloc[start_idx]["Description"]
        # Determine marker type based on start description
        if "10" in start_desc:
            marker_type = "S10"
        elif "11" in start_desc:
            marker_type = "S11"
        else:
            marker_type = "Unknown"

        # Find the nearest end index greater than the start index
        valid_end_indices = end_indices[end_indices > start_idx]
        if valid_end_indices.size > 0:
            end_idx = valid_end_indices[0]
            end_onset = annotations_df.iloc[end_idx]["Onset (s)"]

            # Extract the segment from raw data
            segment = raw.copy().crop(tmin=start_onset, tmax=end_onset)
            segments.append(segment)

            # Add details to summary including the marker type
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
    """
    Exports the raw EEG data in BrainVision format.
    
    Parameters:
    - raw: mne.io.Raw object
    - output_dir: Path or str
         Directory where the BrainVision file will be saved.
    - subject_code: str
         The subject code to include in the file name.
    - segment_label: str
         A label for the segment (e.g., "Baseline", "S10_segment_1").
    """
    output_dir = Path(output_dir)  # ensure output_dir is a Path
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

    # Create a subject-specific subfolder named "{subject_code}_SSAEP" inside output_dir
    subject_output_dir = output_dir / f"{subject_code}_SSAEP"
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

    # Extract segments using the new function with summary and marker info
    segments, segment_summary = extract_segments_with_summary(raw, annotations_df)

    # Save each segment with marker info in the file name
    for i, segment in enumerate(segments):
        summary = segment_summary[f"Segment {i + 1}"]
        marker = summary["Marker"]  # This will be either "S10" or "S11"

        # Create a file name that includes the marker type
        segment_filename = f"{subject_code}_{marker}_segment_{i+1}_raw.fif"
        fif_output_path = subject_output_dir / segment_filename

        # Save the segment in FIF format
        segment.save(fif_output_path, overwrite=True)
        print(f"Successfully saved segment {i+1} as FIF: {fif_output_path}")

        # Export in BrainVision format with the marker in the file name
        save_as_brainvision(segment, subject_output_dir, subject_code, f"{marker}_segment_{i+1}")

    # Optionally, print the segment summary
    print("Segment Summary:")
    for seg, info in segment_summary.items():
        print(f"{seg}: Start {info['Start (s)']}s, End {info['End (s)']}s, Marker: {info['Marker']}")

# Batch Processing for All Subjects and Runs
def batch_process_all_subjects(root_dir):
    

    subjects_path = Path(root_dir)
    missing_files = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code_match = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code_match.group(1) if subject_code_match else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            segmented_folder = run_folder / "Segmented"
            preprocessed_path = run_folder / "Preprocessed"

            # Find all files ending with SSAEP.vhdr
            SSAEP_files = list(segmented_folder.glob("*SSAEP.vhdr"))

            if SSAEP_files:
                for SSAEP_file in SSAEP_files:
                    print(f"Processing {SSAEP_file}")
                    process_eeg(
                        file_path=str(SSAEP_file),
                        output_dir=str(preprocessed_path),
                        subject_code=subject_code
                    )
            else:
                missing_files.append(f"Missing SSAEP.vhdr in {run_folder}")

    if missing_files:
        print("\nSummary of Missing SSAEP Files:")
        for missing in missing_files:
            print(missing)
    else:
        print("\nAll subjects and runs contain the required SSAEP files.")

def delete_previous_subject_outputs(root_dir):
    """
    Deletes subject-specific output folders (e.g., folders ending with _SSAEP)
    in all Preprocessed directories within the given root directory.
    """
    subjects_path = Path(root_dir)
    deleted_folders = []

    # Iterate over all subject folders
    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code_match = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code_match.group(1) if subject_code_match else "Unknown"

        # Check both Run_1 and Run_2 directories
        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            preprocessed_dir = run_folder / "Preprocessed"
            subject_output_dir = preprocessed_dir / f"{subject_code}_SSAEP"
            if subject_output_dir.exists():
                try:
                    shutil.rmtree(subject_output_dir)
                    deleted_folders.append(str(subject_output_dir))
                    print(f"Deleted subject output folder: {subject_output_dir}")
                except Exception as e:
                    print(f"Error deleting folder {subject_output_dir}: {e}")
            else:
                print(f"No subject output folder {subject_code}_SSAEP found in {run_folder}")

    if deleted_folders:
        print("\nSummary of Deleted Subject Output Folders:")
        for folder in deleted_folders:
            print(folder)
    else:
        print("\nNo subject output folders were deleted.")

# Example Usage
if __name__ == "__main__":
    root_directory = r"G:\Raw Data\Test_folder_3"
    # Delete previous outputs (subject-specific folders) first:
    delete_previous_subject_outputs(root_directory)
    # Then run your batch processing
    batch_process_all_subjects(root_directory)
