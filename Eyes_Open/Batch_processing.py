import mne
import pandas as pd
from pathlib import Path
import os
import re
import shutil

def delete_previous_combined_eyesopen(root_dir):
    """
    Delete any previously created combined EyesOpen folders.
    Looks for folders named as "{subject_code}_combined_EyesOpen" in each run's Preprocessed folder.
    """
    subjects_path = Path(root_dir)
    deleted_folders = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code.group(1) if subject_code else "Unknown"

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            combined_eyesopen_path = run_folder / "Preprocessed" / f"{subject_code}_combined_EyesOpen"

            if combined_eyesopen_path.exists():
                try:
                    shutil.rmtree(combined_eyesopen_path)
                    deleted_folders.append(str(combined_eyesopen_path))
                    print(f"Deleted folder: {combined_eyesopen_path}")
                except Exception as e:
                    print(f"Error deleting folder {combined_eyesopen_path}: {e}")
            else:
                print(f"No {subject_code}_combined_EyesOpen folder found in {run_folder}")

    if deleted_folders:
        print("\nSummary of Deleted Folders:")
        for folder in deleted_folders:
            print(folder)
    else:
        print("\nNo folders were deleted. Combined EyesOpen folders may be missing.")


def load_eeg_data(file_path):
    """
    Load EEG data based on file extension.
    Supports .edf and BrainVision (.vhdr) files.
    """
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
    """
    Retrieve and clean annotations from the raw EEG data.
    Returns a DataFrame with cleaned descriptions, onsets, and durations.
    """
    annotations = raw_data.annotations
    annotations_df = pd.DataFrame({
        "Description": [re.sub(r'\s+', ' ', desc.strip()) for desc in annotations.description],
        "Onset (s)": annotations.onset,
        "Duration (s)": annotations.duration
    })
    print("Unique annotations in the dataset:", annotations_df["Description"].unique())
    return annotations_df


def extract_eyes_open_segment(raw, annotations_df, marker="Stimulus/S 1", default_duration=2.0):
    """
    Extract the Eyes_open segment from raw EEG data.
    
    For the Eyes_open task, we expect exactly two occurrences of the marker 'Stimulus/S 1':
      - The first occurrence marks the start.
      - The second occurrence marks the end.
      
    If the second marker is not found, a default duration is added to the start time.
    """
    # Find indices of annotations that contain the marker "Stimulus/S 1"
    marker_indices = annotations_df[
        annotations_df["Description"].str.contains(marker, case=False, na=False)
    ].index.tolist()

    if len(marker_indices) < 2:
        print(f"Not enough '{marker}' markers found for Eyes_open segmentation.")
        return None, None

    start_time = annotations_df.loc[marker_indices[0], "Onset (s)"]
    end_time = annotations_df.loc[marker_indices[1], "Onset (s)"]
    max_time = raw.times.max()

    if end_time > max_time:
        print(f"Adjusting end time from {end_time:.4f}s to max time {max_time:.4f}s.")
        end_time = max_time

    segment = raw.copy().crop(tmin=start_time, tmax=end_time, include_tmax=True)
    segment_times = {"Start (s)": start_time, "End (s)": end_time}
    print(f"Eyes_open segment: Start = {start_time:.4f}s, End = {end_time:.4f}s")
    return segment, segment_times


def batch_process_all_subjects(root_dir):
    """
    Process all subjects for the Eyes_open task.
    
    For each subject and each run, the code:
      1. Deletes previous combined EyesOpen folders.
      2. Searches for the Eyes_open EEG file in the 'Segmented' folder (expected to be a .vhdr file).
      3. Loads the EEG data, extracts annotations, and segments the Eyes_open period.
      4. Exports the segmented Eyes_open data in two formats:
         - A FIF file (using MNE's save method).
         - BrainVision format (.vhdr, .eeg, .vmrk) using MNE's built-in export method.
         Before exporting the BrainVision files, previously saved files are cleared.
    """
    # Remove any old "combined_EyesOpen" folders
    delete_previous_combined_eyesopen(root_dir)

    # Define the marker for Eyes_open task (matching the annotation in your files)
    eyesopen_marker = "Stimulus/S 1"
    subjects_path = Path(root_dir)
    missing_files = []

    for subject_folder in subjects_path.glob("*Subject_*"):
        subject_code = re.match(r"(\d+)_Subject_", subject_folder.name)
        subject_code = subject_code.group(1) if subject_code else "Unknown"

        # Loop over runs (Run_1 and Run_2)
        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            segmented_folder = run_folder / "Segmented"
            
            # Create a Preprocessed folder for Eyes_open data
            preprocessed_path = run_folder / "Preprocessed" / f"{subject_code}_EyesOpen"
            preprocessed_path.mkdir(parents=True, exist_ok=True)

            # Look for Eyes_open files in the "Segmented" folder.
            # This pattern matches files like "Mrudul_run_1_Eyes_open.vhdr"
            eyesopen_files = list(segmented_folder.glob("*[eE]yes[_][oO]pen.vhdr"))

            if eyesopen_files:
                for eyes_file in eyesopen_files:
                    print(f"Processing Eyes_open file: {eyes_file}")
                    raw = load_eeg_data(str(eyes_file))
                    annotations_df = get_annotations(raw)

                    # Extract the Eyes_open segment between the first two occurrences of the marker
                    segment, segment_times = extract_eyes_open_segment(raw, annotations_df, marker=eyesopen_marker)
                    if segment:
                        # --------------------------
                        # Export as FIF file format
                        # --------------------------
                        fif_filename = f"{subject_code}_EyesOpen_segment_raw.fif"
                        fif_output_path = preprocessed_path / fif_filename
                        segment.save(fif_output_path, overwrite=True)
                        print(f"Successfully saved Eyes_open segment as FIF: {fif_output_path}")

                        # --------------------------
                        # Export in BrainVision format using MNE's export method
                        # --------------------------
                        # Define the base filename (without extension) for BrainVision files
                        base_filename = preprocessed_path / f"{subject_code}_EyesOpen_segment"
                        
                        # Clear any previously saved BrainVision files (.vhdr, .eeg, .vmrk)
                        for ext in ['.vhdr', '.eeg', '.vmrk']:
                            file_path = preprocessed_path / f"{subject_code}_EyesOpen_segment{ext}"
                            if file_path.exists():
                                file_path.unlink()
                                print(f"Cleared previous file: {file_path}")

                        brainvision_filename = str(base_filename) + '.vhdr'
                        print(f"Exporting Eyes_open segment to BrainVision format with base filename: {base_filename}")
                        # Use MNE's built-in export method; this will create .vhdr, .eeg, and .vmrk files.
                        segment.export(brainvision_filename, fmt='brainvision', overwrite=True)
                        print(f"Successfully saved Eyes_open segment in BrainVision format with base: {base_filename}")
            else:
                missing_files.append(f"Missing Eyes_open.vhdr in {run_folder}")

    # Report missing files if any
    if missing_files:
        print("\nSummary of Missing Eyes_open Files:")
        for missing in missing_files:
            print(missing)
    else:
        print("\nAll subjects and runs contain the required Eyes_open files.")


if __name__ == "__main__":
    root_directory = r"G:\Raw Data\Test_folder_3"
    batch_process_all_subjects(root_directory)
