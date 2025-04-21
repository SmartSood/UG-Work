import os
import logging
from pathlib import Path

import h5py
import numpy as np
import mne

# ----------------------------------------------------------------------------- #
# Global Configuration
# ----------------------------------------------------------------------------- #
config = {
    "window_size": 2,      # seconds
    "overlap": 0.5,        # fraction of overlap
    "target_fs": 500       # Hz
}

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MODULE 1: DATA LOADING
# =============================================================================
def load_raw_data(file_path):
    """
    Load raw EEG data from a .fif or .vhdr file using MNE.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == '.fif':
        logger.debug("Loading FIF file: %s", file_path)
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif suffix == '.vhdr':
        logger.debug("Loading BrainVision file: %s", file_path)
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported EEG file format: {suffix}")

# =============================================================================
# MODULE 2: EXTRACT RAW EPOCHS
# =============================================================================
def extract_epochs_for_EyesOpen(root_dir, cfg):
    """
    Traverse Subject_/Run folders, locate EyesOpen .vhdr files,
    load raw, resample, trim 1s from beginning/end, and slice into epochs.
    """
    epochs = []
    root = Path(root_dir)
    fs = cfg["target_fs"]
    win_samp = int(cfg["window_size"] * fs)
    step = int(win_samp * (1 - cfg["overlap"]))

    for subj_folder in root.glob("*Subject_*"):
        subj = subj_folder.name.split('_')[0]
        for run_folder in (subj_folder/"Run_1", subj_folder/"Run_2"):
            if not run_folder.exists():
                continue
            run_id = run_folder.name.split('_')[-1]
            eyes_dir = run_folder/"Preprocessed"/f"{subj}_EyesOpen"
            if not eyes_dir.exists():
                continue

            for vhdr in eyes_dir.glob("*.vhdr"):
                try:
                    raw = load_raw_data(vhdr)

                    # ---- NEW: resample to target_fs ----
                    raw.resample(fs, npad='auto')

                    data = raw.get_data()
                except Exception as e:
                    logger.error("Error loading %s: %s", vhdr, e)
                    continue

                # Trim 1s from start and end
                trim = fs
                if data.shape[1] <= 2 * trim:
                    logger.warning("Segment too short after trimming: %s", vhdr)
                    continue
                data = data[:, trim:-trim]

                # Sliding window to create epochs
                ep_idx = 0
                for start in range(0, data.shape[1] - win_samp + 1, step):
                    epoch = data[:, start:start + win_samp]
                    epochs.append({
                        "data":        epoch,
                        "class_label": subj,
                        "epoch":       ep_idx,
                        "run_id":      run_id
                    })
                    ep_idx += 1

    return epochs

# =============================================================================
# MODULE 3: SAVE RAW EPOCHS
# =============================================================================
def save_epochs_by_class(epochs, filename="eeg_epochs_EyesOpen.h5"):
    """
    Save raw epochs to HDF5 under:
      /class_{label}/Run_{run_id}/epoch_{n}/data
    """
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as h5f:
        for ep in epochs:
            grp = (h5f
                   .require_group(f"class_{ep['class_label']}")
                   .require_group(f"Run_{ep['run_id']}"))
            egrp = grp.create_group(f"epoch_{ep['epoch']}")
            egrp.create_dataset("data", data=ep["data"])
            egrp.attrs.update(
                class_label=ep["class_label"],
                epoch=ep["epoch"],
                run_id=ep["run_id"]
            )
    logger.info("Saved %d epochs to %s", len(epochs), filename)

# =============================================================================
# MODULE 4: MAIN
# =============================================================================
def main():
    directory = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"  # Update as needed
    logger.info("Starting raw epoch extraction for EyesOpen (No preprocessing)")
    epochs = extract_epochs_for_EyesOpen(directory, config)
    if not epochs:
        logger.error("No epochs extracted; check file paths and folder structure")
        return
    save_epochs_by_class(epochs, filename="AllSub_EyesOpen_epochs_RAW_unprocessed.h5")
    logger.info("Done!")

if __name__ == "__main__":
    main()
