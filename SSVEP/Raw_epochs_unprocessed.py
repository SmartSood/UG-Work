import os
import glob
import logging
from pathlib import Path

import h5py
import numpy as np
import mne

# ----------------------------------------------------------------------------- #
# Global Configuration
# ----------------------------------------------------------------------------- #
config = {
    "target_fs": 500,     # Desired sampling frequency
    "window_size": 2.0,   # Epoch window size (seconds)
    "overlap": 0.5        # Fractional overlap
}

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MODULE: RAW DATA LOADING
# =============================================================================
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fif':
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif file_path.suffix.lower() == '.vhdr':
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported format: {file_path}")

# =============================================================================
# MODULE: EPOCH EXTRACTION FOR S9
# =============================================================================
def extract_epochs_for_S9(directory, config):
    """
    For each subject/run, load segmented stimulus files,
    resample, trim 1s from start/end, baseline correct, and slice into epochs.
    """
    epochs_list = []
    fs         = config['target_fs']
    win_samp   = int(config['window_size'] * fs)
    step       = int(win_samp * (1 - config['overlap']))

    root     = Path(directory)
    subjects = list(root.glob('*Subject_*'))

    for subj in subjects:
        subject_code = subj.name.split('_')[0]
        for run in ['Run_1', 'Run_2']:
            run_path = subj / run / 'Preprocessed'
            if not run_path.exists():
                continue

            stim_folder = run_path / f"{subject_code}_Stimulus"
            pattern     = f"{subject_code}_S__9_seg_*_raw.vhdr"
            files       = list(stim_folder.glob(pattern))
            counter     = 0

            for f in files:
                try:
                    raw = load_raw_data(f)
                except Exception as e:
                    logger.error(f"Error loading {f}: {e}")
                    continue

                # ---- NEW: resample to target_fs ----
                raw.resample(fs, npad='auto')

                data = raw.get_data()
                trim = int(fs * 1)
                if data.shape[1] < 2 * trim:
                    continue

                data = data[:, trim:-trim]
                data = data - np.mean(data, axis=1, keepdims=True)

                for start in range(0, data.shape[1] - win_samp + 1, step):
                    epoch = data[:, start:start + win_samp]
                    epochs_list.append({
                        'data':        epoch,
                        'class_label': subject_code,
                        'epoch':       counter,
                        'run_id':      run.split('_')[-1]
                    })
                    counter += 1

    return epochs_list

# =============================================================================
# MODULE: SAVE EPOCHS
# =============================================================================
def save_epochs_by_class(epochs, filename='raw_epochs_S9.h5'):
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as h5f:
        for e in epochs:
            grp      = h5f.require_group(f"class_{e['class_label']}")
            run_grp  = grp.require_group(f"Run_{e['run_id']}")
            ep_grp   = run_grp.create_group(f"epoch_{e['epoch']}")
            ep_grp.create_dataset('data', data=e['data'])
    logger.info(f"Saved {len(epochs)} epochs to {filename}")

# =============================================================================
# MODULE: MAIN
# =============================================================================
def main():
    data_dir = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"  # adjust as needed
    logger.info("Extracting raw epochs without preprocessing...")
    epochs = extract_epochs_for_S9(data_dir, config)
    if not epochs:
        logger.error("No epochs extracted. Check paths and files.")
        return
    logger.info(f"Extracted {len(epochs)} epochs.")
    save_epochs_by_class(epochs, 'All_sub_raw_epochs_S9_unprocessed.h5')

if __name__ == '__main__':
    main()
