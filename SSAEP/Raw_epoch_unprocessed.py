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
    "window_size": 2.0,    # seconds
    "overlap": 0.5,        # fraction
    "fs": 500,             # target sampling frequency
}

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Configuration: %s", config)

# ----------------------------------------------------------------------------- #
# Extract raw multichannel epochs for S11 — NO preprocessing, but with resampling
# ----------------------------------------------------------------------------- #
def extract_epochs_for_S11(directory, fs, window_size, overlap):
    epochs   = []
    win_samp = int(window_size * fs)
    step     = int(win_samp * (1 - overlap))
    root     = Path(directory)
    subjects = list(root.glob("*Subject_*"))

    for subj in subjects:
        subj_code = subj.name.split('_')[0]
        for run in ['Run_1', 'Run_2']:
            run_dir = subj / run / 'Preprocessed' / f"{subj_code}_SSAEP"
            if not run_dir.exists():
                logger.warning("Skipping missing directory: %s", run_dir)
                continue

            pattern   = f"{subj_code}_S11_segment_*_raw.vhdr"
            seg_files = list(run_dir.glob(pattern))
            epoch_idx = 0

            for seg in seg_files:
                try:
                    raw_seg = mne.io.read_raw_brainvision(str(seg), preload=True)

                    # ---- NEW: resample to target fs ----
                    raw_seg.resample(fs, npad='auto')

                    data = raw_seg.get_data()

                    # Trim first and last second
                    trim = int(fs * 1)
                    if data.shape[1] <= 2 * trim:
                        logger.warning("Segment too short after trim: %s", seg.name)
                        continue
                    data = data[:, trim:-trim]

                    # Baseline correct
                    data = data - np.mean(data, axis=1, keepdims=True)

                    # Sliding window segmentation
                    for start in range(0, data.shape[1] - win_samp + 1, step):
                        seg_data = data[:, start:start + win_samp]
                        epochs.append({
                            'data':        seg_data,
                            'class_label': subj_code,
                            'run_id':      run,
                            'epoch':       epoch_idx
                        })
                        epoch_idx += 1

                except Exception as e:
                    logger.error("Error reading %s: %s", seg.name, e)

    return epochs

# ----------------------------------------------------------------------------- #
# Save epochs into HDF5
# ----------------------------------------------------------------------------- #
def save_epochs_by_class(epochs, filename='eeg_epochs_S11.h5'):
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as h5f:
        for item in epochs:
            class_grp = h5f.require_group(f"class_{item['class_label']}")
            run_grp   = class_grp.require_group(item['run_id'])
            ep_name   = f"epoch_{item['epoch']}"
            if ep_name in run_grp:
                del run_grp[ep_name]
            ep_grp = run_grp.create_group(ep_name)
            ep_grp.create_dataset('data', data=item['data'])
    logger.info("Saved %d epochs to %s", len(epochs), filename)

# ----------------------------------------------------------------------------- #
# Main pipeline
# ----------------------------------------------------------------------------- #
def main():
    directory  = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"
    params     = config
    logger.info("Extracting raw epochs for S11...")

    epoch_list = extract_epochs_for_S11(
        directory,
        params['fs'],
        params['window_size'],
        params['overlap']
    )
    if not epoch_list:
        logger.error("No epochs extracted. Check paths and data.")
        return

    save_epochs_by_class(epoch_list, filename='All_sub_eeg_epochs_S11_unprocessed.h5')

if __name__ == '__main__':
    main()
