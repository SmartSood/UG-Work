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
    "overlap": 0.5,        # fraction (50%)
    "target_fs": 500       # Hz
}

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Configuration: %s", config)

# =============================================================================
# MODULE: EPOCH EXTRACTION & SAVING (No Preprocessing, but with Resampling)
# =============================================================================

def save_epochs_for_S4(root_dir, output_file, fs, window_size, overlap):
    """
    Save raw multichannel epochs of *S4* stimulus files to an HDF5 organized by subject and run.
    This version skips filtering/ICA but DOES resample to `fs` so that
    trimming and windowing remain consistent.
    """
    root = Path(root_dir)

    with h5py.File(output_file, 'w') as h5f:
        for subj_path in root.glob('*Subject_*'):
            subj_code = subj_path.name.split('_')[0]
            for run_name in ['Run_1', 'Run_2']:
                proc_path = subj_path / run_name / 'Preprocessed' / f"{subj_code}_Motor"
                if not proc_path.exists():
                    logger.warning("Skipping missing folder: %s", proc_path)
                    continue

                s4_files = list(proc_path.glob(f"{subj_code}_Motor_S4_seg_*_raw.vhdr"))
                if not s4_files:
                    logger.warning("No S4 files for %s %s", subj_code, run_name)
                    continue

                grp_subj = h5f.require_group(f'class_{subj_code}')
                grp_run  = grp_subj.require_group(run_name)
                epoch_ctr = 0

                for raw_file in s4_files:
                    try:
                        raw = mne.io.read_raw_brainvision(str(raw_file), preload=True)

                        # ---- RESAMPLE to target_fs ----
                        raw.resample(fs, npad='auto')
                        sfreq = raw.info['sfreq']  # now equals fs

                        data = raw.get_data()  # shape: (n_channels, n_times)

                        # compute sample counts from up‑to‑date sfreq
                        win_samp = int(window_size * sfreq)
                        step     = int(win_samp * (1 - overlap))
                        trim     = int(1.0 * sfreq)  # trim 1 second

                        # skip if too short
                        if data.shape[1] <= 2 * trim:
                            logger.warning("Too short after trim: %s", raw_file.name)
                            continue

                        # global baseline subtraction after trim
                        data = data[:, trim:-trim]
                        data -= data.mean(axis=1, keepdims=True)

                        # sliding window epochs
                        for start in range(0, data.shape[1] - win_samp + 1, step):
                            epoch = data[:, start:start + win_samp]
                            grp_epoch = grp_run.create_group(f'epoch_{epoch_ctr}')
                            dset = grp_epoch.create_dataset('data', data=epoch)
                            dset.attrs['subject']     = subj_code
                            dset.attrs['run']         = run_name
                            dset.attrs['source_file'] = raw_file.name
                            epoch_ctr += 1

                    except Exception as e:
                        logger.error("Error processing %s: %s", raw_file.name, e)

    logger.info("Epochs saved to %s", output_file)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    data_dir = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"
    out_h5   = "eeg_raw_epochs_S4_unprocessed_resampled.h5"
    fs       = config['target_fs']
    save_epochs_for_S4(data_dir, out_h5, fs, config['window_size'], config['overlap'])

if __name__ == '__main__':
    main()
