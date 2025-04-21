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
    "window_size": 2,    # seconds
    "overlap": 0.5,      # fraction
    "target_fs": 500     # Hz
}

# Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_and_save_epochs_for_S5(directory, output_file):
    """
    Slides a window over stimulus EEG data, trims 1s at both ends,
    baseline-corrects globally, and saves each epoch to HDF5 organized by class/run/epoch.
    """
    fs = config['target_fs']
    win_samp = int(config['window_size'] * fs)
    step = int(win_samp * (1 - config['overlap']))

    if os.path.exists(output_file):
        os.remove(output_file)

    with h5py.File(output_file, 'w') as h5f:
        subjects = list(Path(directory).glob('*Subject_*'))
        for subj in subjects:
            subject_code = subj.name.split('_')[0]
            for run_folder in (subj / 'Run_1', subj / 'Run_2'):
                if not run_folder.exists():
                    continue

                run_id = run_folder.name.split('_')[-1]
                preproc_folder = run_folder / 'Preprocessed'
                stim_folder = preproc_folder / f'{subject_code}_Memory{subject_code}_Memory'
                files = list(stim_folder.glob(f'{subject_code}_S5_segment_*_raw.vhdr'))

                if not files:
                    logger.warning('No S5 stimulus files for %s, %s', subject_code, run_id)
                    continue

                epoch_counter = 0
                for eeg_file in files:
                    try:
                        raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True)
                        data = raw.get_data()  # shape: [n_channels, n_times]
                    except Exception as e:
                        logger.error('Error loading %s: %s', eeg_file, e)
                        continue

                    # Trim 1s at start/end
                    trim = fs
                    if data.shape[1] < 2 * trim:
                        continue
                    data = data[:, trim:-trim]

                    # Global baseline correction
                    ch_mean = data.mean(axis=1, keepdims=True)
                    data = data - ch_mean

                    # Sliding windows
                    n_samples = data.shape[1]
                    for start in range(0, n_samples - win_samp + 1, step):
                        epoch = data[:, start:start + win_samp]
                        grp = h5f.require_group(f'class_{subject_code}/Run_{run_id}/epoch_{epoch_counter}')
                        grp.create_dataset('data', data=epoch, dtype='float64')
                        grp.attrs['class_label'] = subject_code
                        grp.attrs['run'] = run_id
                        grp.attrs['epoch'] = epoch_counter
                        epoch_counter += 1

                logger.info('Saved %d epochs for %s Run_%s', epoch_counter, subject_code, run_id)

    logger.info('All epochs saved to %s', output_file)


def main():
    data_dir = r"D:\Smarth_work\Final_new_data - Copy"  # Update as needed
    out_file = 'raw_S5_epochs.h5'
    extract_and_save_epochs_for_S5(data_dir, out_file)


if __name__ == '__main__':
    main()
