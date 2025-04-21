import os
import glob
import logging
from pathlib import Path

import h5py
import numpy as np
import mne
from mne.preprocessing import ICA

# -----------------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------------- #
config = {
    "window_size": 2.0,    # seconds
    "overlap": 0.5,        # fraction
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
# MODULE 1: DATA LOADING & PREPROCESSING
# =============================================================================

def load_raw_data(file_path):
    """
    Load raw EEG data from a .fif or .vhdr file using MNE.
    """
    path = Path(file_path)
    if path.suffix.lower() == '.fif':
        return mne.io.read_raw_fif(str(path), preload=True)
    elif path.suffix.lower() == '.vhdr':
        return mne.io.read_raw_brainvision(str(path), preload=True)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def preprocess_raw_data(raw, l_freq=1.0, h_freq=40.0, target_fs=None, ica_corr_threshold=0.5):
    """
    Preprocess raw EEG: resample, filter, interpolate, and ICA-clean.
    """
    # Resample if needed
    sf = raw.info['sfreq']
    if target_fs and sf > target_fs + 1:
        raw.resample(target_fs, npad='auto')

    # Montage
    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception:
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)

    # Bandpass and notch
    iir = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir,
               phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir,
                     phase='zero', verbose=False)

    # Interpolate bad channels
    if not raw.info['bads']:
        logger.warning("No bad channels marked; skipping interpolation.")
    else:
        raw.interpolate_bads(reset_bads=True, verbose=False)

    # Pick EEG channels only
    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks=eeg_idx)
    logger.info("Picked %d EEG channels.", len(eeg_idx))

    # ICA artifact removal
    ica = ICA(n_components=20, random_state=97, max_iter=1000, verbose=False)
    ica.fit(raw)
    frontal = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.info['ch_names']]
    if frontal:
        sources = ica.get_sources(raw).get_data()
        frontal_data = raw.copy().pick_channels(frontal).get_data()
        excl = []
        for comp in range(sources.shape[0]):
            for fch in range(frontal_data.shape[0]):
                corr = np.corrcoef(sources[comp], frontal_data[fch])[0, 1]
                if abs(corr) >= ica_corr_threshold:
                    excl.append(comp)
                    break
        ica.exclude = list(set(excl))
        logger.info("Excluding ICA components: %s", ica.exclude)
        raw = ica.apply(raw)
    else:
        logger.warning("Frontal channels missing; skipping ICA exclusion.")

    return raw

# =============================================================================
# MODULE 2: EPOCH EXTRACTION & SAVING
# =============================================================================

def save_epochs_for_S4(root_dir, output_file, fs, window_size, overlap):
    """
    Save raw multichannel epochs of *S4* stimulus files to an HDF5 organized by subject and run.
    """
    root = Path(root_dir)
    win_samp = int(window_size * fs)
    step = int(win_samp * (1 - overlap))

    with h5py.File(output_file, 'w') as h5f:
        for subj_path in root.glob('*Subject_*'):
            subj_code = subj_path.name.split('_')[0]
            for run_name in ['Run_1', 'Run_2']:
                proc_path = subj_path / run_name / 'Preprocessed' / f"{subj_code}_Motor"
                if not proc_path.exists():
                    logger.warning("Skipping missing folder: %s", proc_path)
                    continue

                # Preprocess baseline (no saving)
                base_glob = glob.glob(str(proc_path / f"{subj_code}*Baseline*.fif"))
                if base_glob:
                    try:
                        raw_base = mne.io.read_raw_fif(base_glob[0], preload=True)
                        preprocess_raw_data(raw_base, target_fs=fs)
                    except Exception as e:
                        logger.error("Baseline error: %s", e)
                else:
                    logger.warning("No baseline file for %s %s", subj_code, run_name)

                # Stimulus segments
                s4_files = list(proc_path.glob(f"{subj_code}_Motor_S4_seg_*_raw.vhdr"))
                if not s4_files:
                    logger.warning("No S4 files for %s %s", subj_code, run_name)
                    continue

                grp_subj = h5f.require_group(f'class_{subj_code}')
                grp_run = grp_subj.require_group(run_name)

                epoch_ctr = 0
                for raw_file in s4_files:
                    try:
                        raw = mne.io.read_raw_brainvision(str(raw_file), preload=True)
                        raw = preprocess_raw_data(raw, target_fs=fs)
                        data = raw.get_data()  # shape (n_channels, n_times)

                        # Trim first/last second
                        trim = fs
                        if data.shape[1] <= 2 * trim:
                            logger.warning("Too short after trim: %s", raw_file.name)
                            continue
                        data = data[:, trim:-trim]
                        # Global baseline
                        data -= data.mean(axis=1, keepdims=True)

                        # Sliding windows
                        for start in range(0, data.shape[1] - win_samp + 1, step):
                            epoch = data[:, start:start + win_samp]
                            # Create epoch group
                            grp_epoch = grp_run.create_group(f'epoch_{epoch_ctr}')
                            dset = grp_epoch.create_dataset('data', data=epoch)
                            # Metadata
                            dset.attrs['subject'] = subj_code
                            dset.attrs['run'] = run_name
                            dset.attrs['source_file'] = raw_file.name
                            epoch_ctr += 1

                    except Exception as e:
                        logger.error("Error processing %s: %s", raw_file.name, e)

    logger.info("Epochs saved to %s", output_file)

# =============================================================================
# MODULE 3: MAIN EXECUTION
# =============================================================================

def main():
    data_dir = r"G:\Raw Data\Final_new_data"
    out_h5 = "eeg_raw_epochs_S4.h5"
    fs = config['target_fs']
    save_epochs_for_S4(data_dir, out_h5, fs, config['window_size'], config['overlap'])

if __name__ == '__main__':
    main()
