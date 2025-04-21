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
# -----------------------------------------------------------------------------
config = {
    "window_size": 2,    # seconds
    "overlap": 0.5,      # fraction
    "target_fs": 500     # Hz
}

# Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_baseline_file(preprocessed_path, subject_code):
    """
    Finds the correct baseline file in the given directory based on naming patterns.
    """
    patterns = [
        f"{subject_code}_Baseline_raw.fif",
        f"{subject_code}_S__Baseline_raw.fif",
        f"{subject_code}_S_Baseline_raw.fif"
    ]
    for p in patterns:
        fpath = Path(preprocessed_path) / p
        if fpath.exists():
            return str(fpath)
    candidates = glob.glob(str(Path(preprocessed_path) / f"{subject_code}*Baseline*.fif"))
    return candidates[0] if candidates else None


def preprocess_raw_data(raw, l_freq=1.0, h_freq=40.0, target_fs=500):
    """
    Preprocess raw EEG: resample, montage, filter, notch, interpolate, ICA.
    """
    # Resample
    if raw.info['sfreq'] > target_fs + 1:
        raw.resample(target_fs, npad='auto')
    # Montage
    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception:
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    # Bandpass + notch
    iir = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq, h_freq, method='iir', iir_params=iir, phase='zero', verbose=False)
    raw.notch_filter(50, method='iir', iir_params=iir, phase='zero', verbose=False)
    # Interpolate bads
    raw.interpolate_bads(reset_bads=True, verbose=False)
    # Pick EEG channels
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks)
    # ICA to remove eye artifacts
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)
    # correlate with frontal channels
    frontal = [ch for ch in ('Fp1','Fp2') if ch in raw.info['ch_names']]
    if frontal:
        src = ica.get_sources(raw).get_data()
        fr = raw.copy().pick_channels(frontal).get_data()
        excl = []
        for comp in range(src.shape[0]):
            for idx, ch in enumerate(frontal):
                corr = np.corrcoef(src[comp], fr[idx])[0,1]
                if abs(corr) >= 0.5:
                    excl.append(comp)
                    break
        ica.exclude = list(set(excl))
    raw = ica.apply(raw)
    return raw


def extract_and_save_epochs_for_S5(directory, output_file):
    """
    Slides a window over stimulus EEG data, trims and baseline-corrects,
    then saves each multichannel epoch to HDF5 organized by class/run/epoch.
    """
    fs = config['target_fs']
    win_samp = int(config['window_size'] * fs)
    step = int(win_samp * (1 - config['overlap']))

    # Remove existing file
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
                                # Stimulus segments are under the Preprocessed folder
                preproc_folder = run_folder / 'Preprocessed'
                stim_folder = preproc_folder / f'{subject_code}_Memory''{subject_code}_Memory'
                files = list(stim_folder.glob(f'{subject_code}_S5_segment_*_raw.vhdr'))
                if not files:
                    logger.warning('No S5 stimulus files for %s, %s', subject_code, run_id)
                    continue

                epoch_counter = 0
                for eeg_file in files:
                    try:
                        raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True)
                        raw = preprocess_raw_data(raw)
                        data = raw.get_data()  # shape: [n_channels, n_times]
                    except Exception as e:
                        logger.error('Error loading %s: %s', eeg_file, e)
                        continue

                    # Trim 1s at start/end
                    trim = fs
                    if data.shape[1] < 2*trim:
                        continue
                    data = data[:, trim:-trim]

                    # Global baseline correction
                    ch_mean = data.mean(axis=1, keepdims=True)
                    data = data - ch_mean

                    # Sliding windows
                    n_samples = data.shape[1]
                    for start in range(0, n_samples - win_samp + 1, step):
                        epoch = data[:, start:start+win_samp]
                        grp = h5f.require_group(f'class_{subject_code}/Run_{run_id}/epoch_{epoch_counter}')
                        grp.create_dataset('data', data=epoch, dtype='float64')
                        grp.attrs['class_label'] = subject_code
                        grp.attrs['run'] = run_id
                        grp.attrs['epoch'] = epoch_counter
                        epoch_counter += 1

                logger.info('Saved %d epochs for %s Run_%s', epoch_counter, subject_code, run_id)

    logger.info('All epochs saved to %s', output_file)


def main():
    data_dir = r"G:\Raw Data\Final_new_data"  # update as needed
    out_file = 'raw_S5_epochs.h5'
    extract_and_save_epochs_for_S5(data_dir, out_file)


if __name__ == '__main__':
    main()
