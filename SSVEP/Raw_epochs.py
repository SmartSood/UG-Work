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
    "target_fs": 500,     # Desired sampling frequency
    "l_freq": 1.0,        # High-pass filter
    "h_freq": 40.0,       # Low-pass filter
    "ica_corr_threshold": 0.5,  # Threshold for artifact ICA removal
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
# MODULE 1: PREPROCESSING
# =============================================================================
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fif':
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif file_path.suffix.lower() == '.vhdr':
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported format: {file_path}")


def find_baseline_file(preprocessed_path, subject_code):
    patterns = [f"{subject_code}_Baseline_raw.fif",
                f"{subject_code}_S__Baseline_raw.fif",
                f"{subject_code}_S_Baseline_raw.fif"]
    for p in patterns:
        candidate = Path(preprocessed_path) / p
        if candidate.exists():
            return str(candidate)
    # fallback glob
    files = glob.glob(os.path.join(preprocessed_path, f"{subject_code}*Baseline*.fif"))
    return files[0] if files else None


def preprocess_raw_data(raw, l_freq, h_freq, target_fs, ica_corr_threshold):
    # Resample
    if raw.info['sfreq'] > target_fs + 1:
        raw.resample(target_fs, npad='auto')
    # Montage
    try:
        raw.set_montage('easycap-M1', verbose=False)
    except:
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    # Filter
    iir = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq, h_freq, method='iir', iir_params=iir, phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir, phase='zero', verbose=False)
    # Interpolate bads
    raw.interpolate_bads(reset_bads=True, verbose=False)
    # Pick EEG only
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks)
    # ICA
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)
    # Identify artifacts via frontal channel correlation
    available = set(raw.info['ch_names'])
    fr = [ch for ch in ['Fp1','Fp2'] if ch in available]
    excl = []
    if fr:
        sources = ica.get_sources(raw).get_data()
        frontal_data = raw.copy().pick_channels(fr).get_data()
        for comp in range(sources.shape[0]):
            for idx, chdata in enumerate(frontal_data):
                corr = np.corrcoef(sources[comp], chdata)[0,1]
                if abs(corr) >= ica_corr_threshold:
                    excl.append(comp)
                    break
    ica.exclude = list(set(excl))
    raw = ica.apply(raw)
    return raw

# =============================================================================
# MODULE 2: EPOCH EXTRACTION
# =============================================================================
def extract_epochs_for_S9(directory, config):
    """
    For each subject/run, load segmented stimulus files, preprocess,
    trim 1s from start/end, baseline correct, and slice into epochs.
    """
    epochs_list = []
    fs = config['target_fs']
    win_samp = int(config['window_size'] * fs)
    step = int(win_samp * (1 - config['overlap']))

    root = Path(directory)
    subjects = list(root.glob('*Subject_*'))

    for subj in subjects:
        subject_code = subj.name.split('_')[0]
        for run in ['Run_1','Run_2']:
            run_path = subj / run / 'Preprocessed'
            if not run_path.exists():
                continue
            # (Baseline loading retained but not used for epochs)
            baseline_file = find_baseline_file(run_path, subject_code)
            if baseline_file:
                try:
                    braw = load_raw_data(baseline_file)
                    braw = preprocess_raw_data(braw,
                                              config['l_freq'],
                                              config['h_freq'],
                                              fs,
                                              config['ica_corr_threshold'])
                except Exception as e:
                    logger.warning(f"Baseline preprocess failed: {e}")
            stim_folder = run_path / f"{subject_code}_Stimulus"
            pattern = f"{subject_code}_S__9_seg_*_raw.vhdr"
            files = list(stim_folder.glob(pattern))
            counter = 0
            for f in files:
                try:
                    raw = load_raw_data(f)
                    raw = preprocess_raw_data(raw,
                                              config['l_freq'],
                                              config['h_freq'],
                                              fs,
                                              config['ica_corr_threshold'])
                except Exception as e:
                    logger.error(f"Error loading {f}: {e}")
                    continue
                data = raw.get_data()
                # Trim 1s off each end
                trim = int(fs*1)
                if data.shape[1] < 2*trim:
                    continue
                data = data[:, trim:-trim]
                # Baseline correction
                data = data - np.mean(data, axis=1, keepdims=True)
                # Sliding window
                for start in range(0, data.shape[1] - win_samp + 1, step):
                    epoch = data[:, start:start+win_samp]
                    epochs_list.append({
                        'data': epoch,
                        'class_label': subject_code,
                        'epoch': counter,
                        'run_id': run.split('_')[-1]
                    })
                    counter += 1
    return epochs_list

# =============================================================================
# MODULE 3: SAVE EPOCHS
# =============================================================================
def save_epochs_by_class(epochs, filename='raw_epochs_S9.h5'):
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as h5f:
        for e in epochs:
            grp = h5f.require_group(f"class_{e['class_label']}")
            run_grp = grp.require_group(f"Run_{e['run_id']}")
            ep_grp = run_grp.create_group(f"epoch_{e['epoch']}")
            ep_grp.create_dataset('data', data=e['data'])
    logger.info(f"Saved {len(epochs)} epochs to {filename}")

# =============================================================================
# MODULE 4: MAIN
# =============================================================================
def main():
    data_dir = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"  # adjust as needed
    logger.info("Extracting raw epochs...")
    epochs = extract_epochs_for_S9(data_dir, config)
    if not epochs:
        logger.error("No epochs extracted. Check paths and files.")
        return
    logger.info(f"Extracted {len(epochs)} epochs.")
    save_epochs_by_class(epochs, 'All_sub_raw_epochs_S9.h5')

if __name__ == '__main__':
    main()
