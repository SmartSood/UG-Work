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
    "window_size": 2.0,    # seconds
    "overlap": 0.5,        # fraction
    "fs": 500,             # target sampling frequency
}

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Configuration: %s", config)

# -----------------------------------------------------------------------------
# Utility: find_baseline_file
# -----------------------------------------------------------------------------
def find_baseline_file(subject_output_dir, subject_code):
    patterns = [
        f"{subject_code}_Baseline_raw.fif",
        f"{subject_code}_S__Baseline_raw.fif",
        f"{subject_code}_S_Baseline_raw.fif"
    ]
    subject_output_dir = Path(subject_output_dir)
    for p in patterns:
        fp = subject_output_dir / p
        if fp.exists():
            return str(fp)
    alt = glob.glob(str(subject_output_dir / f"{subject_code}*Baseline*.fif"))
    return alt[0] if alt else None

# -----------------------------------------------------------------------------
# Preprocessing Function
# -----------------------------------------------------------------------------
def preprocess_raw_data(raw, l_freq=1.0, h_freq=40.0, target_fs=None, ica_corr_threshold=0.5):
    current_fs = raw.info['sfreq']
    if target_fs and current_fs > target_fs + 1:
        raw.resample(target_fs, npad="auto")
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               method='iir',
               iir_params=dict(order=4, ftype='butter', output='sos'),
               phase='zero')
    raw.notch_filter(freqs=50,
                     method='iir',
                     iir_params=dict(order=4, ftype='butter', output='sos'),
                     phase='zero')
    raw.interpolate_bads(reset_bads=True)
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks=picks_eeg)
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)
    excluded = []
    frontal = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.info['ch_names']]
    if frontal:
        picks_frontal = mne.pick_channels(raw.info['ch_names'], frontal)
        frontal_data = raw.copy().pick(picks=picks_frontal).get_data()
        sources = ica.get_sources(raw).get_data()
        for comp in range(sources.shape[0]):
            for ch_idx in range(frontal_data.shape[0]):
                corr = np.corrcoef(sources[comp], frontal_data[ch_idx])[0, 1]
                if abs(corr) >= ica_corr_threshold:
                    excluded.append(comp)
                    break
        ica.exclude = list(set(excluded))
        raw = ica.apply(raw)
    else:
        logger.warning("Frontal channels missing; skipping ICA-based artifact removal.")
    return raw

# -----------------------------------------------------------------------------
# Extract raw multichannel epochs for S11
# -----------------------------------------------------------------------------
def extract_epochs_for_S11(directory, fs, window_size, overlap):
    epochs = []
    win_samp = int(window_size * fs)
    step = int(win_samp * (1 - overlap))
    root = Path(directory)
    subjects = list(root.glob("*Subject_*"))
    for subj in subjects:
        subj_code = subj.name.split('_')[0]
        for run in ['Run_1', 'Run_2']:
            run_dir = subj / run
            if not run_dir.exists():
                continue
            baseline_dir = run_dir / 'Preprocessed' / f"{subj_code}_SSAEP"
            baseline_file = find_baseline_file(baseline_dir, subj_code)
            if baseline_file:
                raw_base = mne.io.read_raw_fif(baseline_file, preload=True)
                preprocess_raw_data(raw_base, l_freq=1.0, h_freq=40.0, target_fs=fs)
            pattern = f"{subj_code}_S11_segment_*_raw.vhdr"
            seg_files = list(baseline_dir.glob(pattern))
            epoch_idx = 0  # Moved outside segment loop to accumulate across all segments
            for seg in seg_files:
                raw_seg = mne.io.read_raw_brainvision(str(seg), preload=True)
                raw_seg = preprocess_raw_data(raw_seg, l_freq=1.0, h_freq=40.0, target_fs=fs)
                data = raw_seg.get_data()
                trim = int(fs * 1)
                if data.shape[1] <= 2 * trim:
                    continue
                data = data[:, trim:-trim]
                data = data - np.mean(data, axis=1, keepdims=True)
                for start in range(0, data.shape[1] - win_samp + 1, step):
                    seg_data = data[:, start:start + win_samp]
                    epochs.append({
                        'data': seg_data,
                        'class_label': subj_code,
                        'run_id': run,
                        'epoch': epoch_idx
                    })
                    epoch_idx += 1
    return epochs

# -----------------------------------------------------------------------------
# Save epochs into HDF5
# -----------------------------------------------------------------------------
def save_epochs_by_class(epochs, filename='eeg_epochs_S11.h5'):
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as h5f:
        for item in epochs:
            class_grp = h5f.require_group(f"class_{item['class_label']}")
            run_grp = class_grp.require_group(item['run_id'])
            ep_name = f"epoch_{item['epoch']}"
            if ep_name in run_grp:
                del run_grp[ep_name]
            ep_grp = run_grp.create_group(ep_name)
            ep_grp.create_dataset('data', data=item['data'])
    logger.info("Saved %d epochs to %s", len(epochs), filename)

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    directory = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"
    params = config
    logger.info("Extracting raw epochs for S11...")
    epoch_list = extract_epochs_for_S11(directory, params['fs'], params['window_size'], params['overlap'])
    if not epoch_list:
        logger.error("No epochs extracted. Check paths and data.")
        return
    save_epochs_by_class(epoch_list, filename='All_sub_eeg_epochs_S11.h5')

if __name__ == '__main__':
    main()
