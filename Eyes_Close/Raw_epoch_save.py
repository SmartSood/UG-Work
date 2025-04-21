import os
import logging
from pathlib import Path
import numpy as np
import h5py
import mne
from mne.preprocessing import ICA

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================= #
# MODULE 1: DATA LOADING & PREPROCESSING
# ============================================================================= #
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fif':
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif file_path.suffix.lower() == '.vhdr':
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_raw_data(raw, l_freq=1.0, h_freq=40.0, target_fs=500, ica_corr_threshold=0.5):
    # Resample if needed
    current_fs = raw.info['sfreq']
    if current_fs > target_fs + 1:
        logger.info("Resampling from %s Hz to %s Hz...", current_fs, target_fs)
        raw.resample(target_fs, npad="auto")
    else:
        logger.info("Sampling rate OK: %s Hz", current_fs)

    # Apply bandpass and notch
    iir_params = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               method='iir', iir_params=iir_params,
               phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir',
                     iir_params=iir_params,
                     phase='zero', verbose=False)

    # Interpolate bad channels if any
    if not raw.info['bads']:
        logger.warning("No bad channels marked.")
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # Keep only EEG
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks)
    logger.info("Channels after pick: %d EEG channels", len(picks))

    # Run ICA
    logger.info("Running ICA...")
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)

    # Exclude components correlated with frontal channels
    excluded = []
    frontal = [ch for ch in ('Fp1','Fp2') if ch in raw.ch_names]
    if frontal:
        srcs = ica.get_sources(raw).get_data()
        frontal_data = raw.copy().pick_channels(frontal).get_data()
        for comp in range(srcs.shape[0]):
            for idx, ch in enumerate(frontal):
                corr = np.corrcoef(srcs[comp], frontal_data[idx])[0,1]
                if abs(corr) >= ica_corr_threshold:
                    excluded.append(comp)
                    break
        excluded = list(set(excluded))
        logger.info("Excluding ICA comps: %s", excluded)
    else:
        logger.warning("Frontal channels missing; skipping ICA exclusion.")
    ica.exclude = excluded
    raw = ica.apply(raw)
    logger.info("Preprocessing done.")
    return raw

# ============================================================================= #
# MODULE 2: RAW‐EPOCH EXTRACTION
# ============================================================================= #
def extract_epochs_for_S3(directory, fs, window_size=2, overlap=0.5):
    """
    Walk subject folders named *Subject_*
    Load & preprocess baseline files, trim 1s ends,
    then slide a [n_ch × window_samples] window
    and collect raw epochs.
    """
    epochs = []
    summary = {}
    root = Path(directory)
    win_samp = int(window_size * fs)
    step = int(win_samp * (1 - overlap))

    for subj in root.glob("*Subject_*"):
        subj_code = subj.name.split('_')[0]
        total_files = 0

        for run in (subj / "Run_1", subj / "Run_2"):
            if not run.exists():
                continue
            raw_dir = run / "Preprocessed"
            files = list(raw_dir.rglob("*.fif"))
            base_files = [f for f in files if "baseline" in f.name.lower()]
            total_files += len(base_files)

            cnt = 0
            for fpath in base_files:
                try:
                    raw = mne.io.read_raw_fif(str(fpath), preload=True)
                    raw = preprocess_raw_data(raw,
                                              l_freq=1.0,
                                              h_freq=40.0,
                                              target_fs=fs)
                    data = raw.get_data()  # shape: (n_ch, n_samples)
                except Exception as e:
                    logger.error("Failed %s: %s", fpath, e)
                    continue

                # Trim 1s at start/end
                trim = int(fs * 1)
                if data.shape[1] < 2*trim + win_samp:
                    logger.warning("Too short after trim: %s", fpath.name)
                    continue
                data = data[:, trim:-trim]

                # Slide window
                for start in range(0, data.shape[1] - win_samp + 1, step):
                    window = data[:, start:start + win_samp]
                    epochs.append({
                        'data':     window,
                        'class_label': subj_code,
                        'run_id':     run.name.split('_')[-1],
                        'epoch':      cnt
                    })
                    cnt += 1

        summary[subj_code] = total_files

    return epochs, summary

# ============================================================================= #
# MODULE 3: SAVE RAW EPOCHS TO HDF5
# ============================================================================= #
def save_epochs_by_class(epochs, filename="raw_epochs_S3.h5"):
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as hf:
        for ep in epochs:
            g_cls   = hf.require_group(f"class_{ep['class_label']}")
            g_run   = g_cls.require_group(f"Run_{ep['run_id']}")
            g_epoch = g_run.create_group(f"epoch_{ep['epoch']}")
            g_epoch.create_dataset("data",
                                   data=ep['data'],
                                   compression="gzip")
            g_epoch.attrs['class_label'] = ep['class_label']
            g_epoch.attrs['run_id']      = ep['run_id']
            g_epoch.attrs['epoch']       = ep['epoch']

    logger.info("Saved %d epochs into %s", len(epochs), filename)

# ============================================================================= #
# MODULE 4: MAIN PIPELINE
# ============================================================================= #
def main():
    directory   = r"G:\Smarth_work\unprocesed_dATA\Final_new_data"
    fs          = 500
    window_size = 2    # seconds
    overlap     = 0.5  # 50%

    logger.info("Extracting raw epochs...")
    epochs, summary = extract_epochs_for_S3(directory, fs, window_size, overlap)
    if not epochs:
        logger.error("No epochs extracted. Check your data paths.")
        return

    save_epochs_by_class(epochs, filename="All_sub_eyes_close_epochs.h5")

    logger.info("Baseline file counts per subject:")
    for subj, cnt in summary.items():
        logger.info("  %s → %d files", subj, cnt)

if __name__ == "__main__":
    main()
