import os
import logging
from pathlib import Path

import h5py
import numpy as np
import mne
from mne.preprocessing import ICA

# ----------------------------------------------------------------------------- #
# Global Configuration
# ----------------------------------------------------------------------------- #
config = {
    "window_size": 2,         # seconds
    "overlap": 0.5,           # fraction of overlap
    "target_fs": 500,         # Hz
    "l_freq": 1.0,            # Hz
    "h_freq": 40.0,           # Hz
    "ica_corr_threshold": 0.5 # correlation threshold for ICA artifact removal
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
# MODULE 2: PREPROCESSING
# =============================================================================
def preprocess_raw_data(raw, l_freq, h_freq, target_fs, ica_corr_threshold):
    """
    Preprocess raw EEG data: resampling, filtering, notch, interpolation,
    channel selection, and ICA-based artifact removal.
    """
    # Resample if needed
    current_fs = raw.info['sfreq']
    if current_fs > target_fs + 1:
        logger.info("Resampling from %.1f Hz to %d Hz", current_fs, target_fs)
        raw.resample(target_fs, npad='auto')
    else:
        logger.info("Sampling rate OK: %.1f Hz", current_fs)

    # Montage
    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception as e:
        logger.warning("Easycap montage failed: %s; using standard_1020", e)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose=False)

    # Band-pass & notch filters
    iir_params = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq, h_freq, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir_params, phase='zero', verbose=False)

    # Interpolate bad channels if any are marked
    if not raw.info['bads']:
        logger.warning("No bad channels marked")
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # Pick only EEG channels
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks)
    logger.info("Selected %d EEG channels", len(picks))

    # ICA for artifact removal
    logger.info("Fitting ICA (20 components)...")
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)

    # Identify components to exclude based on frontal channels correlation
    excluded = []
    frontal = [ch for ch in ['Fp1','Fp2'] if ch in raw.ch_names]
    if frontal:
        logger.info("Checking ICA components against %s", frontal)
        data_frontal = raw.copy().pick_channels(frontal).get_data()
        sources = ica.get_sources(raw).get_data()
        for comp in range(sources.shape[0]):
            for chan_idx in range(data_frontal.shape[0]):
                corr = np.corrcoef(sources[comp], data_frontal[chan_idx])[0,1]
                if abs(corr) >= ica_corr_threshold:
                    excluded.append(comp)
                    break
    else:
        logger.warning("Frontal channels missing; skipping correlation check")

    ica.exclude = list(set(excluded))
    if ica.exclude:
        logger.info("Excluding ICA components: %s", ica.exclude)
    raw = ica.apply(raw)
    logger.info("Preprocessing complete")
    return raw

# =============================================================================
# MODULE 3: EXTRACT RAW EPOCHS
# =============================================================================
def extract_epochs_for_EyesOpen(root_dir, cfg):
    """
    Traverse Subject_/Run folders, locate EyesOpen .vhdr files,
    preprocess, trim 1s at beginning/end, and slide a window to collect raw epochs.
    Returns list of dicts: {data, class_label, epoch, run_id}.
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
                    raw = preprocess_raw_data(raw,
                                              l_freq=cfg["l_freq"],
                                              h_freq=cfg["h_freq"],
                                              target_fs=fs,
                                              ica_corr_threshold=cfg["ica_corr_threshold"])
                    data = raw.get_data()  # shape: (n_channels, n_samples)
                except Exception as e:
                    logger.error("Error loading %s: %s", vhdr, e)
                    continue

                # Trim 1s at start/end
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
                        "data": epoch,
                        "class_label": subj,
                        "epoch": ep_idx,
                        "run_id": run_id
                    })
                    ep_idx += 1

    return epochs

# =============================================================================
# MODULE 4: SAVE RAW EPOCHS
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
# MODULE 5: MAIN
# =============================================================================
def main():
    directory = r"G:\Raw Data\Final_new_data"  # update as needed
    logger.info("Starting raw epoch extraction for EyesOpen")
    epochs = extract_epochs_for_EyesOpen(directory, config)
    if not epochs:
        logger.error("No epochs extracted; check directories and preprocessing")
        return
    save_epochs_by_class(epochs, filename="AllSub_EyesOpen_epochs.h5")
    logger.info("Done!")

if __name__ == "__main__":
    main()
