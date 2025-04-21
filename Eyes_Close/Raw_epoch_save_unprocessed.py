import os
import logging
from pathlib import Path

import h5py
import numpy as np
import mne

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================= #
# MODULE 1: DATA LOADING
# ============================================================================= #
def load_raw_data(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == '.fif':
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif suffix == '.vhdr':
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

# ============================================================================= #
# MODULE 2: RAW‑EPOCH EXTRACTION
# ============================================================================= #
def extract_epochs_for_S3(directory, fs, window_size=2, overlap=0.5):
    """
    Walk subject folders named *Subject_*
    Load raw files, resample to fs, trim 1s ends,
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
            total_files += len(files)

            cnt = 0
            for fpath in files:
                try:
                    raw = load_raw_data(fpath)

                    # ---- NEW: resample to target_fs ----
                    raw.resample(fs, npad='auto')

                    data = raw.get_data()  # shape: (n_ch, n_samples)
                except Exception as e:
                    logger.error("Failed %s: %s", fpath, e)
                    continue

                # Trim 1s at start/end
                trim = int(fs * 1)
                if data.shape[1] < 2 * trim + win_samp:
                    logger.warning("Too short after trim: %s", fpath.name)
                    continue
                data = data[:, trim:-trim]

                # Slide window
                for start in range(0, data.shape[1] - win_samp + 1, step):
                    window = data[:, start:start + win_samp]
                    epochs.append({
                        'data':        window,
                        'class_label': subj_code,
                        'run_id':      run.name.split('_')[-1],
                        'epoch':       cnt
                    })
                    cnt += 1

        summary[subj_code] = total_files

    return epochs, summary

# ============================================================================= #
# MODULE 3: SAVE RAW EPOCHS TO HDF5
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
# MODULE 4: MAIN PIPELINE
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

    save_epochs_by_class(epochs, filename="All_sub_eyes_close_epochs_unprocessed.h5")

    logger.info("File counts per subject:")
    for subj, cnt in summary.items():
        logger.info("  %s → %d files", subj, cnt)

if __name__ == "__main__":
    main()
