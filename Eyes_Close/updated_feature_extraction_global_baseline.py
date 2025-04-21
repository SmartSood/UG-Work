import os
import glob
import logging
from pathlib import Path

import h5py
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from antropy.entropy import spectral_entropy, sample_entropy
import ordpy
from librosa.feature import mfcc
from PyEMD import EMD
import pickle

# ----------------------------------------------------------------------------- #
# Global Configuration
# ----------------------------------------------------------------------------- #
config = {
    "n_mfcc": 13,     # Number of MFCC coefficients
    "n_fft": 2048     # FFT window size for MFCC computation
}

# ----------------------------------------------------------------------------- #
# Setup Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Configuration: %s", config)

# ============================================================================= #
# MODULE 1: DATA LOADING & PREPROCESSING
# ============================================================================= #
def load_raw_data(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fif':
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif file_path.suffix.lower() == '.vhdr':
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_raw_data(raw, l_freq=0.1, h_freq=70.0, target_fs=500, ica_corr_threshold=0.5):
    """
    Preprocess raw EEG data while ensuring only EEG channels are considered.
    """
    current_fs = raw.info['sfreq']
    if current_fs > target_fs + 1:
        logger.info("Resampling from %s Hz to %s Hz...", current_fs, target_fs)
        raw.resample(target_fs, npad="auto")
    else:
        logger.info("No resampling needed. Current sampling frequency: %s Hz.", current_fs)

    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception as e:
        logger.warning("Montage setting failed, using standard 10-20 positions: %s", e)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose=False)

    iir_params = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir_params, phase='zero', verbose=False)

    if not raw.info['bads']:
        logger.warning("No bad channels marked for interpolation.")
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # Select only EEG channels
    eeg_channels = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks=eeg_channels)
    logger.info("Using %d EEG channels for feature extraction.", len(eeg_channels))

    # Run ICA for artifact removal
    logger.info("Running ICA to remove artifacts...")
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)

    excluded_components = []
    frontal_channels = ['Fp1', 'Fp2']
    available_channels = set(raw.info['ch_names'])
    frontal_channels = [ch for ch in frontal_channels if ch in available_channels]

    if frontal_channels:
        logger.info("Checking ICA components against frontal channels: %s", frontal_channels)
        frontal_data = raw.copy().pick_channels(frontal_channels).get_data()
        ica_sources = ica.get_sources(raw).get_data()
        for comp_idx in range(ica_sources.shape[0]):
            for ch in range(frontal_data.shape[0]):
                corr_value = np.corrcoef(ica_sources[comp_idx], frontal_data[ch])[0, 1]
                logger.debug("ICA Component %d - %s Correlation: %.4f", comp_idx, frontal_channels[ch], corr_value)
                if abs(corr_value) >= ica_corr_threshold:
                    excluded_components.append(comp_idx)
                    break
        excluded_components = list(set(excluded_components))
        logger.info("Excluding ICA components: %s", excluded_components)
    else:
        logger.warning("Frontal channels (Fp1, Fp2) missing. Skipping ICA artifact removal based on correlation.")

    ica.exclude = excluded_components
    raw = ica.apply(raw)
    logger.info("Preprocessing complete.")
    return raw

# ============================================================================= #
# MODULE 2: FEATURE EXTRACTION FUNCTIONS
# ============================================================================= #
def compute_features(data, fs, eps=1e-6):
    features = {}
    # Statistical Features
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)
    features['zero_crossing_rate'] = ((data[:-1] * data[1:]) < 0).sum()
    
    # Hjorth Parameters
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / (activity + eps))
    complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + eps)) / (mobility + eps))
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    
    # Frequency-Domain Features using Welch's method
    nperseg = int(fs * 2) if len(data) >= fs * 2 else len(data)
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + eps
    
    # Delta Relative Power (0.5 - 4 Hz)
    delta_idx = (freqs >= 0.5) & (freqs <= 4)
    delta_power = np.sum(psd[delta_idx])
    features['delta_relative_power'] = delta_power / total_power
    
    # Beta Relative Power (13 - 30 Hz)
    beta_idx = (freqs >= 13) & (freqs <= 30)
    beta_power = np.sum(psd[beta_idx])
    features['beta_relative_power'] = beta_power / total_power
    
    # Theta and Alpha for band ratios
    theta_idx = (freqs >= 4) & (freqs <= 8)
    theta_power = np.sum(psd[theta_idx])
    alpha_idx = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.sum(psd[alpha_idx])
    
    try:
        features['spectral_entropy'] = spectral_entropy(data, sf=fs, method='welch', normalize=True)
    except Exception as e:
        logger.error("Spectral entropy error: %s", e)
        features['spectral_entropy'] = np.nan
    
    features['theta_alpha_ratio'] = theta_power / (alpha_power + eps)
    features['beta_theta_ratio'] = beta_power / (theta_power + eps)
    
    try:
        features['permutation_entropy'] = ordpy.permutation_entropy(data, dx=3)
    except Exception as e:
        logger.error("Permutation entropy error: %s", e)
        features['permutation_entropy'] = np.nan
    
    def hurst_exponent(signal):
        N = len(signal)
        Y = np.cumsum(signal - np.mean(signal))
        R = np.max(Y) - np.min(Y)
        S = np.std(signal)
        return np.log(R / (S + eps) + eps) / np.log(N)
    features['hurst_exponent'] = hurst_exponent(data)
    
    def katz_fd(signal):
        L = np.sum(np.sqrt(1 + np.diff(signal) ** 2))
        d = np.max(np.abs(signal - signal[0]))
        N = len(signal)
        return np.log10(N) / (np.log10(N) + np.log10(d / L + eps))
    features['katz_fd'] = katz_fd(data)
    
    def higuchi_fd(signal, kmax=10):
        L = []
        N = len(signal)
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = np.sum(np.abs(np.diff(signal[m::k])))
                Lmk *= (N - 1) / (len(signal[m::k]) * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        L = np.array(L)
        lnL = np.log(L + eps)
        lnk = np.log(1.0 / np.arange(1, kmax + 1))
        higuchi, _ = np.polyfit(lnk, lnL, 1)
        return higuchi
    features['higuchi_fd'] = higuchi_fd(data)
    
    # MFCCs (n_mfcc coefficients)
    mfccs = mfcc(y=data, sr=fs, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])
    for i in range(config["n_mfcc"]):
        features[f'mfcc_{i + 1}'] = np.mean(mfccs[i, :])
    
    # EMD-Based Feature: imf_1_entropy
    emd = EMD()
    imfs = emd(data)
    if len(imfs) > 0:
        try:
            features['imf_1_entropy'] = spectral_entropy(imfs[0], sf=fs, method='welch', normalize=True)
        except Exception as e:
            logger.error("IMF entropy error: %s", e)
            features['imf_1_entropy'] = np.nan
    else:
        features['imf_1_entropy'] = np.nan
    
    return features

def get_feature_order(fs=500, n_samples=1000):
    """
    Run compute_features on a dummy signal to determine the feature order.
    """
    dummy_signal = np.random.randn(n_samples)
    feats = compute_features(dummy_signal, fs)
    return list(feats.keys())

def extract_all_features(epoch_data, fs):
    """
    Compute and return features for one epoch of data.
    """
    spectral_feats = compute_features(epoch_data, fs)
    feat_vector = np.array(list(spectral_feats.values()))
    logger.info("Extracted Feature Vector Shape: %s", feat_vector.shape)
    return feat_vector

# ============================================================================= #
# MODULE 3: FEATURE EXTRACTION FOR EEG SEGMENTS & BASELINE
# ============================================================================= #
def extract_features_for_S3(directory, fs, window_size=2, overlap=0.5):
    feature_matrices = []
    baseline_summary = {}  # Dictionary to store baseline file count per subject
    root_dir = Path(directory)
    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))

    subject_folders = list(root_dir.glob("*Subject_*"))
    
    for subject_folder in subject_folders:
        subject_code = subject_folder.name.split('_')[0]
        run_folders = [subject_folder / "Run_1", subject_folder / "Run_2"]
        total_baseline_files = 0  # Counter for this subject

        for run_folder in run_folders:
            if not run_folder.exists():
                continue

            # Define run_id from run folder name
            run_id = run_folder.name.split("_")[-1]
            preprocessed_path = run_folder / "Preprocessed"

            # Use rglob for a recursive search in the preprocessed folder
            all_fif_files = list(preprocessed_path.rglob("*.fif"))
            eeg_files = [f for f in all_fif_files if "baseline" in f.name.lower()]
            total_baseline_files += len(eeg_files)
            
            if not eeg_files:
                logger.warning("No baseline files found for %s in %s", subject_code, preprocessed_path)
                continue
            logger.debug("Found %d baseline file(s) for subject %s in run %s", len(eeg_files), subject_code, run_id)

            epoch_counter = 0
            for eeg_file in eeg_files:
                try:
                    logger.debug("Loading file: %s", eeg_file)
                    raw = mne.io.read_raw_fif(str(eeg_file), preload=True)
                    raw = preprocess_raw_data(raw, l_freq=1.0, h_freq=40.0, target_fs=fs)
                    eeg_data = raw.get_data()
                    logger.debug("Loaded data shape: %s", eeg_data.shape)
                except Exception as e:
                    logger.error("Error processing file %s for %s in %s: %s", eeg_file, subject_code, run_folder, e)
                    continue

                # Trim first and last 1 second
                n_samples_total = eeg_data.shape[1]
                trim_samples = int(fs * 1)
                if n_samples_total < 2 * trim_samples:
                    logger.warning("Segment %s too short after trimming. Skipping.", eeg_file.name)
                    continue
                eeg_data = eeg_data[:, trim_samples:n_samples_total - trim_samples]
                logger.debug("Data shape after trimming: %s", eeg_data.shape)

                # Apply global baseline correction (subtract channel-wise overall mean)
                global_baseline = np.mean(eeg_data, axis=1, keepdims=True)
                eeg_data = eeg_data - global_baseline
                logger.debug("Global baseline correction applied.")

                # Create overlapping epochs via sliding window
                for start in range(0, eeg_data.shape[1] - window_samples + 1, step_size):
                    segment_start_idx = start
                    segment_end_idx = start + window_samples
                    logger.debug("Window from %.3fs to %.3fs", segment_start_idx/fs, segment_end_idx/fs)
                    epoch_features = []
                    for ch_data in eeg_data:
                        feat_vector = extract_all_features(ch_data[segment_start_idx:segment_end_idx], fs)
                        epoch_features.append(feat_vector)
                    epoch_features_matrix = np.vstack(epoch_features)
                    logger.debug("Epoch %d feature matrix shape: %s", epoch_counter, epoch_features_matrix.shape)
                    feature_matrices.append({
                        'features': epoch_features_matrix,
                        'class_label': subject_code,
                        'epoch': epoch_counter,
                        'run_id': run_id
                    })
                    epoch_counter += 1

        # Update the baseline summary for this subject
        baseline_summary[subject_code] = total_baseline_files

    return feature_matrices, baseline_summary

# ============================================================================= #
# MODULE 4: SAVE FEATURES IN HDF5 FORMAT
# ============================================================================= #
def save_features_by_class(feature_matrices, filename='eeg_features_S4.h5'):
    """
    Save extracted feature matrices into an HDF5 file, organized by:
      - Class Label (Subject)
      - Run (e.g., 'Run_1' or 'Run_2')
      - Epoch Number
    """
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, 'w') as h5f:
        for data in feature_matrices:
            class_label = data['class_label']
            epoch_number = data['epoch']
            run_id = data.get('run_id', 'Unknown')
            class_group = h5f.require_group(f'class_{class_label}')
            run_group = class_group.require_group(f'Run_{run_id}')
            if f'epoch_{epoch_number}' in run_group:
                del run_group[f'epoch_{epoch_number}']
            epoch_group = run_group.create_group(f'epoch_{epoch_number}')
            epoch_group.create_dataset('features', data=data['features'])
            epoch_group.attrs['class_label'] = class_label
            epoch_group.attrs['epoch'] = epoch_number
            epoch_group.attrs['run'] = run_id
    logger.info("Features saved successfully to %s", filename)

# ============================================================================= #
# MODULE 5: MAIN PIPELINE
# ============================================================================= #
def main():
    # Set parameters and file paths
    directory = r"G:\Raw Data\Test_folder_3"  # Update this path as needed
    fs = 500       # Sampling frequency
    window_size = 2  # Window size in seconds
    overlap = 0.5    # 50% overlap

    logger.info("Starting feature extraction for S3 segments (hierarchical save, no ratio normalization)...")
    feature_matrices, baseline_summary = extract_features_for_S3(directory, fs, window_size, overlap)
    if not feature_matrices:
        logger.error("No feature matrices extracted. Please check data folders and file paths.")
        return

    logger.info("Extracted %d feature matrices.", len(feature_matrices))
    filename_out = "new_chetan_eyes_close_hierarchical.h5"
    save_features_by_class(feature_matrices, filename_out)
    logger.info("Saved hierarchical features to %s. Done!", filename_out)
    
    # Print the baseline summary at the end of processing
    logger.info("Baseline Files Summary:")
    for subject, count in baseline_summary.items():
        logger.info("Subject %s: %d baseline file(s) found", subject, count)

if __name__ == "__main__":
    main()
