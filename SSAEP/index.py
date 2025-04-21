import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy.entropy import spectral_entropy
import ordpy
from librosa.feature import mfcc
from PyEMD import EMD

# Configuration for MFCC
config = {
    "n_mfcc": 13,
    "n_fft": 2048
}

def compute_features(data, fs, eps=1e-6):
    features = {}
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)
    features['zero_crossing_rate'] = ((data[:-1] * data[1:]) < 0).sum()

    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / (activity + eps))
    complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + eps)) / (mobility + eps))
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity

    nperseg = int(fs * 2) if len(data) >= fs * 2 else len(data)
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + eps

    delta_idx = (freqs >= 0.5) & (freqs <= 4)
    beta_idx = (freqs >= 13) & (freqs <= 30)
    theta_idx = (freqs >= 4) & (freqs <= 8)
    alpha_idx = (freqs >= 8) & (freqs <= 13)

    delta_power = np.sum(psd[delta_idx])
    beta_power = np.sum(psd[beta_idx])
    theta_power = np.sum(psd[theta_idx])
    alpha_power = np.sum(psd[alpha_idx])

    features['delta_relative_power'] = delta_power / total_power
    features['beta_relative_power'] = beta_power / total_power

    try:
        features['spectral_entropy'] = spectral_entropy(data, sf=fs, method='welch', normalize=True)
    except:
        features['spectral_entropy'] = np.nan

    features['theta_alpha_ratio'] = theta_power / (alpha_power + eps)
    features['beta_theta_ratio'] = beta_power / (theta_power + eps)

    try:
        features['permutation_entropy'] = ordpy.permutation_entropy(data, dx=3)
    except:
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

    mfccs = mfcc(y=data, sr=fs, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])
    for i in range(config["n_mfcc"]):
        features[f'mfcc_{i + 1}'] = np.mean(mfccs[i, :])

    emd = EMD()
    imfs = emd(data)
    if len(imfs) > 0:
        try:
            features['imf_1_entropy'] = spectral_entropy(imfs[0], sf=fs, method='welch', normalize=True)
        except:
            features['imf_1_entropy'] = np.nan
    else:
        features['imf_1_entropy'] = np.nan

    return features

def save_feature_index_map(fs=500, n_samples=1000, output_csv='feature_index_map.csv'):
    dummy_signal = np.random.randn(n_samples)
    feats = compute_features(dummy_signal, fs)
    feature_names = list(feats.keys())
    df = pd.DataFrame({
        'Index': range(len(feature_names)),
        'Feature Name': feature_names
    })
    df.to_csv(output_csv, index=False)
    print(f"Feature index map saved to '{output_csv}'")

if __name__ == "__main__":
    save_feature_index_map()
