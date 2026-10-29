"""
Shared audio-loading and feature-extraction utilities.
"""

import re
from pathlib import Path

import numpy as np
import librosa
from sklearn.model_selection import StratifiedGroupKFold

SR = 22050
DURATION = 3.0  # seconds, all clips are trimmed/padded to this length
N_MFCC = 40
N_MELS = 128


def recording_id(filepath) -> str:
    """
    IRMAS clips cut from the same recording differ only in the trailing __N:
    '008__[flu][nod][cla]0393__1.wav' -> '008__[flu][nod][cla]0393'
    """
    return re.sub(r"__\d+$", "", Path(filepath).stem)


def group_train_test_split(y, groups, test_size: float = 0.2, seed: int = 42):
    """
    Stratified train/test split that keeps every clip of a recording on the
    same side, so the test set only contains recordings the model never saw.
    Returns (train_idx, test_idx).
    """
    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Trim (keeping the center) or zero-pad the last axis to target_len samples."""
    n = y.shape[-1]
    if n > target_len:
        start = (n - target_len) // 2
        y = y[..., start:start + target_len]
    elif n < target_len:
        pad = [(0, 0)] * (y.ndim - 1) + [(0, target_len - n)]
        y = np.pad(y, pad)
    return y


def load_fixed_length(filepath: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Load audio, mono, resampled, trimmed/padded to a fixed length."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return fix_length(y, int(sr * duration))


def load_fixed_length_stereo(filepath: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Like load_fixed_length but keeps both channels: shape (2, samples). Mono files are duplicated."""
    y, _ = librosa.load(filepath, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return fix_length(y, int(sr * duration))


def sliding_windows(y: np.ndarray, sr: int = SR, duration: float = DURATION, hop: float = 1.5):
    """
    Cut a long signal into fixed-length windows (the length the models were
    trained on). Returns a list of (start_seconds, window) pairs; a final
    partial window is zero-padded, and anything shorter than one window
    becomes a single padded window.
    """
    win, step = int(sr * duration), int(sr * hop)
    starts = list(range(0, max(len(y) - win, 0) + 1, step))
    if starts[-1] + win < len(y):
        starts.append(len(y) - win)  # cover the tail without padding
    out = []
    for s in starts:
        w = y[s:s + win]
        if len(w) < win:
            w = np.pad(w, (0, win - len(w)))
        out.append((s / sr, w))
    return out


def extract_mfcc_features(filepath: str) -> np.ndarray:
    return mfcc_features_from_audio(load_fixed_length(filepath))


def extract_mel_spectrogram(filepath: str) -> np.ndarray:
    return mel_spectrogram_from_audio(load_fixed_length(filepath))


def mfcc_features_from_audio(y: np.ndarray) -> np.ndarray:
    """
    Returns a fixed-size feature vector: mean + std of MFCCs (and deltas)
    across time. Good input for classical ML models (Random Forest, SVM).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ])
    return feats


def mel_power_from_audio(y: np.ndarray) -> np.ndarray:
    """Un-normalised mel power spectrogram (n_mels x time). Powers add, so these can be mixed."""
    return librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS).astype(np.float32)


def normalize_mel_power(mel: np.ndarray) -> np.ndarray:
    """Mel power -> log scale -> roughly [0, 1]; the CNN's input format."""
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [0, 1] for stable training
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def mel_spectrogram_from_audio(y: np.ndarray) -> np.ndarray:
    """
    Returns a 2D log-mel spectrogram (n_mels x time), suitable as a
    single-channel "image" input to a CNN.
    """
    return normalize_mel_power(mel_power_from_audio(y))
