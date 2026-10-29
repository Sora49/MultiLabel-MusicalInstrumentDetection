"""
Transfer-learning comparison: frozen PANNs CNN14 embeddings -> logistic regression.

CNN14 (Kong et al., 2020) was trained on ~2M AudioSet clips. We don't fine-tune
it; each 3 s clip is passed through once to get a 2048-d embedding, and only a
linear classifier is trained on top. The train/val/test split is identical to
train_cnn.py, so the test numbers are directly comparable.

Setup (once): pip install panns_inference, and put Cnn14_mAP=0.431.pth and
class_labels_indices.csv in ~/panns_data (see README).

Usage:
    python train_pretrained.py --data_dir IRMAS-TrainingData
"""

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from features import DURATION, group_train_test_split, recording_id

PANNS_SR = 32000  # CNN14 was trained on 32 kHz audio
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_clip(path, sr=PANNS_SR, duration=DURATION):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    n = int(sr * duration)
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def extract_embeddings(data_dir: Path, cache: Path, batch_size: int = 32):
    """Embed every clip once and cache the result; later runs just load it."""
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cached embeddings from {cache}")
        return d["X"], d["y"], d["groups"]

    from panns_inference import AudioTagging  # imported here: slow, prints a banner

    tagger = AudioTagging(checkpoint_path=None, device="cpu")
    files, y, groups = [], [], []
    for label_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        for f in label_dir.iterdir():  # same order as train_cnn.py, so the split matches
            if f.suffix.lower() in AUDIO_EXTS:
                files.append(f)
                y.append(label_dir.name)
                groups.append(recording_id(f))

    X = []
    for i in tqdm(range(0, len(files), batch_size), desc="CNN14 embeddings", unit="batch"):
        audio = np.stack([load_clip(f) for f in files[i:i + batch_size]])
        _, emb = tagger.inference(audio)
        X.append(emb)
    X, y, groups = np.concatenate(X), np.array(y), np.array(groups)
    np.savez(cache, X=X, y=y, groups=groups)
    print(f"Saved embeddings to {cache}")
    return X, y, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default="panns_embeddings.npz")
    args = parser.parse_args()

    X, y, groups = extract_embeddings(Path(args.data_dir), Path(args.cache))
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n{len(X)} clips, {X.shape[1]}-d embeddings, classes: {list(le.classes_)}")

    # exactly the same split as train_cnn.py
    train_idx, test_idx = group_train_test_split(y_enc, groups, args.test_size, args.seed)
    sub_train, sub_val = group_train_test_split(y_enc[train_idx], groups[train_idx], args.val_size, args.seed)
    train_idx, val_idx = train_idx[sub_train], train_idx[sub_val]
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # pick the regularisation strength on validation, never on test
    best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        )
        clf.fit(X[train_idx], y_enc[train_idx])
        val_acc = clf.score(X[val_idx], y_enc[val_idx])
        print(f"  C={C:<6} val_acc={val_acc:.3f}")
        if best is None or val_acc > best[0]:
            best = (val_acc, C, clf)
    val_acc, C, clf = best
    print(f"Best C={C} (val_acc={val_acc:.3f})")

    y_pred = clf.predict(X[test_idx])
    print(f"\nTest accuracy: {(y_pred == y_enc[test_idx]).mean():.3f}")
    print("\n=== Classification Report (test set) ===")
    print(classification_report(y_enc[test_idx], y_pred, target_names=le.classes_))
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(le.classes_)
    print(confusion_matrix(y_enc[test_idx], y_pred))

    joblib.dump({"classifier": clf, "label_encoder": le}, "pretrained_model.pkl")
    print("\nSaved pretrained_model.pkl")


if __name__ == "__main__":
    main()
