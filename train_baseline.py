"""
Baseline instrument classifier: MFCC features -> Random Forest.

Usage:
    python train_baseline.py --data_dir dataset
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from features import extract_mfcc_features, group_train_test_split, recording_id

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_dataset(data_dir: Path):
    X, y, groups = [], [], []
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not labels:
        raise SystemExit(f"No class subfolders found in {data_dir}")

    for label in labels:
        files = [f for f in (data_dir / label).iterdir() if f.suffix.lower() in AUDIO_EXTS]
        for f in tqdm(files, desc=f"Extracting MFCCs: {label}"):
            try:
                feats = extract_mfcc_features(str(f))
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
                continue
            X.append(feats)
            y.append(label)
            groups.append(recording_id(f))

    return np.array(X), np.array(y), np.array(groups), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X, y, groups, class_names = load_dataset(data_dir)
    print(f"\nLoaded {len(X)} samples from {len(set(groups))} recordings "
          f"across {len(class_names)} classes: {class_names}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split by recording, not by clip, so no recording appears in both sets
    train_idx, test_idx = group_train_test_split(y_enc, groups, args.test_size)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    print(f"Train: {len(train_idx)} clips | Test: {len(test_idx)} clips "
          f"(shared recordings: {len(set(groups[train_idx]) & set(groups[test_idx]))})")

    # grow the forest a chunk of trees at a time (warm_start keeps earlier
    # trees) so the progress bar can track it
    clf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42, n_jobs=-1)
    with tqdm(total=args.n_estimators, desc="Training Random Forest", unit="tree") as pbar:
        while clf.n_estimators < args.n_estimators:
            clf.n_estimators = min(clf.n_estimators + 25, args.n_estimators)
            clf.fit(X_train, y_train)
            pbar.update(clf.n_estimators - pbar.n)

    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(le.classes_)
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "baseline_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("\nSaved baseline_model.pkl and label_encoder.pkl")


if __name__ == "__main__":
    main()
