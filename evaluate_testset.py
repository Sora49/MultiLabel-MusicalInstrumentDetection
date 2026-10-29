"""
Evaluate trained models on the official IRMAS test set.

Unlike the training clips (3 s, one label), each test excerpt is 5-20 s long
and lists *every* predominant instrument, so many excerpts have 2+ labels.
The models only know one instrument per 3 s clip, so each excerpt is cut into
3 s windows, every window is classified, and the softmax outputs are averaged
over the excerpt.

Only excerpts that contain at least one of the instruments the model was
trained on are used; labels for other instruments (e.g. electric guitar,
organ) are ignored when scoring, since the model can never predict them.

Usage:
    python evaluate_testset.py --test_dir IRMAS-TestingData
    python evaluate_testset.py --test_dir IRMAS-TestingData --model cnn
"""

import argparse
import csv
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from tqdm import tqdm

from features import SR, mel_spectrogram_from_audio, mfcc_features_from_audio, sliding_windows
from train_cnn import load_cnn_checkpoint


def read_labels(wav_path: Path) -> set:
    txt = wav_path.with_suffix(".txt")
    return {line.strip() for line in txt.read_text(errors="ignore").splitlines() if line.strip()}


def excerpt_features(wav_path: Path, hop: float, need_mel: bool, need_mfcc: bool):
    """Load one excerpt and return per-window mel and/or MFCC features."""
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    windows = [w for _, w in sliding_windows(y, hop=hop)]
    mels = np.stack([mel_spectrogram_from_audio(w) for w in windows]) if need_mel else None
    mfccs = np.stack([mfcc_features_from_audio(w) for w in windows]) if need_mfcc else None
    return mels, mfccs


def report(name, probs, labels, only_known, classes, threshold):
    """Print top-1 and multi-label scores for one model."""
    probs = np.array(probs)
    top1 = probs.argmax(axis=1)
    hit = np.array([classes[t] in labs for t, labs in zip(top1, labels)])
    single = np.array([len(labs) == 1 for labs in labels]) & only_known

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("Top-1 hit rate (top prediction is one of the labelled instruments):")
    print(f"  all {len(hit)} excerpts with >=1 known instrument: {hit.mean():.3f}")
    print(f"  {only_known.sum()} excerpts with only known instruments:  {hit[only_known].mean():.3f}")
    print(f"  {single.sum()} single-instrument excerpts (accuracy): {hit[single].mean():.3f}")

    # multi-label: every instrument whose averaged probability passes the
    # threshold counts as "present"; the top prediction always does
    y_true = np.array([[c in labs for c in classes] for labs in labels], dtype=int)
    y_pred = (probs >= threshold).astype(int)
    y_pred[np.arange(len(top1)), top1] = 1
    print(f"\nMulti-label scores (instrument predicted present if avg prob >= {threshold}):")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    return top1, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model", choices=["cnn", "baseline", "both"], default="both")
    parser.add_argument("--cnn_model", default="cnn_model.pt")
    parser.add_argument("--baseline_model", default="baseline_model.pkl")
    parser.add_argument("--encoder", default="label_encoder.pkl")
    parser.add_argument("--hop", type=float, default=1.5, help="seconds between window starts")
    # fixed in advance, not tuned on the test set
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--out_csv", default="testset_predictions.csv")
    args = parser.parse_args()

    le = joblib.load(args.encoder)
    classes = [str(c) for c in le.classes_]
    use_cnn, use_base = args.model in ("cnn", "both"), args.model in ("baseline", "both")
    cnn = load_cnn_checkpoint(args.cnn_model) if use_cnn else None
    base = joblib.load(args.baseline_model) if use_base else None

    excerpts = []
    for wav in sorted(Path(args.test_dir).rglob("*.wav")):
        labs = read_labels(wav)
        if labs & set(classes):
            excerpts.append((wav, labs))
    if not excerpts:
        raise SystemExit(f"No test excerpts with any of {classes} found in {args.test_dir}")
    labels = [labs for _, labs in excerpts]
    only_known = np.array([labs <= set(classes) for labs in labels])
    print(f"Using {len(excerpts)} excerpts containing at least one of {classes} "
          f"({only_known.sum()} contain only those)")

    cnn_probs, base_probs = [], []
    chunk = 64  # extract features for a chunk in parallel, then free them
    with tqdm(total=len(excerpts), desc="Evaluating", unit="excerpt") as pbar:
        for i in range(0, len(excerpts), chunk):
            batch = excerpts[i:i + chunk]
            feats = Parallel(n_jobs=-1)(
                delayed(excerpt_features)(wav, args.hop, use_cnn, use_base) for wav, _ in batch
            )
            for mels, mfccs in feats:
                # average the per-window class probabilities over the whole excerpt
                if use_cnn:
                    with torch.no_grad():
                        logits = cnn(torch.from_numpy(mels).unsqueeze(1))
                        cnn_probs.append(torch.softmax(logits, dim=1).mean(dim=0).numpy())
                if use_base:
                    base_probs.append(base.predict_proba(mfccs).mean(axis=0))
            pbar.update(len(batch))

    results = {}
    if use_base:
        results["baseline"] = report("MFCC + Random Forest", base_probs, labels, only_known, classes, args.threshold)
    if use_cnn:
        results["cnn"] = report("Mel-spectrogram CNN", cnn_probs, labels, only_known, classes, args.threshold)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["file", "true_labels"]
        for name in results:
            header += [f"{name}_top1", f"{name}_predicted_set"] + [f"{name}_p_{c}" for c in classes]
        writer.writerow(header)
        for j, (wav, labs) in enumerate(excerpts):
            row = [str(wav.relative_to(args.test_dir)), " ".join(sorted(labs))]
            for name, (top1, y_pred) in results.items():
                probs = base_probs if name == "baseline" else cnn_probs
                row += [classes[top1[j]], " ".join(c for c, p in zip(classes, y_pred[j]) if p)]
                row += [f"{p:.3f}" for p in probs[j]]
            writer.writerow(row)
    print(f"\nSaved per-excerpt predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
