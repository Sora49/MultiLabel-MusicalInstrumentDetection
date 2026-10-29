"""
Predict instruments in an audio file using a trained model.

Short clip (one 3 s window), as before:
    python predict.py --file clip.wav --model cnn
    python predict.py --file clip.wav --model cnn --model_path results_improve\\width32\\cnn_model.pt
    python predict.py --file clip.wav --model baseline

Any length, with a per-window timeline and a summary of how much of the audio
each instrument is in:
    python predict.py --file song.mp3 --timeline
    python predict.py --file song.mp3 --timeline --model_path results_multilabel\\from_combined\\multilabel_model.pt --csv song_timeline.csv

With a multi-label model, each window can contain several instruments (or none);
"present" uses the thresholds.json saved next to the model by
evaluate_multilabel.py (0.5 if missing). With a single-label CNN, each window
gets exactly one instrument: the most prominent one.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch

from features import SR, extract_mel_spectrogram, extract_mfcc_features, mel_spectrogram_from_audio, sliding_windows
from train_cnn import load_cnn_checkpoint


def predict_baseline(filepath: str):
    clf = joblib.load("baseline_model.pkl")
    le = joblib.load("label_encoder.pkl")

    feats = extract_mfcc_features(filepath).reshape(1, -1)
    probs = clf.predict_proba(feats)[0]
    pred_idx = np.argmax(probs)

    return le.classes_[pred_idx], probs, le.classes_


def load_model_info(model_path: str):
    """Returns (model, class names, is_multilabel)."""
    ckpt = torch.load(model_path, map_location="cpu")
    model = load_cnn_checkpoint(model_path)
    if "classes" in ckpt:
        classes = [str(c) for c in ckpt["classes"]]
    else:
        enc = Path(model_path).with_name("label_encoder.pkl")
        classes = [str(c) for c in joblib.load(enc if enc.exists() else "label_encoder.pkl").classes_]
    return model, classes, ckpt.get("task") == "multilabel"


def predict_cnn(filepath: str, model_path: str = "cnn_model.pt"):
    model, classes, multilabel = load_model_info(model_path)

    spec = extract_mel_spectrogram(filepath)
    x = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)

    with torch.no_grad():
        logits = model(x)
        probs = (torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)).numpy()[0]
    pred_idx = np.argmax(probs)

    return classes[pred_idx], probs, classes


def fmt_time(seconds: float) -> str:
    # rounded, so the last window (shifted to end exactly at the end of the file)
    # doesn't print with the same start time as the window before it
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def load_thresholds(model_path: str, classes, multilabel: bool):
    """Per-instrument thresholds saved by evaluate_multilabel.py; None if the model has none."""
    th_file = Path(model_path).with_name("thresholds.json")
    if multilabel and th_file.exists():
        saved = json.loads(th_file.read_text())["thresholds"]
        return np.array([saved[c] for c in classes])
    return None


def analyse_audio(y, model, multilabel: bool, thresholds=None, hop: float = 3.0, smooth: int = 1):
    """
    Score a whole signal (SR, mono) in 3 s windows. Returns (window start times,
    scores per window x instrument, present per window x instrument).
    smooth > 1 averages each window's scores with its neighbours (a centred
    moving average over `smooth` windows) before deciding, so instruments don't
    flicker on and off between adjacent windows.
    """
    windows = sliding_windows(y, hop=hop)
    starts = np.array([s for s, _ in windows])
    specs = np.stack([mel_spectrogram_from_audio(w) for _, w in windows])
    out = []
    with torch.no_grad():
        for i in range(0, len(specs), 64):
            logits = model(torch.from_numpy(specs[i:i + 64]).unsqueeze(1))
            out.append((torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)).numpy())
    scores = np.concatenate(out)

    if smooth > 1 and len(scores) > 1:
        k = smooth // 2
        padded = np.pad(scores, ((k, k), (0, 0)), mode="edge")
        scores = np.stack([padded[i:i + smooth].mean(0) for i in range(len(scores))])

    if multilabel:
        th = thresholds if thresholds is not None else np.full(scores.shape[1], 0.5)
        present = scores >= th
    else:
        # single-label: exactly one instrument per window, the most prominent
        present = np.zeros_like(scores, dtype=bool)
        present[np.arange(len(scores)), scores.argmax(1)] = True
    return starts, scores, present


def predict_timeline(filepath: str, model_path: str, hop: float, csv_path: str = None, smooth: int = 1):
    model, classes, multilabel = load_model_info(model_path)
    thresholds = load_thresholds(model_path, classes, multilabel)
    if multilabel and thresholds is None:
        print("(no thresholds.json next to the model; using 0.5 for every instrument)")

    y, _ = librosa.load(filepath, sr=SR, mono=True)
    starts, scores, present = analyse_audio(y, model, multilabel, thresholds, hop, smooth)
    windows = [(s, None) for s in starts]

    print(f"\n{Path(filepath).name}: {fmt_time(len(y) / SR)} long, {len(windows)} windows of 3 s "
          f"(every {hop:g} s) | {'multi-label' if multilabel else 'single-label'} model {model_path}\n")
    for (start, _), s, p in zip(windows, scores, present):
        found = [f"{c} {v:.2f}" for c, v, on in sorted(zip(classes, s, p), key=lambda t: -t[1]) if on]
        print(f"  {fmt_time(start)}-{fmt_time(start + 3)}  {', '.join(found) if found else '(none of the 5)'}")

    share = present.mean(0)
    label = "of the windows it is detected in" if multilabel else "of the windows it is the main instrument"
    print(f"\nSummary (share {label}):")
    for c, v, m in sorted(zip(classes, share, scores.mean(0)), key=lambda t: -t[1]):
        print(f"  {c:5s} {v * 100:5.1f}%   (mean score {m:.2f})")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s"] + [f"score_{c}" for c in classes] + [f"present_{c}" for c in classes])
            for (start, _), s, p in zip(windows, scores, present):
                w.writerow([f"{start:.2f}", f"{start + 3:.2f}"] + [f"{v:.3f}" for v in s] + [int(v) for v in p])
        print(f"\nSaved timeline to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to audio file (wav, mp3, flac, ...)")
    parser.add_argument("--model", choices=["baseline", "cnn"], default="cnn")
    parser.add_argument("--model_path", default="cnn_model.pt", help="which saved CNN to use")
    parser.add_argument("--timeline", action="store_true", help="score the whole file in 3 s windows")
    parser.add_argument("--hop", type=float, default=3.0, help="timeline: seconds between window starts")
    parser.add_argument("--csv", default=None, help="timeline: also save it as a CSV")
    parser.add_argument("--smooth", type=int, default=1,
                        help="timeline: average scores over this many neighbouring windows (1 = off, try 3)")
    args = parser.parse_args()

    if args.timeline:
        if args.model == "baseline":
            parser.error("--timeline works with CNN models")
        predict_timeline(args.file, args.model_path, args.hop, args.csv, args.smooth)
        return

    if args.model == "baseline":
        pred, probs, classes = predict_baseline(args.file)
    else:
        pred, probs, classes = predict_cnn(args.file, args.model_path)

    print(f"\nPredicted instrument: {pred}\n")
    print("Confidence by class:")
    for cls, p in sorted(zip(classes, probs), key=lambda t: -t[1]):
        print(f"  {cls:15s} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
