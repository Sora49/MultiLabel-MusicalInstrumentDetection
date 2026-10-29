"""
Score instrument *detection* on the official IRMAS test set (real music, every
excerpt labelled with all predominant instruments, often several).

For each excerpt: cut into 3 s windows, score every window (sigmoid for
multi-label models, softmax for the single-label CNNs), and aggregate over the
excerpt (mean or max). An instrument counts as "present" if its score passes a
per-instrument threshold.

Honest thresholds: the 208 songs are split in half. Thresholds (and mean-vs-max
aggregation) are chosen on half A only; all reported numbers are on half B.
The chosen settings are saved as thresholds.json next to each model, which
predict.py uses.

Excerpts containing none of the 5 instruments are kept: for a detector the
correct answer there is "none of them".

Usage:
    python evaluate_multilabel.py --models results_multilabel\\from_combined\\multilabel_model.pt cnn_model.pt
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, classification_report, f1_score
from tqdm import tqdm

from evaluate_testset import excerpt_features, read_labels
from train_cnn import load_cnn_checkpoint


def song_id(wav: Path) -> str:
    # "Debussy - Arabesque-4.wav" -> "Debussy - Arabesque"
    return re.sub(r"-\d+$", "", wav.stem)


def model_classes(path):
    ckpt = torch.load(path, map_location="cpu")
    if "classes" in ckpt:
        return [str(c) for c in ckpt["classes"]], ckpt.get("task") == "multilabel"
    # single-label checkpoints store the classes in the label encoder next to them
    enc = Path(path).with_name("label_encoder.pkl")
    return [str(c) for c in joblib.load(enc if enc.exists() else "label_encoder.pkl").classes_], False


def tune(scores, y_true, grid):
    """Best threshold per instrument by F1; 0.5 if the instrument never occurs in the tuning half."""
    out = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            out.append(0.5)
            continue
        out.append(max(grid, key=lambda t: f1_score(y_true[:, c], scores[:, c] >= t, zero_division=0)))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="checkpoints (.pt) to evaluate")
    parser.add_argument("--test_dir", default="IRMAS-TestingData")
    parser.add_argument("--hop", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0, help="which songs go to the threshold half")
    parser.add_argument("--out", default="results_multilabel/irmas_test_scores.npz")
    args = parser.parse_args()

    models = {}
    for p in args.models:
        classes, multilabel = model_classes(p)
        models[p] = (load_cnn_checkpoint(p), classes, multilabel)
    classes = models[args.models[0]][1]
    if any(m[1] != classes for m in models.values()):
        raise SystemExit("all models must predict the same instruments")

    wavs = sorted(Path(args.test_dir).rglob("*.wav"))
    labels = [read_labels(w) for w in wavs]
    y_true = np.array([[c in labs for c in classes] for labs in labels], dtype=int)
    songs = np.array([song_id(w) for w in wavs])
    uniq = np.array(sorted(set(songs)))
    tune_songs = set(np.random.RandomState(args.seed).permutation(uniq)[: len(uniq) // 2])
    in_tune = np.array([s in tune_songs for s in songs])
    if in_tune.all() or not in_tune.any():
        raise SystemExit(f"need at least 2 songs to split into threshold/report halves (found {len(uniq)})")
    print(f"{len(wavs)} excerpts from {len(uniq)} songs | threshold half: {in_tune.sum()} excerpts, "
          f"report half: {(~in_tune).sum()} | excerpts with none of {classes}: {(y_true.sum(1) == 0).sum()}")

    # per-excerpt scores, aggregated over windows by mean and by max
    agg = {p: {"mean": [], "max": []} for p in models}
    chunk = 64
    with tqdm(total=len(wavs), desc="Scoring excerpts", unit="excerpt") as bar:
        for i in range(0, len(wavs), chunk):
            feats = Parallel(n_jobs=-1)(delayed(excerpt_features)(w, args.hop, True, False) for w in wavs[i:i + chunk])
            for mels, _ in feats:
                x = torch.from_numpy(mels).unsqueeze(1)
                for p, (model, _, multilabel) in models.items():
                    with torch.no_grad():
                        logits = model(x)
                    s = (torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)).numpy()
                    agg[p]["mean"].append(s.mean(0))
                    agg[p]["max"].append(s.max(0))
            bar.update(len(wavs[i:i + chunk]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, files=np.array([str(w) for w in wavs]), y_true=y_true, in_tune=in_tune,
             classes=np.array(classes), **{f"{Path(p).parent.name}_{Path(p).stem}_{a}": np.array(v)
                                            for p in models for a, v in agg[p].items()})

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    summary = []
    for p, (_, _, multilabel) in models.items():
        # choose aggregation + thresholds on the tuning half only
        best = None
        for a in ("mean", "max"):
            s = np.array(agg[p][a])
            th = tune(s[in_tune], y_true[in_tune], grid)
            f = f1_score(y_true[in_tune], s[in_tune] >= th, average="macro", zero_division=0)
            if best is None or f > best[0]:
                best = (f, a, th)
        _, a, th = best
        s = np.array(agg[p][a])[~in_tune]
        yt = y_true[~in_tune]
        pred = s >= th
        mAP = np.mean([average_precision_score(yt[:, c], s[:, c]) for c in range(len(classes))])
        micro = f1_score(yt, pred, average="micro", zero_division=0)
        macro = f1_score(yt, pred, average="macro", zero_division=0)
        none_ok = (pred[yt.sum(1) == 0].sum(1) == 0).mean()

        print(f"\n{'=' * 70}\n{p}  ({'multi-label' if multilabel else 'single-label'}; "
              f"aggregation={a}; thresholds {dict(zip(classes, th.tolist()))})\n{'=' * 70}")
        print(f"Report half: mAP={mAP:.3f} | micro F1={micro:.3f} | macro F1={macro:.3f} | "
              f"'none of the 5' excerpts correctly empty: {none_ok:.0%}")
        print(classification_report(yt, pred, target_names=classes, zero_division=0))
        summary.append((p, mAP, micro, macro, none_ok))

        with open(Path(p).with_name("thresholds.json"), "w") as f:
            json.dump({"aggregation": a, "thresholds": dict(zip(classes, th.tolist())),
                       "tuned_on": f"IRMAS test set, half of the songs (seed {args.seed})",
                       "report_half": {"mAP": mAP, "micro_f1": micro, "macro_f1": macro}}, f, indent=2)

    print(f"\n{'model':60s}  mAP   microF1 macroF1 none-ok")
    for p, mAP, micro, macro, none_ok in summary:
        print(f"{p:60s}  {mAP:.3f}  {micro:.3f}   {macro:.3f}   {none_ok:.0%}")
    print(f"\nPer-excerpt scores saved to {args.out}; thresholds.json written next to each model.")


if __name__ == "__main__":
    main()
