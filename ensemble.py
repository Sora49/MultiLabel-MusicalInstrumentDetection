"""
Average the predictions of several trained CNN runs.

Each run folder must contain the probs.npz written by train_cnn.py, and all
runs must share the same data split: train them with the same --seed and a
different --init_seed each.

Usage:
    python ensemble.py results/run_a results/run_b results/run_c
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="folders containing probs.npz")
    args = parser.parse_args()

    runs = [np.load(Path(d) / "probs.npz", allow_pickle=True) for d in args.run_dirs]
    ref = runs[0]
    for d, r in zip(args.run_dirs, runs):
        if not (np.array_equal(r["val_labels"], ref["val_labels"])
                and np.array_equal(r["test_labels"], ref["test_labels"])):
            raise SystemExit(f"{d} was trained on a different split; use the same --seed for every run")

    classes = ref["classes"]
    print(f"{'run':45s} val_acc  test_acc")
    for d, r in zip(args.run_dirs, runs):
        va = (r["val_probs"].argmax(1) == r["val_labels"]).mean()
        ta = (r["test_probs"].argmax(1) == r["test_labels"]).mean()
        print(f"{d:45s} {va:.3f}    {ta:.3f}")

    # average the softmax outputs, then take the most likely class
    val_pred = np.mean([r["val_probs"] for r in runs], axis=0).argmax(1)
    test_pred = np.mean([r["test_probs"] for r in runs], axis=0).argmax(1)
    print(f"{'ENSEMBLE (' + str(len(runs)) + ' runs)':45s} "
          f"{(val_pred == ref['val_labels']).mean():.3f}    {(test_pred == ref['test_labels']).mean():.3f}")

    print("\n=== Ensemble Classification Report (test set) ===")
    print(classification_report(ref["test_labels"], test_pred, target_names=classes))
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(classes)
    print(confusion_matrix(ref["test_labels"], test_pred))


if __name__ == "__main__":
    main()
