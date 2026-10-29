"""
Evaluate a saved CNN on the held-out test split, without retraining.

Rebuilds exactly the same train/val/test split as train_cnn.py (same --seed),
loads the saved weights, and reports validation + test results.

Usage:
    python evaluate_model.py --model results_improve\\width32\\cnn_model.pt
    python evaluate_model.py                     (main model, cnn_model.pt)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from features import group_train_test_split
from train_cnn import SpectrogramDataset, evaluate, load_cnn_checkpoint, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_model.pt")
    parser.add_argument("--data_dir", default="IRMAS-TrainingData")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="must match the --seed used in training")
    args = parser.parse_args()

    model = load_cnn_checkpoint(args.model)
    ckpt = torch.load(args.model, map_location="cpu")
    print(f"Model: {args.model} | arch={ckpt.get('arch', 'small')} {ckpt.get('model_kwargs', {})} "
          f"| saved from epoch {ckpt.get('best_epoch', '?')}")

    X, y, groups, class_names, _ = load_dataset(Path(args.data_dir))
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # same split as train_cnn.py
    train_idx, test_idx = group_train_test_split(y_enc, groups, args.test_size, args.seed)
    sub_train, sub_val = group_train_test_split(y_enc[train_idx], groups[train_idx], args.val_size, args.seed)
    val_idx = train_idx[sub_val]

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    _, val_acc, _, _ = evaluate(model, DataLoader(SpectrogramDataset(X, y_enc, val_idx), batch_size=64), criterion, device)
    _, test_acc, probs, labels = evaluate(model, DataLoader(SpectrogramDataset(X, y_enc, test_idx), batch_size=64), criterion, device)
    preds = probs.argmax(1)

    print(f"\nValidation accuracy: {val_acc:.3f} ({len(val_idx)} clips)")
    print(f"Test accuracy:       {test_acc:.3f} ({len(test_idx)} clips)")
    print("\n=== Classification Report (test set) ===")
    print(classification_report(labels, preds, target_names=le.classes_))
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(le.classes_)
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
