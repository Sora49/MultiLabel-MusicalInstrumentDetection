"""
Multi-label instrument detector: for each of the 5 instruments, an independent
"is it playing?" probability (sigmoid + binary cross-entropy), so a window can
contain several instruments, or none of them.

IRMAS training clips each have one predominant instrument, so layered examples
are made on the fly: 1-3 clips are mixed at random volumes (in the mel-power
domain, where sources add), and the label is the union of their instruments.
Clips of the 6 unused instruments are mixed in too, labelled as none of the 5,
so the model learns to ignore other sounds instead of forcing them into a class.

Splits: the 5 target instruments use exactly the same train/val/test split as
train_cnn.py; IRMAS-unused is split the same way (by recording). Validation
mixtures are fixed, so the val score is comparable across epochs.

Usage (start from a trained single-label CNN or a pretrained backbone):
    python train_multilabel.py --init_from results_improve\\combined\\cnn_model.pt --out_dir results_multilabel\\from_combined
    python train_multilabel.py --init_from results_improve\\combined_pretrain\\pretrained_backbone.pt --out_dir results_multilabel\\from_pretrain
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from features import (
    N_MELS, group_train_test_split, load_fixed_length, mel_power_from_audio, normalize_mel_power, recording_id,
)
from train_cnn import ARCHS, AUDIO_EXTS, CACHE_DIR, cached_features, make_optimizer


def load_power_dataset(data_dir: Path):
    """Un-normalised mel power per clip (cached), in the same file order as train_cnn.load_dataset."""
    labels = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    files, y = [], []
    for label in labels:
        for f in (data_dir / label).iterdir():
            if f.suffix.lower() in AUDIO_EXTS:
                files.append(f)
                y.append(label)
    names = np.array([f"{f.parent.name}/{f.name}" for f in files])
    groups = np.array([recording_id(f) for f in files])
    P = cached_features(CACHE_DIR / f"mel_{data_dir.resolve().name}_power.npz", files, names,
                        lambda p: mel_power_from_audio(load_fixed_length(p)), f"mel power ({data_dir.name})")
    return P, np.array(y), groups, labels


def split(y_enc, groups, test_size, val_size, seed):
    train_idx, test_idx = group_train_test_split(y_enc, groups, test_size, seed)
    sub_train, sub_val = group_train_test_split(y_enc[train_idx], groups[train_idx], val_size, seed)
    return train_idx[sub_train], train_idx[sub_val], test_idx


class MixtureDataset(Dataset):
    """
    Each item mixes 1..max_sources clips: each source is an unknown-instrument
    clip with probability p_unknown, else a target-instrument clip, at a random
    gain in [min_gain_db, 0] dB. Target = multi-hot over the target instruments
    present. With n_fixed set, the mixtures are drawn once (for validation/test).
    """

    def __init__(self, P_t, y_t, idx_t, P_u, idx_u, n_classes, n_items, max_sources=3,
                 p_unknown=0.3, min_gain_db=-12.0, n_fixed=None, seed=0):
        self.P_t, self.y_t, self.idx_t = P_t, y_t, np.asarray(idx_t)
        self.P_u, self.idx_u = P_u, np.asarray(idx_u)
        self.n_classes, self.n_items = n_classes, n_items
        self.max_sources, self.p_unknown, self.min_gain_db = max_sources, p_unknown, min_gain_db
        self.recipes = None
        if n_fixed:
            rng = np.random.RandomState(seed)
            self.recipes = [self._recipe(rng) for _ in range(n_fixed)]
            self.n_items = n_fixed

    def _recipe(self, rng):
        k = rng.randint(1, self.max_sources + 1)
        sources = []
        for _ in range(k):
            if len(self.idx_u) and rng.rand() < self.p_unknown:
                sources.append(("u", int(rng.choice(self.idx_u))))
            else:
                sources.append(("t", int(rng.choice(self.idx_t))))
        gains = rng.uniform(self.min_gain_db, 0.0, size=k)
        return sources, gains

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        if self.recipes is not None:
            sources, gains = self.recipes[i]
        else:
            # torch's RNG, so torch.manual_seed makes training reproducible
            seed = int(torch.randint(0, 2**31 - 1, ()))
            sources, gains = self._recipe(np.random.RandomState(seed))
        mix = np.zeros_like(self.P_t[0])
        target = np.zeros(self.n_classes, dtype=np.float32)
        for (pool, j), g in zip(sources, gains):
            mix += (10 ** (g / 10)) * (self.P_t[j] if pool == "t" else self.P_u[j])
            if pool == "t":
                target[self.y_t[j]] = 1.0
        x = torch.from_numpy(normalize_mel_power(mix)).unsqueeze(0)
        return x, torch.from_numpy(target)


def build_model(args, n_classes, time_steps, device):
    """Fresh model, or one initialised from a checkpoint (head replaced if its size differs)."""
    if not args.init_from:
        kwargs = {"width": args.width, "deltas": False}
        return ARCHS["deep"](n_classes, N_MELS, time_steps, **kwargs).to(device), "deep", kwargs
    ckpt = torch.load(args.init_from, map_location="cpu")
    arch, kwargs = ckpt.get("arch", "small"), ckpt.get("model_kwargs", {})
    if arch != "deep":
        raise SystemExit("--init_from needs a deep-architecture checkpoint")
    model = ARCHS[arch](ckpt["n_classes"], ckpt["n_mels"], ckpt["time_steps"], **kwargs)
    model.load_state_dict(ckpt["model_state"])
    if ckpt["n_classes"] != n_classes:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    print(f"Initialised from {args.init_from} ({ckpt['n_classes']} outputs, arch {arch} {kwargs})")
    return model.to(device), arch, kwargs


def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, t in loader:
            probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            targets.append(t.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def macro_ap(probs, targets):
    return float(np.mean([average_precision_score(targets[:, c], probs[:, c]) for c in range(targets.shape[1])]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="IRMAS-TrainingData")
    parser.add_argument("--unknown_dir", default="IRMAS-unused", help="non-target instruments, used as negatives")
    parser.add_argument("--init_from", default=None, help="cnn_model.pt or pretrained_backbone.pt to start from")
    parser.add_argument("--width", type=int, default=16, help="only used without --init_from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch_size", type=int, default=4000, help="random mixtures per epoch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="1e-4 suits fine-tuning; use ~3e-4 from scratch")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--max_sources", type=int, default=3)
    parser.add_argument("--p_unknown", type=float, default=0.3, help="chance each mixed source is a non-target instrument")
    parser.add_argument("--min_gain_db", type=float, default=-12.0)
    parser.add_argument("--n_val_mixtures", type=int, default=1500)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="data split (keep 42 to match the other scripts)")
    parser.add_argument("--init_seed", type=int, default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print("Config: " + ", ".join(f"{k}={v}" for k, v in vars(args).items()))

    torch.manual_seed(args.seed if args.init_seed is None else args.init_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P_t, y_t, g_t, classes = load_power_dataset(Path(args.data_dir))
    y_t = LabelEncoder().fit_transform(y_t)
    tr_t, va_t, te_t = split(y_t, g_t, args.test_size, args.val_size, args.seed)
    P_u, y_u, g_u, u_classes = load_power_dataset(Path(args.unknown_dir))
    if set(u_classes) & set(classes):
        raise SystemExit(f"{args.unknown_dir} overlaps the target instruments: {set(u_classes) & set(classes)}")
    tr_u, va_u, te_u = split(LabelEncoder().fit_transform(y_u), g_u, args.test_size, args.val_size, args.seed)
    print(f"Targets {classes}: train {len(tr_t)} | val {len(va_t)} | test {len(te_t)} clips (same split as train_cnn.py)")
    print(f"Negatives {u_classes}: train {len(tr_u)} | val {len(va_u)} | test {len(te_u)} clips")
    if args.init_from and "pretrained_backbone" in args.init_from:
        print("Note: this backbone was pretrained on all IRMAS-unused clips, so the val/test negatives "
              "are not unseen for it; judge it on the IRMAS test set (evaluate_multilabel.py).")

    n = len(classes)
    mk = dict(n_classes=n, max_sources=args.max_sources, p_unknown=args.p_unknown, min_gain_db=args.min_gain_db)
    train_ds = MixtureDataset(P_t, y_t, tr_t, P_u, tr_u, n_items=args.epoch_size, **mk)
    val_ds = MixtureDataset(P_t, y_t, va_t, P_u, va_u, n_items=0, n_fixed=args.n_val_mixtures, seed=1, **mk)
    test_ds = MixtureDataset(P_t, y_t, te_t, P_u, te_u, n_items=0, n_fixed=2000, seed=2, **mk)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    time_steps = P_t.shape[2]
    model, arch, model_kwargs = build_model(args, n, time_steps, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer, scheduler = make_optimizer(model, args, args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_map": []}
    best_map, best_epoch, best_state = -1.0, 0, None
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training multi-label", unit="epoch"):
        model.train()
        running = 0.0
        for x, t in tqdm(train_loader, desc=f"  Epoch {epoch}/{args.epochs}", unit="batch", leave=False):
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), t)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        probs, targets = predict(model, val_loader, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(probs), torch.from_numpy(targets)))
        val_map = macro_ap(probs, targets)
        history["train_loss"].append(running / len(train_ds))
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)
        marker = ""
        if val_map > best_map:
            best_map, best_epoch, best_state = val_map, epoch, copy.deepcopy(model.state_dict())
            marker = "  <- best so far"
        tqdm.write(f"Epoch {epoch:2d}/{args.epochs} | train_loss={history['train_loss'][-1]:.4f} | "
                   f"val_loss={val_loss:.4f} | val_mAP={val_map:.3f}{marker}")

    model.load_state_dict(best_state)
    probs, targets = predict(model, test_loader, device)
    print(f"\nBest epoch: {best_epoch} (val mAP={best_map:.3f})")
    print(f"Test mixtures (own split, 2000 synthetic): mAP={macro_ap(probs, targets):.3f}")
    for c, name in enumerate(classes):
        print(f"  {name}: AP={average_precision_score(targets[:, c], probs[:, c]):.3f}")
    print("\nAt threshold 0.5 (tune per-instrument thresholds with evaluate_multilabel.py):")
    print(classification_report(targets, probs >= 0.5, target_names=classes, zero_division=0))

    torch.save({
        "task": "multilabel", "model_state": model.state_dict(), "arch": arch, "model_kwargs": model_kwargs,
        "n_classes": n, "classes": list(classes), "n_mels": N_MELS, "time_steps": time_steps,
        "best_epoch": best_epoch, "init_from": args.init_from,
    }, out_dir / "multilabel_model.pt")
    print(f"Saved {out_dir / 'multilabel_model.pt'}")

    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("BCE loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_map"])
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    plt.title("Validation mAP (synthetic mixtures)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png")


if __name__ == "__main__":
    main()
