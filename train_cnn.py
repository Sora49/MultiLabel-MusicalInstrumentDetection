"""
CNN instrument classifier: mel-spectrogram "image" -> CNN (PyTorch).

Usage:
    python train_cnn.py --data_dir IRMAS-TrainingData

Every improvement below is off by default, so the plain command reproduces the
reference deep-CNN run; turn them on individually to compare:
    --weight_decay 1e-4 --warmup 3 --clip_grad 1.0    steadier optimisation
    --crop_frames 108                                  random 2.5 s crops
    --mixup 0.4                                        blend pairs of clips
    --augment --n_masks 1 --mask_freq 8 --mask_time 10 light SpecAugment
    --stereo                                           left/right channels as extra views
    --deltas                                           add delta + delta-delta input channels
    --width 32                                         wider network
    --pretrain_dir IRMAS-unused --pretrain_epochs 20   pretrain on 11 instruments first
                                                       (saves pretrained_backbone.pt)
    --init_from <dir>\\pretrained_backbone.pt          fine-tune from a saved pretraining
"""

import argparse
import copy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from features import (
    N_MELS, extract_mel_spectrogram, group_train_test_split, load_fixed_length_stereo,
    mel_spectrogram_from_audio, recording_id,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
# extracted features are cached here (outside OneDrive, they're a few hundred MB)
CACHE_DIR = Path.home() / ".cache" / "instrument_cnn"


def spec_augment(spec, n_masks=2, max_freq=16, max_time=20):
    """
    SpecAugment-style masking: blank out a few random bands of mel bins and of
    time frames, so the model can't rely on any single narrow region.
    spec: (1, n_mels, time), values in [0, 1]; masked regions are set to 0.
    """
    spec = spec.clone()
    _, n_mels, n_time = spec.shape
    for _ in range(n_masks):
        f = int(torch.randint(0, max_freq + 1, ()))
        f0 = int(torch.randint(0, n_mels - f + 1, ()))
        spec[:, f0:f0 + f, :] = 0
        t = int(torch.randint(0, max_time + 1, ()))
        t0 = int(torch.randint(0, n_time - t + 1, ()))
        spec[:, :, t0:t0 + t] = 0
    return spec


class SpectrogramDataset(Dataset):
    """
    Serves X[idx[k]]. Augmentations (training set only):
      X_lr     - (N, 2, n_mels, time) per-channel spectrograms; each draw picks
                 the mono, left or right version at random
      crop     - random crop of this many time frames
      specaug  - (n_masks, max_freq, max_time) for spec_augment, or None
    """

    def __init__(self, X, y, idx, X_lr=None, crop=0, specaug=None):
        self.X, self.y, self.idx = X, y, np.asarray(idx)
        self.X_lr, self.crop, self.specaug = X_lr, crop, specaug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        spec = self.X[i]
        if self.X_lr is not None:
            c = int(torch.randint(0, 3, ()))
            if c > 0:
                spec = self.X_lr[i, c - 1]
        spec = torch.tensor(spec).unsqueeze(0)  # add channel dim -> (1, n_mels, time)
        if self.crop:
            t0 = int(torch.randint(0, spec.shape[-1] - self.crop + 1, ()))
            spec = spec[..., t0:t0 + self.crop]
        if self.specaug:
            spec = spec_augment(spec, *self.specaug)
        return spec, torch.tensor(self.y[i], dtype=torch.long)


class InstrumentCNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int, time_steps: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # figure out flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, time_steps)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def time_delta(x):
    """Central difference along the time axis, same shape as x."""
    # replicate padding on a 4D tensor must cover the last two dims; pad time only
    padded = F.pad(x, (1, 1, 0, 0), mode="replicate")
    return (padded[..., 2:] - padded[..., :-2]) / 2


class DeepInstrumentCNN(nn.Module):
    """
    Four double-conv blocks, then global average pooling instead of flattening.
    Pooling over the whole map keeps the classifier tiny (channels -> classes)
    rather than a ~130k-input dense layer, which is what let the small model
    memorise the training clips. It also accepts any clip length.

    deltas=True stacks the spectrogram with its first and second time
    differences as 3 input channels (computed inside the model, so callers
    still pass a plain spectrogram).
    """

    def __init__(self, n_classes: int, n_mels: int = None, time_steps: int = None,
                 width: int = 16, deltas: bool = False):
        super().__init__()
        self.deltas = deltas
        self.conv = nn.Sequential(
            conv_block(3 if deltas else 1, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 8),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 8, n_classes),
        )

    def forward(self, x):
        if self.deltas:
            d = time_delta(x)
            x = torch.cat([x, d, time_delta(d)], dim=1)
        return self.classifier(self.conv(x))


ARCHS = {"small": InstrumentCNN, "deep": DeepInstrumentCNN}


def load_cnn_checkpoint(path: str):
    """Rebuild the right architecture from a saved checkpoint, in eval mode."""
    checkpoint = torch.load(path, map_location="cpu")
    model = ARCHS[checkpoint.get("arch", "small")](
        checkpoint["n_classes"], checkpoint["n_mels"], checkpoint["time_steps"],
        **checkpoint.get("model_kwargs", {}),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def cached_features(path: Path, files, names, fn, desc):
    """Compute fn(file) for every file, or load it from path if the file list matches."""
    if path.exists():
        with np.load(path) as d:
            if d["names"].shape == names.shape and (d["names"] == names).all():
                print(f"Loaded cached {desc} from {path}")
                return d["X"]
    X = np.stack([fn(str(f)) for f in tqdm(files, desc=f"Extracting {desc}")])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, names=names)
    return X


def load_dataset(data_dir: Path, stereo: bool = False):
    """
    Returns (X, y, groups, labels, X_lr): mono mel-spectrograms (N, n_mels, time),
    string labels, recording ids, sorted class names, and per-channel
    spectrograms (N, 2, n_mels, time) if stereo else None.
    """
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not labels:
        raise SystemExit(f"No class subfolders found in {data_dir}")
    files, y = [], []
    for label in labels:
        for f in (data_dir / label).iterdir():
            if f.suffix.lower() in AUDIO_EXTS:
                files.append(f)
                y.append(label)
    names = np.array([f"{f.parent.name}/{f.name}" for f in files])
    groups = np.array([recording_id(f) for f in files])

    key = data_dir.resolve().name
    X = cached_features(CACHE_DIR / f"mel_{key}_mono.npz", files, names,
                        extract_mel_spectrogram, f"mel-specs ({data_dir.name})")
    X_lr = None
    if stereo:
        X_lr = cached_features(
            CACHE_DIR / f"mel_{key}_stereo.npz", files, names,
            lambda p: np.stack([mel_spectrogram_from_audio(ch) for ch in load_fixed_length_stereo(p)]),
            f"left/right mel-specs ({data_dir.name})",
        )
    return X, np.array(y), groups, labels, X_lr


def evaluate(model, loader, criterion, device):
    """Returns (mean loss, accuracy, softmax probabilities, true labels) over a loader."""
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            out = model(specs)
            total_loss += criterion(out, labels).item() * specs.size(0)
            all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs, all_labels = np.concatenate(all_probs), np.array(all_labels)
    return total_loss / len(all_labels), (all_probs.argmax(1) == all_labels).mean(), all_probs, all_labels


def make_optimizer(model, args, epochs):
    # plain Adam when weight decay is off, so the reference run is unchanged
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # decay the learning rate smoothly to ~0 so the weights settle in later epochs,
    # optionally after a few epochs of linear warm-up from lr/10
    if args.warmup > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - args.warmup)),
        ], milestones=[args.warmup])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def train_epochs(model, train_loader, epochs, train_criterion, args, device, desc,
                 val_loader=None, criterion=None):
    """
    Train for `epochs`. With a val_loader, keeps the weights from the epoch with
    the best validation accuracy and returns the history; without one (the
    pretraining phase) the final weights are kept.
    """
    optimizer, scheduler = make_optimizer(model, args, epochs)
    mix = torch.distributions.Beta(args.mixup, args.mixup) if args.mixup > 0 else None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, 0, None

    epoch_bar = tqdm(range(1, epochs + 1), desc=desc, unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss, n_seen = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for specs, labels in batch_bar:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            if mix is not None:
                # mixup: train on a blend of two clips, with the loss blended the same way
                lam = float(mix.sample())
                perm = torch.randperm(specs.size(0), device=device)
                out = model(lam * specs + (1 - lam) * specs[perm])
                loss = lam * train_criterion(out, labels) + (1 - lam) * train_criterion(out, labels[perm])
            else:
                out = model(specs)
                loss = train_criterion(out, labels)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
            n_seen += specs.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / n_seen
        history["train_loss"].append(train_loss)
        if val_loader is None:
            tqdm.write(f"{desc} epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f}")
            continue

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # keep a copy of the weights from the best epoch on validation
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best so far"

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.3f}")
        # tqdm.write keeps the per-epoch log lines from breaking the bar
        tqdm.write(f"Epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    # 1e-3 made val accuracy swing between ~53% and ~78% mid-training; 3e-4 is much steadier
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="fraction of the training recordings held out for picking the best epoch")
    parser.add_argument("--seed", type=int, default=42, help="seeds the data split (and the model, unless --init_seed)")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="seed for weights/shuffling/augmentation only; vary this for ensembles so the split stays fixed")
    parser.add_argument("--no_class_weights", action="store_true", help="turn off class-weighted loss")
    parser.add_argument("--out_dir", default=".", help="where to save the model, encoder, plot and probabilities")
    parser.add_argument("--arch", choices=list(ARCHS), default="deep")
    parser.add_argument("--width", type=int, default=16, help="deep arch: channels in the first block")
    parser.add_argument("--deltas", action="store_true", help="deep arch: add delta/delta-delta input channels")
    # optimisation
    parser.add_argument("--weight_decay", type=float, default=0.0, help="> 0 switches Adam to AdamW")
    parser.add_argument("--warmup", type=int, default=0, help="epochs of linear learning-rate warm-up")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="max gradient norm (0 = off)")
    # augmentation (training set only)
    # SpecAugment at 2x16 bins / 2x20 frames lowered test accuracy on the small CNN
    parser.add_argument("--augment", action="store_true", help="turn on SpecAugment masking")
    parser.add_argument("--n_masks", type=int, default=2)
    parser.add_argument("--mask_freq", type=int, default=16)
    parser.add_argument("--mask_time", type=int, default=20)
    parser.add_argument("--crop_frames", type=int, default=0, help="deep arch: train on random crops of this many frames")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a, a) parameter (0 = off)")
    parser.add_argument("--stereo", action="store_true", help="use left/right channels as extra training views")
    # pretraining on other instruments
    parser.add_argument("--pretrain_dir", default=None,
                        help="folder of *other* instruments; pretrain on them + this run's train split first")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--init_from", default=None,
                        help="skip pretraining: start from a pretrained_backbone.pt saved by an earlier "
                             "--pretrain_dir run (same --data_dir and --seed); its architecture is used")
    args = parser.parse_args()

    if args.arch == "small" and (args.crop_frames or args.deltas or args.width != 16 or args.pretrain_dir):
        parser.error("--crop_frames, --deltas, --width and --pretrain_dir need --arch deep")
    if args.init_from and args.pretrain_dir:
        parser.error("use either --pretrain_dir (pretrain now) or --init_from (reuse a saved pretraining), not both")
    print("Config: " + ", ".join(f"{k}={v}" for k, v in vars(args).items()))

    torch.manual_seed(args.seed if args.init_seed is None else args.init_seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, class_names, X_lr = load_dataset(data_dir, stereo=args.stereo)
    print(f"\nLoaded {len(X)} samples from {len(set(groups))} recordings "
          f"across {len(class_names)} classes: {class_names}")

    time_steps = X[0].shape[1]  # all fixed-length so this is constant

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split by recording, not by clip, so no recording appears in two sets.
    # The test split matches train_baseline.py, so both models face the same clips.
    train_idx, test_idx = group_train_test_split(y_enc, groups, args.test_size, args.seed)
    # carve the validation set out of the training recordings; the test set
    # stays untouched until the very end
    sub_train, sub_val = group_train_test_split(y_enc[train_idx], groups[train_idx], args.val_size, args.seed)
    train_idx, val_idx = train_idx[sub_train], train_idx[sub_val]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(groups[splits[a]]) & set(groups[splits[b]]), f"{a}/{b} share a recording"
    print(f"Train: {len(train_idx)} clips | Val: {len(val_idx)} clips | Test: {len(test_idx)} clips "
          f"(no recording shared between splits)")

    specaug = (args.n_masks, args.mask_freq, args.mask_time) if args.augment else None
    train_ds = SpectrogramDataset(X, y_enc, train_idx, X_lr=X_lr, crop=args.crop_frames, specaug=specaug)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(X, y_enc, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(SpectrogramDataset(X, y_enc, test_idx), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"width": args.width, "deltas": args.deltas} if args.arch == "deep" else {}

    def build(n_classes):
        return ARCHS[args.arch](n_classes=n_classes, n_mels=N_MELS, time_steps=time_steps, **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()  # unweighted, for val/test so losses stay comparable

    if args.pretrain_dir:
        # Pretrain on this run's *training* clips plus every clip of the other
        # instruments, as one big 5 + k class problem. Val/test clips are never
        # used, so the comparison stays fair. Then swap the output layer for a
        # fresh 5-class one and train as usual.
        Xp, yp, _, p_labels, _ = load_dataset(Path(args.pretrain_dir))
        if set(p_labels) & set(class_names):
            raise SystemExit(f"--pretrain_dir must hold other instruments; overlap: {set(p_labels) & set(class_names)}")
        X_pre = np.concatenate([X[train_idx], Xp])
        y_pre = np.concatenate([y_enc[train_idx], len(class_names) + LabelEncoder().fit_transform(yp)])
        n_pre = len(class_names) + len(p_labels)
        print(f"Pretraining on {len(X_pre)} clips across {n_pre} classes "
              f"({class_names} train split + {p_labels})")
        model = build(n_pre)
        pre_ds = SpectrogramDataset(X_pre, y_pre, np.arange(len(X_pre)), crop=args.crop_frames, specaug=specaug)
        train_epochs(model, DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True),
                     args.pretrain_epochs, criterion, args, device, desc="Pretraining")
        # save the pretrained network so later runs can fine-tune from it (--init_from)
        # without repeating pretraining; data_dir/seed are recorded so the split can be checked
        torch.save({
            "model_state": model.state_dict(), "arch": args.arch, "model_kwargs": model_kwargs,
            "n_classes": n_pre, "classes": list(class_names) + list(p_labels),
            "n_mels": N_MELS, "time_steps": time_steps,
            "data_dir": data_dir.resolve().name, "seed": args.seed,
            "pretrain_dir": Path(args.pretrain_dir).resolve().name, "pretrain_epochs": args.pretrain_epochs,
        }, out_dir / "pretrained_backbone.pt")
        print(f"Saved the {n_pre}-instrument pretrained network to {out_dir / 'pretrained_backbone.pt'}")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names)).to(device)
        del X_pre, Xp
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        # the pretraining saw this split's training clips, so it must be the same split
        if ckpt.get("data_dir") != data_dir.resolve().name or ckpt.get("seed") != args.seed:
            raise SystemExit(f"{args.init_from} was pretrained with data_dir={ckpt.get('data_dir')}, "
                             f"seed={ckpt.get('seed')}; this run must use the same, or its val/test "
                             f"clips may have been seen during pretraining")
        args.arch, model_kwargs = ckpt["arch"], ckpt["model_kwargs"]
        model = build(ckpt["n_classes"])
        model.load_state_dict(ckpt["model_state"])
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names)).to(device)
        print(f"Starting from pretrained network {args.init_from} "
              f"({ckpt['n_classes']} instruments, {ckpt.get('pretrain_epochs')} epochs)")
    else:
        model = build(len(class_names))
    print(f"Architecture: {args.arch} {model_kwargs} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # weight each class by n_samples / (n_classes * class_count), so rare classes
    # (flute) count for more and big ones (voice) for less in the training loss
    if args.no_class_weights:
        train_criterion = criterion
    else:
        counts = np.bincount(y_enc[train_idx], minlength=len(le.classes_))
        weights = len(train_idx) / (len(counts) * counts)
        train_criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        print("Class weights: " + ", ".join(f"{c}={w:.2f}" for c, w in zip(le.classes_, weights)))

    history, best_epoch, best_val_acc = train_epochs(
        model, train_loader, args.epochs, train_criterion, args, device, desc="Training CNN",
        val_loader=val_loader, criterion=criterion,
    )

    # Final evaluation: best epoch's weights are loaded; touch the test set once
    _, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    _, test_acc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    all_preds = test_probs.argmax(1)
    print(f"\nBest epoch: {best_epoch} (val_acc={best_val_acc:.3f})")
    print(f"Test accuracy with best-epoch weights: {test_acc:.3f}")

    print("\n=== Classification Report (test set) ===")
    print(classification_report(test_labels, all_preds, target_names=le.classes_))
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(le.classes_)
    print(confusion_matrix(test_labels, all_preds))

    torch.save({
        "model_state": model.state_dict(),
        "n_classes": len(class_names),
        "n_mels": N_MELS,
        "time_steps": time_steps,
        "best_epoch": best_epoch,
        "arch": args.arch,
        "model_kwargs": model_kwargs,
    }, out_dir / "cnn_model.pt")
    joblib.dump(le, out_dir / "label_encoder.pkl")
    # per-clip probabilities, so ensemble.py can average several runs
    np.savez(out_dir / "probs.npz", val_probs=val_probs, val_labels=val_labels,
             test_probs=test_probs, test_labels=test_labels, classes=le.classes_)
    print(f"\nSaved cnn_model.pt (epoch {best_epoch} weights), label_encoder.pkl and probs.npz to {out_dir}")

    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"])
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png")
    print(f"Saved training_curve.png to {out_dir}")


if __name__ == "__main__":
    main()
