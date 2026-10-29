"""
Chop raw audio files into fixed-length segments, organized by instrument label.

Expected input structure (raw, possibly long clips):
    raw_audio/
        guitar/
            song1.wav
            song2.wav
        piano/
            song1.wav
        ...

Produces:
    dataset/
        guitar/
            song1_000.wav
            song1_001.wav
            ...
        piano/
            song1_000.wav
        ...
"""

import argparse
import os
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def chop_file(filepath: Path, out_dir: Path, segment_seconds: float, sr: int):
    try:
        y, orig_sr = librosa.load(str(filepath), sr=sr, mono=True)
    except Exception as e:
        print(f"  [skip] Could not load {filepath.name}: {e}")
        return 0

    segment_len = int(segment_seconds * sr)
    n_segments = len(y) // segment_len
    if n_segments == 0:
        print(f"  [skip] {filepath.name} shorter than one segment")
        return 0

    stem = filepath.stem
    count = 0
    for i in range(n_segments):
        chunk = y[i * segment_len:(i + 1) * segment_len]
        # skip near-silent chunks
        if chunk.max() - chunk.min() < 1e-4:
            continue
        out_path = out_dir / f"{stem}_{i:03d}.wav"
        sf.write(str(out_path), chunk, sr)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Chop audio into labeled fixed-length segments.")
    parser.add_argument("--input_dir", required=True, help="Folder of raw audio, one subfolder per instrument")
    parser.add_argument("--output_dir", required=True, help="Where to write chopped segments")
    parser.add_argument("--segment_seconds", type=float, default=3.0)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    labels = [d for d in input_dir.iterdir() if d.is_dir()]
    if not labels:
        raise SystemExit(f"No subfolders found in {input_dir}. Expected one folder per instrument.")

    total = 0
    for label_dir in labels:
        out_label_dir = output_dir / label_dir.name
        out_label_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in label_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS]
        print(f"\n{label_dir.name}: {len(files)} source files")
        for f in tqdm(files, desc=label_dir.name):
            total += chop_file(f, out_label_dir, args.segment_seconds, args.sr)

    print(f"\nDone. Wrote {total} segments to {output_dir}")


if __name__ == "__main__":
    main()
