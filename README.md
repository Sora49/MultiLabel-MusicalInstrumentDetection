# Multi-Label Musical Instrument Detection

Upload any piece of music and find out **which instruments are playing, how much of the track
each one is in, and when**. A CNN trained on log-mel spectrograms detects five instruments:
**flute, acoustic guitar, piano, violin and voice**. Each 3-second window can contain several of
them, or none.

Neural Networks & Deep Learning course project, built on the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset.

## Results

| Test | Final model |
|---|---|
| Clean single-instrument clips (633 held-out clips) | **84.7 %** accuracy (best single-label version: 86.4 %) |
| Real songs, several instruments at once (IRMAS test set) | **micro F1 0.68 · macro F1 0.60** |
| Songs with none of the 5 instruments | correctly reports nothing **74 %** of the time |

Per instrument (clean clips): voice 93.5 % · acoustic guitar 89.8 % · piano 84.0 % ·
violin 75.9 % · flute 74.7 %.

For comparison, an MFCC + Random Forest baseline gets 64.6 %, and a large pretrained audio model
(PANNs CNN14) gets 89.6 %. The full write-up, with every experiment, is in
[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

## How it works

```
audio ─► 3 s windows ─► log-mel spectrogram (128 × 130) ─► CNN ─► 5 sigmoid scores ─► thresholds ─► timeline
```

- **Network:** 4 blocks of [3×3 conv → batch norm → ReLU] × 2 + max-pool
  (32 → 64 → 128 → 256 channels), global average pooling, dropout, linear layer → 5 outputs.
  1.17 M parameters.
- **Training:** single-label training on IRMAS clips (AdamW, warm-up + cosine learning rate,
  class weights, mixup, stereo channels as extra views), then multi-label training on random
  mixtures of 1–3 clips, with 6 other instruments mixed in as "none of the 5".
- **Evaluation:** data is split by recording, so no recording appears in both training and test.
  On the IRMAS test set, per-instrument thresholds are chosen on half of the songs and all scores
  are reported on the other half.

## Quick start: use the trained model

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload an mp3/wav/flac file to see per-instrument shares, a timeline, scores over time and a
downloadable CSV. Or from the command line:

```bash
python predict.py --file song.mp3 --timeline --model_path results_multilabel/from_combined/multilabel_model.pt --smooth 3
```

The trained models are included:

| File | Model |
|---|---|
| `results_multilabel/from_combined/multilabel_model.pt` | final multi-label detector (default in the app) |
| `results_multilabel/from_pretrain/multilabel_model.pt` | multi-label detector started from the 11-instrument pretraining |
| `cnn_model.pt` + `label_encoder.pkl` | best single-label CNN (86.4 %) |

Each multi-label model has its tuned per-instrument thresholds in `thresholds.json`.

## Reproducing the training

The IRMAS audio is **not included** (its licence doesn't allow redistribution).

1. Download *IRMAS-TrainingData* (and, for the real-music evaluation, the three
   *IRMAS-TestingData* parts) from [Zenodo](https://zenodo.org/records/1290750).
2. Arrange the folders like this:
   ```
   IRMAS-TrainingData/  flu/  gac/  pia/  vio/  voi/        (the 5 target instruments)
   IRMAS-unused/        cel/  cla/  gel/  org/  sax/  tru/  (the other 6: pretraining + negatives)
   IRMAS-TestingData/   (the unzipped test parts)
   ```
3. Run the pipeline (Windows batch files; each line is a plain `python` command you can also run
   on its own):

| Step | Command | What it does |
|---|---|---|
| Baseline | `python train_baseline.py --data_dir IRMAS-TrainingData` | MFCC + Random Forest |
| Screening | `run_experiments.bat` | deep CNN + one improvement at a time |
| Best single-label | `run_experiments2.bat` | combined improvements + 11-instrument pretraining |
| Multi-label | `run_multilabel.bat` | two multi-label detectors + real-music evaluation |
| Progress | `check_progress.bat` | live progress of whichever run is active |

Everything runs on CPU. The full pipeline takes several hours on a laptop. Features are cached
in `~/.cache/instrument_cnn`.

## Files

| File | Purpose |
|---|---|
| `features.py` | audio loading, log-mel features, recording-level split |
| `train_cnn.py` | single-label CNN; all improvements are opt-in flags (`--help`) |
| `train_multilabel.py` | multi-label detector trained on synthetic mixtures |
| `train_baseline.py`, `train_pretrained.py` | Random Forest baseline, PANNs comparison |
| `evaluate_model.py` | re-evaluate a saved CNN on the held-out split |
| `evaluate_multilabel.py` | real-music detection scores + threshold tuning on IRMAS test set |
| `evaluate_testset.py` | earlier single-label evaluation on the IRMAS test set |
| `ensemble.py` | average several runs' predictions |
| `predict.py` | single clip, or full-file timeline (`--timeline`) |
| `app.py` | Streamlit app |
| `data_prep.py` | chop long recordings into 3 s clips (not needed for IRMAS) |

## Limitations

- Only 5 instruments. Anything else is reported as "none of the 5" rather than named.
- The thresholds were tuned on whole excerpts, so the timeline can show gaps in quiet or
  layered passages. `--smooth 3` (or the app's smoothing setting) helps.
- Acoustic guitar under a singer is often missed (recall about 0.32).

## Credits

- Dataset: J. J. Bosch, J. Janer, F. Fuhrmann, P. Herrera, *A Comparison of Sound Segregation
  Techniques for Predominant Instrument Recognition in Musical Audio Signals*, ISMIR 2012 (IRMAS).
- Comparison model: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
  Pattern Recognition*, 2020.

<!-- Experiment 2: Git and CI/CD workflow practice -->
