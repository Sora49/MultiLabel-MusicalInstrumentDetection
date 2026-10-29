# Instrument Detector: Results Summary

NNDL project: detecting flute, acoustic guitar, piano, violin and voice in audio using the IRMAS dataset.

## In short

The final model correctly identifies the instrument in **84.7%** of 633 held-out test clips
(the best single-label version reaches **86.4%**). By instrument: voice 93.5%, acoustic guitar
89.8%, piano 84.0%, violin 75.9% and flute 74.7%. Voice is the easiest; flute and violin are the
hardest, mostly because they get confused with each other. On real songs where several
instruments play at once, it scores micro F1 **0.68** / macro F1 **0.60**: strongest on voice,
weakest on guitar hidden under singing.

## How good is that?

- **Vs chance:** random guessing among 5 instruments gives 20%. 84–86% shows the model has
  learned the instruments.
- **Vs a classic baseline:** MFCC features + Random Forest on the same data get 64.6%, about
  20 points lower.
- **Vs a large pretrained model:** PANNs CNN14 (trained on ~2 million AudioSet clips) + logistic
  regression gets 89.6%. Our model is within 3–5 points of it while being trained from scratch
  on ~2,000 clips on a laptop CPU.
- **On real songs:** micro F1 0.68 / macro F1 0.60 is in the range published IRMAS models
  report (roughly 0.55–0.65). Those use 11 instruments, so the comparison isn't exact.

## Accuracy by instrument

Held-out test split: 633 clean clips, one instrument each, from recordings never seen in
training. Accuracy = share of that instrument's clips named correctly.

### Final model (multi-label, `results_multilabel/from_combined`), used for uploaded audio

| Instrument | Accuracy (correct / total) |
|---|---|
| Flute | 74.7% (68 / 91) |
| Acoustic guitar | 89.8% (114 / 127) |
| Piano | 84.0% (121 / 144) |
| Violin | 75.9% (88 / 116) |
| Voice | 93.5% (145 / 155) |
| **Overall** | **84.7% (536 / 633)** |

Main confusions: flute → violin (10) or guitar (9), violin → guitar (11) or flute (8),
piano → guitar (15).

### Best single-label model (`cnn_model.pt`, 11-instrument pretraining)

| Instrument | Accuracy (correct / total) | Precision | F1 |
|---|---|---|---|
| Flute | 78.0% (71 / 91) | 0.83 | 0.80 |
| Acoustic guitar | 85.0% (108 / 127) | 0.86 | 0.86 |
| Piano | 84.7% (122 / 144) | 0.87 | 0.86 |
| Violin | 84.5% (98 / 116) | 0.82 | 0.83 |
| Voice | 95.5% (148 / 155) | 0.91 | 0.93 |
| **Overall** | **86.4% (547 / 633)** | macro 0.86 | macro 0.86 |

### Final model on real songs (IRMAS test set)

Excerpts often contain several instruments, so these are detection scores, not accuracy.
Thresholds were chosen on half of the 208 songs; scores are on the other half (1,455 excerpts).

| Instrument | Precision | Recall | F1 |
|---|---|---|---|
| Flute | 0.51 | 0.57 | 0.54 |
| Acoustic guitar | 0.72 | 0.32 | 0.44 |
| Piano | 0.71 | 0.50 | 0.58 |
| Violin | 0.76 | 0.43 | 0.55 |
| Voice | 0.94 | 0.80 | 0.87 |
| **Overall** | micro 0.80 | micro 0.59 | **micro 0.68 · macro 0.60** |

It correctly reports nothing on **74%** of excerpts that contain none of the 5 instruments
(the single-label model managed only 23%).

## Model

- **Input:** 3-second windows → log-mel spectrogram (128 × 130), scaled to [0, 1].
- **Network:** 4 blocks of [3×3 conv → batch norm → ReLU] × 2 + max-pool, with
  32 → 64 → 128 → 256 channels, then global average pooling, dropout 0.3 and a linear layer to
  5 outputs. 1.17 million parameters.
- **Output:** an independent sigmoid score per instrument, so a window can contain several
  instruments or none. Per-instrument thresholds decide "present".
- **Training:** pretraining on all 11 IRMAS instruments → single-label training on the 5
  (AdamW, warm-up + cosine learning rate, class weights, mixup, stereo views) → multi-label
  training on random mixtures of 1–3 clips, with the 6 other instruments as
  "none of the 5" examples.

## What was done, step by step

1. **Starting point:** MFCC + Random Forest and a small CNN on 3 instruments (flute, piano,
   violin).
2. **Fair split by recording** (no recording in both train and test).
   → Random Forest 72%, CNN 85%.
3. **Stable training:** validation set, best-epoch selection, decaying learning rate
   (1e-3 → 3e-4). → CNN steady at 84.9%.
4. **Expanded to 5 instruments** (+ acoustic guitar, voice). → Random Forest 64.6%, CNN 76.9%.
5. **Class weights** for rare instruments. → 78.5% (flute 51% → 66%). SpecAugment made it
   worse (70.9–75%).
6. **Real-music test (IRMAS test set):** found the model gives one answer only and invents
   instruments. → 75% on single-instrument songs, micro F1 0.61 on layered ones.
7. **Pretrained comparison (PANNs CNN14):** 89.6%, used as a reference only.
8. **Deeper CNN with global pooling** (7× fewer parameters). → 84.4%.
9. **Tested 7 improvements one at a time.** → Stereo views and a 2× wider network: 86.6% each.
   Crop, light SpecAugment, mixup and delta features: no gain.
10. **Combined the helpful ones + 11-instrument pretraining.** → 86.4%, best validation
    (0.889); became the single-label main model.
11. **Switched to multi-label detection** (yes/no per instrument, mixture training,
    timeline mode). → Real music mAP 0.54 → 0.66, macro F1 0.52 → 0.60, silent on 74% of
    songs without the 5 instruments (was 23%); clean clips still 84.7%.
12. **Tested on own songs** (Einaudi, Spirited Away flute, Moonlike Smile, Saturn): sensible
    detections, but too many "none of the 5" gaps, because the thresholds were tuned for whole
    excerpts rather than 3-second windows.

## Limitations and next steps

- Timeline is too strict: re-tune the thresholds for 3-second windows and smooth the scores over
  time (no retraining needed).
- Acoustic guitar under vocals is often missed (recall 0.32).
- Only 5 instruments; other sounds are reported as "none of the 5" rather than named.
- Later: a `--pretrain_extra_only` option to fine-tune on other instrument sets without
  leaking test clips; an ensemble of the two multi-label models (mAP 0.676).
