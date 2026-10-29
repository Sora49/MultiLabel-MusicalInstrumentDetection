@echo off
REM Phase 3: multi-label instrument detector (several instruments per window, or none).
REM Needs phase 2 (run_experiments2.bat) to have finished: it starts from those models.
REM   1. from_combined  - starts from the best single-label CNN trained on the 5 instruments
REM   2. from_pretrain  - starts from the network pretrained on all 11 instruments
REM   3. scores both (and the single-label CNN, for comparison) on the real-music IRMAS test set,
REM      choosing thresholds on half of the songs and reporting on the other half
REM Results: results_multilabel\  -  roughly 3.5 hours on CPU.

set OUT=results_multilabel
if not exist %OUT% mkdir %OUT%

python train_multilabel.py --init_from results_improve\combined\cnn_model.pt --out_dir %OUT%\from_combined > %OUT%\from_combined.log 2>&1
python train_multilabel.py --init_from results_improve\combined_pretrain\pretrained_backbone.pt --out_dir %OUT%\from_pretrain > %OUT%\from_pretrain.log 2>&1
python evaluate_multilabel.py --models %OUT%\from_combined\multilabel_model.pt %OUT%\from_pretrain\multilabel_model.pt results_improve\combined\cnn_model.pt > %OUT%\irmas_eval.log 2>&1

echo Done. Real-music results (report half of the IRMAS test set):
findstr /C:"Report half" /C:"aggregation=" %OUT%\irmas_eval.log
