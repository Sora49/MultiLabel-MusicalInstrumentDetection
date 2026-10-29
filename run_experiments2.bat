@echo off
REM Phase 2: re-run deltas (crashed in phase 1), combine the techniques that
REM helped, then add 11-instrument pretraining on top of the combination.
REM Results go to results_improve\ like phase 1. Roughly 4-5 hours on CPU.

set DATA=--data_dir IRMAS-TrainingData
set STAB=--weight_decay 1e-4 --warmup 3 --clip_grad 1.0
set BEST=%STAB% --width 32 --mixup 0.4 --stereo
set OUT=results_improve
if not exist %OUT% mkdir %OUT%

python train_cnn.py %DATA% %STAB% --deltas --out_dir %OUT%\deltas > %OUT%\deltas.log 2>&1
python train_cnn.py %DATA% %BEST% --out_dir %OUT%\combined > %OUT%\combined.log 2>&1
python train_cnn.py %DATA% %BEST% --pretrain_dir IRMAS-unused --pretrain_epochs 15 --out_dir %OUT%\combined_pretrain > %OUT%\combined_pretrain.log 2>&1

echo Done. Summary:
findstr /B /C:"Best epoch" /C:"Test accuracy" %OUT%\deltas.log %OUT%\combined.log %OUT%\combined_pretrain.log
