@echo off
REM Screening experiments for the deep CNN. Each run adds ONE technique on top of
REM the stabilised setup (STAB), so its effect can be read off on its own.
REM Results: results_improve\<name>\ (model, curve, probs.npz) and results_improve\<name>.log
REM Takes roughly 1.5-2 hours in total on CPU.

set DATA=--data_dir IRMAS-TrainingData
set STAB=--weight_decay 1e-4 --warmup 3 --clip_grad 1.0
set OUT=results_improve
if not exist %OUT% mkdir %OUT%

python train_cnn.py %DATA% %STAB% --out_dir %OUT%\stab > %OUT%\stab.log 2>&1
python train_cnn.py %DATA% %STAB% --crop_frames 108 --out_dir %OUT%\crop > %OUT%\crop.log 2>&1
python train_cnn.py %DATA% %STAB% --mixup 0.4 --out_dir %OUT%\mixup > %OUT%\mixup.log 2>&1
python train_cnn.py %DATA% %STAB% --augment --n_masks 1 --mask_freq 8 --mask_time 10 --out_dir %OUT%\specaug_light > %OUT%\specaug_light.log 2>&1
python train_cnn.py %DATA% %STAB% --stereo --out_dir %OUT%\stereo > %OUT%\stereo.log 2>&1
python train_cnn.py %DATA% %STAB% --deltas --out_dir %OUT%\deltas > %OUT%\deltas.log 2>&1
python train_cnn.py %DATA% %STAB% --width 32 --out_dir %OUT%\width32 > %OUT%\width32.log 2>&1

echo Done. Summary:
findstr /B /C:"Best epoch" /C:"Test accuracy" %OUT%\*.log
