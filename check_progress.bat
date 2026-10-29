@echo off
REM Shows progress of the experiment that is currently running (the most recently
REM updated log in results_improve or results_multilabel): the live progress bar of
REM the current epoch, the last finished epochs, and the results of finished runs.
powershell -NoProfile -Command ^
  "$logs = Get-ChildItem results_improve\*.log, results_multilabel\*.log -ErrorAction SilentlyContinue;" ^
  "$log = $logs | Sort-Object LastWriteTime | Select-Object -Last 1;" ^
  "$text = Get-Content $log.FullName -Raw;" ^
  "Write-Host ('Current run: ' + $log.Directory.Name + '\' + $log.BaseName);" ^
  "$bars = [regex]::Matches($text, '(Epoch \d+/\d+|Scoring excerpts|Extracting[^:]*):\s+\d+%%\|[^|]*\|\s*[\d/]+ \[[^\]]*\]');" ^
  "if ($bars.Count) { Write-Host ('  now:  ' + ($bars[$bars.Count - 1].Value -replace '\|[^|]*\|', ' ')) };" ^
  "$done = [regex]::Matches($text, '(Pretraining epoch|Epoch) +\d+/\d+ \|[^\r\n]*');" ^
  "if ($done.Count) { Write-Host '  finished epochs:'; $done | Select-Object -Last 3 | ForEach-Object { '    ' + $_.Value } } else { Write-Host '  (no epoch finished yet)' };" ^
  "if ($text -match 'Traceback') { Write-Host '  !! this run crashed - see the log' };" ^
  "Write-Host ''; Write-Host 'Finished runs:';" ^
  "Select-String $logs -Pattern '^Test accuracy|^Test mixtures' | ForEach-Object { '  ' + ($_.Path | Split-Path -Parent | Split-Path -Leaf) + '\' + $_.Filename.PadRight(24) + $_.Line.Split(':')[-1].Trim() }"
