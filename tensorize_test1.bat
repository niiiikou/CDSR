@echo off
set METHOD=channel
set N_PREFIX=10
set TASK=tune
@REM set TASK=glue
set SPLIT=train
@REM set SPLIT=glue

python train.py ^
  --task %TASK% ^
  --split %SPLIT% ^
  --tensorize_dir tensorized ^
  --seed 100 ^
  --method %METHOD% ^
  --n_prefix_tokens %N_PREFIX% ^
  --do_tensorize ^
  --n_gpu 1 ^
  --n_process 10
