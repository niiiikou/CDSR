@echo off
set METHOD=channel
set LR=1e-3
set N_PREFIX=10
@REM set TASK=tune
set TASK=glue
@REM set SPLIT=train
set SPLIT=glue
set MODEL=gpt2

python train.py ^
    --gpt2 %MODEL% ^
    --task %TASK% ^
    --split %SPLIT% ^
    --method %METHOD% ^
    --n_gpu 1 ^
    --tensorize_dir tensorized ^
    --out_dir checkpoints\%MODEL%\%TASK%-%SPLIT%\prefix={%N_PREFIX%}-{%METHOD%}-lr={%LR%}-initByVocab ^
    --batch_size 16 ^
    --lr %LR% ^
    --n_prefix_tokens %N_PREFIX% ^
    --num_training_steps 5000


