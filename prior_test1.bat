@echo off
set TRAIN_METHOD=channel
set TEST_METHOD=channel
set LR=1e-3
set N_PREFIX=10
@REM set DATASET=emo
set DATASET=glue-sst2
@REM set TRAIN_TASK=tune
set TRAIN_TASK=glue
@REM set SPLIT=train
set SPLIT=glue
set MODEL=gpt2
set TRAIN_SIZE=100
set STEP=3000
set K=4
set MODEL_PRIOR = gpt2

python test.py ^
    --dataset %DATASET% ^
    --gpt %MODEL% ^
    --method %TEST_METHOD% ^
    --test_batch_size 16 ^
    --out_dir out\%MODEL% ^
    --k %K% ^
    --embedding_dir embeddings\ ^
    --use_demonstrations ^
    --concept_temperature 50 ^
    --similarity_temperature 0.1 ^
    --train_size %TRAIN_SIZE% ^
    --difficulty concept_calibrated ^
    --n_prefix_tokens %N_PREFIX% ^
    --concept_dir concept_likelihood\gpt2\%TRAIN_TASK%-%SPLIT%-%TRAIN_SIZE%\%DATASET%-%TRAIN_METHOD%-prefix=%N_PREFIX%-lr=%LR%-%STEP% ^
    --prefix_embed_file checkpoints\gpt2\%TRAIN_TASK%-%SPLIT%\prefix={%N_PREFIX%}-{%TRAIN_METHOD%}-lr={%LR%}-initByVocab\soft_embeddings-%STEP%.pt ^
    --prior easiest ^
    --reorder 
