METHOD=channel
N_PREFIX=10
TASK=glue
SPLIT=glue
CUDA_VISIBLE_DEVICES=5 python train.py\
  --task $TASK\
  --split $SPLIT\
  --tensorize_dir tensorized\
  --seed 100\
  --method $METHOD\
  --n_prefix_tokens $N_PREFIX\
  --do_tensorize\
  --n_gpu 1\
  --n_process 10\