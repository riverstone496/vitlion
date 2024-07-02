#!/bin/sh
#$ -cwd
#$ -l node_f=4
#$ -l h_rt=24:00:00
#$ -o outputs/$JOB_ID
#$ -e errors/$JOB_ID
#$ -p -5

# Load modules
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1
module load cudnn/9.0.0

# swich virtual env
source ~/.bashrc

# Distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# Hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
export NCCL_DEBUG=WARN

mpirun -np $NUM_GPUS \
     --npernode $NUM_GPU_PER_NODE \
     -x MASTER_ADDR=$MASTER_ADDR \
     -x MASTER_PORT=$MASTER_PORT \
     -bind-to none \
     -x PATH \
     python pretrain.py  --model resnet50  --lr 3e-3 --weight-decay 1 --beta2 0.99 --momentum_sync_freq 1000 \
        --input-size 3 224 224 --project_name lion_imagenet\
        --sched cosine_iter --epochs 90 \
        --batch-size 1024 --optimizer_name lion_mvote --num-classes 1000 \
        --warmup-epochs 5 --cooldown-epochs 0 \
        --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --remode pixel --interpolation bicubic --hflip 0.0 \
        -j 1 --eval-metric loss --no-prefetcher \
        --output ./output/pretrain \
        --log-wandb --train_data_dir /gs/bs/tga-bayes-crest/ishikawa/dataset/ImageNet2012/train/  --eval_data_dir /gs/bs/tga-bayes-crest/ishikawa/dataset/ImageNet2012/val/