#!/bin/sh
#$ -cwd
#$ -l node_f=2
# 実行時間を指定
#$ -l h_rt=03:00:00
#$ -o outputs/$JOB_ID
#$ -e errors/$JOB_ID
#$ -p -5
#$ -N res18c100lion

# swich virtual env
source ~/.bashrc
module purge

# Moduleコマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1


# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

export NCCL_IB_DISABLE=1
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
   python pretrain.py \
    --use_cifar \
    --cluster 'tsubame' \
    --project_name 'vision_lion' \
    --wo_infiniband \
    --dataset CIFAR100 \
    --model resnet18 \
    --workers 16 \
    --batch-size 64 \
    --sched cosine_iter \
    --epochs 100 \
    --lr 3e-4 \
    --beta2 0.99 \
    --weight-decay 5e-4 \
    --optimizer_name 'u4_sign_lion' \
    --num-classes 100 \
    --warmup-epochs 0 \
    --cooldown-epochs 0 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa 'rand-m9-mstd0.5-inc1' \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --remode 'pixel' \
    --interpolation 'bicubic' \
    --hflip 0.0 \
    --eval-metric 'loss' \
    --log_wandb \
    --AugmentAll