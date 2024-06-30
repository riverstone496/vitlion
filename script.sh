#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=00:10:00
#$ -o outputs/$JOB_ID
#$ -e outputs/$JOB_ID
#$ -p -5

# Load modules
module load cuda/12.1.0
module load nccl/2.20.5
module load openmpi/5.0.2-gcc
module load ninja/1.11.1
module load cudnn/9.0.0

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

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PE_HOSTFILE" | while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}" >> "$HOSTFILE_NAME"
done

cat $HOSTFILE_NAME
export NCCL_DEBUG=WARN

mpirun -np $NUM_GPUS \
    --npernode $NUM_GPU_PER_NODE \
    -hostfile $HOSTFILE_NAME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -bind-to none \
    -x PATH \
    python pretrain.py  --model resnet18  --lr 1e-2 --train_data_dir /gs/bs/tga-bayes-crest/ishikawa/dataset/ImageNet2012/train/  --eval_data_dir /gs/bs/tga-bayes-crest/ishikawa/dataset/ImageNet2012/val/
