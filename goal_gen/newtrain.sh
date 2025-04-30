#!/usr/bin/env bash
set -x
GPUS_PER_NODE=1 # number of gpus per machine
MASTER_ADDR="127.0.0.1:29500" # modify it with your own address and port
NNODES=1 # number of machines
JOB_ID=107
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    goal_gen/newtrain.py \
    --config ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $NNODES
