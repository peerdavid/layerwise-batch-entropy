#!/bin/bash


# wandb sweep FILE.yaml
SWEEP_ID=$1
GPU=$2
CUDA_VISIBLE_DEVICES=$GPU wandb agent "uibk_iis/Layerwise Batch Entropy/$SWEEP_ID"