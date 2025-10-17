#!/bin/bash

# Data Collection Script for NEBULA
# This script runs motion planning data collection with the Panda robot

CUDA_VISIBLE_DEVICES=0 python -m nebula.data.generation.motionplanning.panda.run \
    -e Spatial-PickCube-Hard \
    -o rgb+depth+segmentation \
    -n 10 \
    --render-mode sensors \
    --save-video \
    --record-dir /HDD1/embodied_ai/data/Nebula/Nebula-demo \
    --subtask-idx 0