#!/bin/bash

# Data Collection Script for NEBULA
# This script runs motion planning data collection with the Panda robot
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 python -m nebula.data.generation.motionplanning.xarm6.run \
    -e Control-PlaceSphere-Easy \
    -o rgb+depth+segmentation \
    -n 3 \
    --render-mode sensors \
    --save-video \
    --record-dir ./demo \
    --subtask-idx 1