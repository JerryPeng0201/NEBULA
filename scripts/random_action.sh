#!/bin/bash

# Data Collection Script for NEBULA
# This script runs motion planning data collection with the Panda robot
cd ..
CUDA_VISIBLE_DEVICES=0 python -m nebula.demos.demo_random_action \
    -e Spatial-KitchenAssembly-Hard \
    --record-dir /HDD1/embodied_ai/data/Nebula/ \