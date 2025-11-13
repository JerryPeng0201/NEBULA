#!/bin/bash

# Convert NEBULA dataset to RLDS format
# Usage: bash nebula2rlds.sh

python -m nebula.data.dataset.nebula_to_rlds \
    --data-root /mnt/rds/VipinRDS/VipinRDS/users/jxp1146/data/NEBULA/beta \
    --output-dir /mnt/rds/VipinRDS/VipinRDS/users/jxp1146/data/NEBULA/rlds \