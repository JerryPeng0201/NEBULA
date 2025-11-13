#!/bin/bash

# Convert NEBULA-Beta dataset into RLDS-compliant TFRecord shards.
#
# Usage examples:
#   bash convert_nebula_to_rlds.sh \
#       --data-root /path/to/nebula/beta \
#       --output-dir /path/to/output/rlds \
#       --tasks Control-PlaceSphere-Easy,Control-PushCube-Easy \
#       --max-episodes 500
#
# Any argument not provided will fall back to the defaults below.

set -euo pipefail

# Defaults (override via CLI flags)
DATA_ROOT="/mnt/rds/VipinRDS/VipinRDS/users/jxp1146/data/NEBULA/beta"
OUTPUT_DIR="/mnt/rds/VipinRDS/VipinRDS/users/jxp1146/data/NEBULA/rlds"
TASKS=""           # e.g. "Control-PlaceSphere-Easy,Control-PushCube-Easy"
MAX_EPISODES=""    # e.g. 1000 for a subset
EPISODES_PER_SHARD=100
COMPRESSION="GZIP"  # Options: GZIP, ZLIB, NONE
ENV_NAME="nebula"   # Conda environment containing tensorflow, tqdm, etc.

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --data-root PATH          Root directory containing NEBULA-Beta tasks.
  --output-dir PATH         Destination directory for RLDS TFRecords.
  --tasks LIST              Optional comma-separated task filter.
  --max-episodes N          Optional cap on number of episodes to convert.
  --episodes-per-shard N    Episodes per TFRecord shard (default: $EPISODES_PER_SHARD).
  --compression CODEC       TFRecord compression (GZIP|ZLIB|NONE).
  --env-name NAME           Conda environment to activate before running.
  -h, --help                Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"; shift 2;
            ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2;
            ;;
        --tasks)
            TASKS="$2"; shift 2;
            ;;
        --max-episodes)
            MAX_EPISODES="$2"; shift 2;
            ;;
        --episodes-per-shard)
            EPISODES_PER_SHARD="$2"; shift 2;
            ;;
        --compression)
            COMPRESSION="$2"; shift 2;
            ;;
        --env-name)
            ENV_NAME="$2"; shift 2;
            ;;
        -h|--help)
            usage; exit 0;
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Activate conda environment if conda is available
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "[WARN] Conda not found in PATH; skipping environment activation" >&2
fi

PYTHON=${PYTHON:-python}

CMD=(
    "$PYTHON" -m nebula.data.dataset.nebula_to_rlds
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --episodes-per-shard "$EPISODES_PER_SHARD"
    --compression "$COMPRESSION"
)

if [[ -n "$TASKS" ]]; then
    CMD+=(--tasks "$TASKS")
fi

if [[ -n "$MAX_EPISODES" ]]; then
    CMD+=(--max-episodes "$MAX_EPISODES")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
