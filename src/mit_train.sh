#!/bin/bash
#SBATCH --job-name=dfsr                   # Name
#SBATCH --gres=gpu:volta:1                # 2 Volta GPUs
#SBATCH --cpus-per-task=40                # 40 CPUs per task
#SBATCH --time=24:00:00                   # Maximum runtime
#SBATCH --mem=340G                        # Memory

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="/home/gridsan/yyao/Research_Projects/workflow/dfsr"
mkdir -p "$OUTPUT_DIR"

OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"

exec > "$OUT_FILE"
exec 2> "$ERR_FILE"

# Load modules
source /etc/profile
module load anaconda/Python-ML-2024b

srun python3 train_dfsr.py \
    --nodes 1 \
    --gpus 1 \
    --epochs 5
