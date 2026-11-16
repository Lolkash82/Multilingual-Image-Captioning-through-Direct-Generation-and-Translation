#!/bin/bash
#SBATCH --job-name=flamingo
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/flamingo_%A_%a.out
# One array task per mode: EN-only, Direct DE, Direct FR.
source scripts/slurm/common.sh
LANGS=(en de fr)
python -m src.inference.run_flamingo --lang "${LANGS[$SLURM_ARRAY_TASK_ID]}" ${LIMIT:+--limit $LIMIT}
