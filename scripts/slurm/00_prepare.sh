#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_%j.out
# Downloads Multi30K, Flickr30K images and all models into the local cache. Needs internet on the node.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv-flamingo/bin/activate
export HF_HOME="$PWD/.hf_cache"
python scripts/prepare_multi30k.py
python scripts/download_models.py
python -m pytest tests -q
