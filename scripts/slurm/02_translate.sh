#!/bin/bash
#SBATCH --job-name=marian
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/marian_%j.out
source scripts/slurm/common.sh
python -m src.translation.run_marian
