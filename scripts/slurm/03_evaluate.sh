#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/evaluate_%j.out
source scripts/slurm/common.sh
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
python -m src.evaluation.evaluate
