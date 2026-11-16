# Sourced by every job script. Edit the paths for your Perun account.
cd "$SLURM_SUBMIT_DIR"
source venv-flamingo/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE=1            # compute nodes: use the cache filled by scripts/download_models.py
export TRANSFORMERS_OFFLINE=1
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi; git rev-parse HEAD 2>/dev/null || true
