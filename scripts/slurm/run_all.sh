#!/bin/bash
# Submit the whole pipeline with dependencies. Usage: bash scripts/slurm/run_all.sh
set -euo pipefail
mkdir -p logs
J1=$(sbatch --parsable scripts/slurm/01_flamingo.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/slurm/02_translate.sh)
J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/slurm/03_evaluate.sh)
echo "Submitted: flamingo=$J1 translate=$J2 evaluate=$J3"
