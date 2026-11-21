#!/usr/bin/env bash
#SBATCH --job-name=nqx-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-cpu-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cpu
export NEURALQX_DISTRIBUTED=0
export NEURALQX_SHARDING=samples
srun --ntasks=1 python -u simulation.py
