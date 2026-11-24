#!/usr/bin/env bash
#SBATCH --job-name=nqx-gpus-tensor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-gpus-tensor-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda
export NEURALQX_DISTRIBUTED=0
export NEURALQX_SHARDING=samples,tensor
export NEURALQX_MESH_TENSOR_AXIS_SIZE=2
srun --ntasks=1 python -u simulation.py
