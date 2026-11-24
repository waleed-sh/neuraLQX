#!/usr/bin/env bash
#SBATCH --job-name=nqx-nodes-tensor
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=nqx-nodes-tensor-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda
export NEURALQX_SHARDING=samples,operators,tensor
export NEURALQX_MESH_SAMPLE_AXIS_SIZE=2
export NEURALQX_MESH_OPERATOR_AXIS_SIZE=2
export NEURALQX_MESH_TENSOR_AXIS_SIZE=2
export NEURALQX_DISTRIBUTED=1
export NEURALQX_DISTRIBUTED_AUTO_INITIALIZE=1
export NEURALQX_DISTRIBUTED_NUM_PROCESSES="${SLURM_NTASKS}"
export NEURALQX_DISTRIBUTED_LOCAL_DEVICE_IDS=0,1,2,3
nqx_coordinator_host=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export NEURALQX_DISTRIBUTED_COORDINATOR_ADDRESS="${nqx_coordinator_host}:12355"
srun --kill-on-bad-exit=1 bash run-slurm-worker.sh
