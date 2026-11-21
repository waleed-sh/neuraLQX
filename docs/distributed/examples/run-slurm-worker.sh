#!/usr/bin/env bash
set -euo pipefail

# This runs inside each Slurm task, where the global task ID is defined.
export NEURALQX_DISTRIBUTED_PROCESS_ID="${SLURM_PROCID}"
exec python -u simulation.py "$@"
