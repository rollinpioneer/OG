#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
REPO="${REPO:-/home/__compress_data/xushijie/OG_ea_v3_materialized_roots}"
cd "$REPO"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate og-ea-v2
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python -m execution_aligned_rl.v3.preflight --repo "$REPO"
python -m execution_aligned_rl.v3.engineering_pilot run-s1 --repo "$REPO"
