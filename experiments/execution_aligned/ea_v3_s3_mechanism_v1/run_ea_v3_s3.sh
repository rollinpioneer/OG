#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=0
unset XLA_FLAGS || true
REPO=/home/__compress_data/xushijie/OG_ea_v3_s3_mechanism
cd "$REPO"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate og-ea-v2
export PYTHONPATH="$REPO"
STAGE="${1:-preflight}"
echo "START $(date -Is) HEAD=$(git rev-parse HEAD) STAGE=$STAGE"
python -m execution_aligned_rl.v3.s3_mechanism.run --stage "$STAGE"
echo "END $(date -Is) EXIT=$?"
