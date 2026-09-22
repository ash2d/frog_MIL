#!/usr/bin/env bash
# Run a command in the Perch/TensorFlow environment with CUDA wired up.
#
# tensorflow[and-cuda] ships the CUDA libraries as pip packages but does not put
# them on the loader path, so TF silently falls back to CPU with only a
# "Cannot dlopen some GPU libraries" warning. Perch v2 on CPU is ~50x slower,
# so always go through this wrapper.
#
#   scripts/perch_env.sh python scripts/embed_perch.py --out-dir embeddings
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.venv-perch"
[ -d "$VENV" ] || { echo "missing $VENV -- run: UV_PROJECT_ENVIRONMENT=.venv-perch uv sync --only-group perch" >&2; exit 1; }
SP="$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
NVLD="$(find "$SP/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLD}${SP}/tensorflow${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec env PATH="$VENV/bin:$PATH" "$@"
