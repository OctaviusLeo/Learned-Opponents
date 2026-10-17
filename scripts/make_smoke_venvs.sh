#!/usr/bin/env bash
# Create the two M0 smoke-test environments on the Linux filesystem (task 2.1).
#   ~/.venvs/lo-mjx    MuJoCo Playground + JAX with CUDA 12 (Playground README)
#   ~/.venvs/lo-mjlab  mjlab + MuJoCo Warp + CUDA 12.8 PyTorch
# Each environment also gets onnx + onnxruntime for the ONNX-path check.
# Installed versions are frozen into docs/evidence/ for docs/ENVIRONMENT.md.
set -euo pipefail

UV="${UV:-$HOME/.local/bin/uv}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EVID="$REPO/docs/evidence"
mkdir -p "$EVID" "$HOME/.venvs"

make_env() {
  local dir="$1"; shift
  if [ ! -x "$dir/bin/python" ]; then
    "$UV" venv --python 3.12 "$dir"
  fi
  VIRTUAL_ENV="$dir" "$UV" pip install --python "$dir/bin/python" "$@"
}

# MJX: Playground's README installs `playground` and then jax[cuda12].
make_env "$HOME/.venvs/lo-mjx" \
  "playground==0.2.0" "jax[cuda12]==0.11.2" "mujoco==3.16.0" "mujoco-mjx==3.16.0" \
  "brax==0.14.2" "onnx==1.23.2" "onnxruntime==1.31.0"
"$UV" pip freeze --python "$HOME/.venvs/lo-mjx/bin/python" > "$EVID/freeze_lo-mjx.txt"

# mjlab: pins mujoco~=3.11 / mujoco-warp~=3.11. The `cu128` extra only adds
# a torch requirement; installed from PyPI, torch 2.14.1 resolves to its
# CUDA 13.0 build, which the Windows driver (CUDA 13.4) supports.
make_env "$HOME/.venvs/lo-mjlab" \
  "mjlab[cu128]==1.6.0" "torch==2.14.1" "onnx==1.23.2" "onnxruntime==1.31.0"
"$UV" pip freeze --python "$HOME/.venvs/lo-mjlab/bin/python" > "$EVID/freeze_lo-mjlab.txt"

echo "smoke venvs ready"
