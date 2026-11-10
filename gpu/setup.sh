#!/usr/bin/env bash
# Prepares the GPU machine: a virtual environment with torch and torchvision inside $RND.
# Usage, on the GPU machine, from the folder that contains gpu/:  bash gpu/setup.sh
set -euo pipefail
RND="${RND:-$HOME/rnd}"
mkdir -p "$RND"
. "$(dirname "$0")/env.sh"
python3 -m venv "$RND/venv"
. "$RND/venv/bin/activate"
pip install -q --no-cache-dir --upgrade pip
pip install -q --no-cache-dir torch torchvision numpy
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
