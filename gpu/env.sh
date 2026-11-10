# Common environment: every cache and temporary file lives inside $RND, so that
# deleting that single folder removes every trace of the runs.
RND="${RND:-$HOME/rnd}"
export RND \
  PIP_CACHE_DIR="$RND/pip" XDG_CACHE_HOME="$RND/cache" XDG_CONFIG_HOME="$RND/config" XDG_DATA_HOME="$RND/share" \
  TORCH_HOME="$RND/torch" HF_HOME="$RND/hf" MPLCONFIGDIR="$RND/mpl" PYTHONPYCACHEPREFIX="$RND/pycache" \
  CUDA_CACHE_PATH="$RND/nv" TRITON_CACHE_DIR="$RND/triton" TORCHINDUCTOR_CACHE_DIR="$RND/inductor" \
  TMPDIR="$RND/tmp" PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 DO_NOT_TRACK=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
