#!/usr/bin/env bash
# Downloads CIFAR-10 (python version) and the CIFAR-10H vote counts into $DATA_DIR (default $RND/data).
# The CIFAR-10 server is slow on a single connection: the archive is fetched in parallel byte ranges,
# then checked against the MD5 that torchvision expects. CIFAR-10H is CC BY-NC-SA 4.0: it is downloaded here,
# used on the GPU machine and never copied into the repository.
set -euo pipefail
. "$(dirname "$0")/env.sh"
DATA_DIR="${DATA_DIR:-$RND/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
URL="${CIFAR_URL:-https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz}"
MD5=c58f30108f718f92721af3b95e74349a
PARTS="${PARTS:-16}"
if [ ! -d cifar-10-batches-py ]; then
  size=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  step=$(( (size + PARTS - 1) / PARTS ))
  mkdir -p parts
  for i in $(seq 0 $((PARTS - 1))); do
    lo=$(( i * step )); hi=$(( lo + step - 1 )); [ "$hi" -ge "$size" ] && hi=$(( size - 1 ))
    ( until curl -sfL --retry 5 --speed-time 30 --speed-limit 1000 -r "$lo-$hi" -o "parts/$i" "$URL" \
            && [ "$(stat -c %s "parts/$i")" -eq $(( hi - lo + 1 )) ]; do sleep 2; done ) &
  done
  wait
  for i in $(seq 0 $((PARTS - 1))); do cat "parts/$i"; done > cifar-10-python.tar.gz
  rm -rf parts
  echo "$MD5  cifar-10-python.tar.gz" | md5sum -c -
  tar xzf cifar-10-python.tar.gz
fi
if [ -z "${SKIP_CIFAR10H:-}" ] && [ ! -f cifar10h-counts.npy ]; then   # only E3 needs it
  curl -sfL --retry 5 -o cifar10h-counts.npy \
    https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy
fi
ls -la "$DATA_DIR"
