# GPU runs

The scripts that produce the raw files of E1, E2 and E3 (`PROTOCOL.md`, sections 5, 6, 7, 10 and 11).
They write only case-by-case cues and logits; every measure is computed afterwards by the analysis,
from `results/raw/`.

Tested on one NVIDIA L4 (24 GB), 4 cores, Python 3.12, torch 2.15 + CUDA 13.2. All data stays on the
GPU: images normalised once, augmentation and frame applied batch by batch, no DataLoader.

## Files

| File | What it does |
|---|---|
| `env.sh` | Puts every cache and temporary file (pip, torch, CUDA, Triton, XDG, `TMPDIR`) inside `$RND` (default `~/rnd`) |
| `setup.sh` | Virtual environment in `$RND/venv` with torch, torchvision, numpy |
| `fetch_data.sh` | CIFAR-10 (parallel byte ranges, MD5 checked) and the CIFAR-10H counts into `$RND/data` |
| `common.py` | Networks (`Net`, `Machine` from turing-meter's `silicon.py`), task splits, augmentation, frame, training loop |
| `pilot.py` | Throughput, frame probe, epochs and learning rate chosen at `mu = 0` on validation |
| `bench.py` | E1: "plus", "minus" and "plus2" networks for every task, `mu` and seed |
| `rephase.py` | E2: person H, control H', machines `M_lambda` trained against H's cue |
| `people.py` | E3: ResNet-18 base machines, fine-tuning against the CIFAR-10H cue on held-out halves |
| `manifest.py` | `manifest.json` of each folder from the per-unit records |
| `run.sh`, `queue.sh` | One stage; the whole sequence E1, E2, E3 |
| `sanity.py` | Accuracies and raw correlations per condition, from `results/raw/` (not the analysis) |
| `diagnose_rephase.py` | E2 diagnostic: retrains H and some `M_lambda` of one unit and measures them on their own training half |

## How to run

From the machine that holds the repository, with `GPU` set to `user@host` of the GPU machine
(nothing about the host is stored in the repository):

```bash
SSH="ssh -i <key> -o IdentitiesOnly=yes"
rsync -a -e "$SSH" gpu "$GPU:rnd/work/"
$SSH "$GPU" 'cd rnd/work && bash gpu/setup.sh && bash gpu/fetch_data.sh'
$SSH "$GPU" 'cd rnd/work && bash gpu/run.sh pilot 3'          # writes rnd/out/pilot/config.json
$SSH "$GPU" 'cd rnd/work && BENCH_SEEDS=0,1 setsid -f bash -c "bash gpu/queue.sh > ~/rnd/queue.log 2>&1"'
rsync -a -e "$SSH" --include '*/' --include '*.npz' --include 'manifest.json' --exclude '*' \
      "$GPU:rnd/out/bench/" results/raw/bench/       # the same for rephase/ and people/ (skip people/ckpt)
$SSH "$GPU" 'rm -rf ~/rnd'                            # nothing is left on the machine
python gpu/sanity.py results/raw path/to/cifar10h-counts.npy
```

Units are resumable: a stage that is started again skips the `.npz` files already written.
An optional `rnd/out/deviations.json` (`{"bench": ["..."], ...}`) is copied into the manifests.

## Choices made here (beyond the protocol)

- **Binary tasks.** Class 1 is the second class of each pair; `z` = logit of class 1 minus logit of class 0.
- **Frame.** The outer 2-pixel border of all three channels is *set* to `a * delta` in normalised units,
  after crop and flip, without clipping. `delta = 1` (one standard deviation of a normalised pixel); a ridge
  probe from the 3,072 pixels recovers `a` with R² = 1.00 for every `delta` tried (0.25, 0.5, 1).
- **Validation and test.** Each image is shown 5 times, `a = sqrt(1 + mu^2) * eps`; the `eps` are the same
  for the three judges of a unit and, as common random numbers, for every `mu`.
- **Seeds.** Every judge has its own seed, `crc32(label)`, which seeds the initialisation and one CUDA
  generator for batch order, crops, flips and training frames. A judge keeps its seed across `mu`; in E2
  all `M_lambda` of a unit share theirs, and in E3 all `lambda` of a fold.
- **Pilot rule** (fixed before the grid): two tasks, one seed, `mu = 0`, epochs {10, 15, 20, 30} x peak
  learning rate {0.05, 0.1}; the smallest number of epochs within 0.5 accuracy points of the best cell, then
  the best learning rate there. Batch 256, one-cycle schedule.
- **E2 penalty.** Pearson correlation in the batch of the two cues towards the true class, after removing
  the class means of the batch; H frozen in eval mode on the same augmented, framed batch; machines'
  training frame `a ~ N(0, 2)`.
- **E3.** Base ResNet-18: 40 epochs, peak learning rate 0.2, batch 256. Fine-tuning on one half of the test
  set: 8 epochs, learning rate 0.01 with cosine decay, batch 256, crop and flip; machine cue = true logit
  minus logsumexp of the others, people cue = `logit((h + 0.5) / (n + 1))`, class means removed in the batch.
  CIFAR-10H (CC BY-NC-SA 4.0) is downloaded on the GPU machine and never written into the outputs.
