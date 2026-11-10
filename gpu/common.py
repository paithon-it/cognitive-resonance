"""Shared pieces for the GPU runs of the cognitive-resonance protocol (v1.1).

Everything lives on the GPU: the CIFAR-10 images are normalised once and kept as tensors,
augmentation and the spurious frame are applied batch by batch, there is no DataLoader.
The networks come from turing-meter's experiments/angle-between-errors/gpu/silicon.py:
`Net` (the small convolutional "person") and `Machine` (a ResNet-18 for CIFAR).
"""
import json
import os
import platform
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
MEAN = (0.491, 0.482, 0.447)
STD = (0.247, 0.243, 0.262)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# Protocol section 5: five confusable pairs; class 1 of each binary task is the second of the pair.
TASKS = {
    "cat-dog": (3, 5),
    "deer-horse": (4, 7),
    "automobile-truck": (1, 9),
    "airplane-ship": (0, 8),
    "bird-frog": (2, 6),
}
SPLIT_SEED = 20261110          # fixed split of each task into training / validation (and halves A / B)
N_VAL_PER_CLASS = 500          # 1,000 of the 10,000 training images of a task are held out
PRESENTATIONS = 5              # every validation and test image is shown 5 times, with different frames
FRAME_WIDTH = 2                # pixels


def seed_of(*parts) -> int:
    """A stable 31-bit seed from a readable label, e.g. seed_of('bench', 'cat-dog', 's0', 'plus')."""
    return zlib.crc32("/".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def setup_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------- networks

class Net(nn.Module):
    """The person: the small convolutional network of silicon.py, with a configurable output."""
    def __init__(self, n_out: int = 2):
        super().__init__()
        b = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.f = nn.Sequential(b(3, 48), b(48, 96), nn.MaxPool2d(2), b(96, 96), b(96, 192), nn.MaxPool2d(2),
                               b(192, 192), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.c = nn.Linear(192, n_out)

    def forward(self, x):
        return self.c(self.f(x))


class Block(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.a = nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o)
        self.b = nn.Conv2d(o, o, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o)
        self.s = nn.Identity() if s == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, stride=s, bias=False),
                                                                       nn.BatchNorm2d(o))

    def forward(self, x):
        return F.relu(self.bn2(self.b(F.relu(self.bn1(self.a(x))))) + self.s(x))


class Machine(nn.Module):
    """The machine: the ResNet-18 for CIFAR of silicon.py."""
    def __init__(self, n_out: int = 10):
        super().__init__()
        self.ini = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        c = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2),
             (512, 512, 1)]
        self.b = nn.Sequential(*[Block(*x) for x in c])
        self.fc = nn.Linear(512, n_out)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.b(self.ini(x)), 1).flatten(1))


def new_model(cls, seed: int, **kw) -> nn.Module:
    torch.manual_seed(seed)
    return cls(**kw).to(DEV).to(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- data

def data_root() -> str:
    """Where CIFAR-10 and CIFAR-10H are kept: $DATA_DIR, or $RND/data."""
    return os.environ.get("DATA_DIR") or os.path.join(os.environ.get("RND", os.path.expanduser("~/rnd")), "data")


def cifar(root: str = None):
    """CIFAR-10 as normalised float tensors on the GPU: (Xtr, Ytr, Xte, Yte)."""
    import torchvision
    root = root or data_root()
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    mean = torch.tensor(MEAN)[None, :, None, None]
    std = torch.tensor(STD)[None, :, None, None]
    conv = lambda x: ((torch.tensor(x).permute(0, 3, 1, 2).float() / 255.0 - mean) / std).to(DEV)
    return (conv(tr.data), torch.tensor(tr.targets, device=DEV), conv(te.data), torch.tensor(te.targets, device=DEV))


def task_split(task: str, ytr: np.ndarray, yte: np.ndarray):
    """Indices for one binary task: training, validation (CIFAR-10 training set) and test (CIFAR-10 test set).

    The validation images are 500 per class, drawn once with SPLIT_SEED; the test images are all 2,000 test
    images of the two classes, in their CIFAR-10 order. Halves A and B (experiment E2) split the training
    images in two, stratified by class.
    """
    c0, c1 = TASKS[task]
    rng = np.random.default_rng(seed_of(SPLIT_SEED, task))
    train, val, half_a, half_b = [], [], [], []
    for c in (c0, c1):
        idx = rng.permutation(np.flatnonzero(ytr == c))
        val.append(idx[:N_VAL_PER_CLASS])
        rest = idx[N_VAL_PER_CLASS:]
        train.append(rest)
        half_a.append(rest[: len(rest) // 2])
        half_b.append(rest[len(rest) // 2:])
    test = np.flatnonzero((yte == c0) | (yte == c1))
    cat = lambda parts: np.sort(np.concatenate(parts))
    return {"train": cat(train), "val": cat(val), "test": test, "half_a": cat(half_a), "half_b": cat(half_b)}


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop of the image reflected by 4 pixels, and a random horizontal flip (vectorised)."""
    b = x.shape[0]
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)          # b, 40, 40, c
    i = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    j = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    rows, cols = (i[:, None] + ar)[:, :, None], (j[:, None] + ar)[:, None, :]
    out = pad[torch.arange(b, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


_BORDER = None


def border_mask() -> torch.Tensor:
    global _BORDER
    if _BORDER is None:
        m = torch.zeros(1, 1, 32, 32, device=DEV)
        w = FRAME_WIDTH
        m[..., :w, :] = 1; m[..., -w:, :] = 1; m[..., :, :w] = 1; m[..., :, -w:] = 1
        _BORDER = m
    return _BORDER


def apply_frame(x: torch.Tensor, a: torch.Tensor, delta: float) -> torch.Tensor:
    """The spurious cue: the outer 2-pixel border of every channel is set to a * delta (normalised units, no clipping)."""
    m = border_mask()
    return x * (1 - m) + (a * delta).to(x.dtype)[:, None, None, None] * m


# ----------------------------------------------------------------------------- training

def batch_corr(c_m: torch.Tensor, c_h: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Pearson correlation in the batch of two cues, after removing the mean of each true class in the batch."""
    c_m, c_h = c_m.float(), c_h.float()
    cnt = torch.zeros(n_classes, device=y.device).index_add_(0, y, torch.ones_like(c_m)).clamp(min=1)
    r_m = c_m - (torch.zeros(n_classes, device=y.device).index_add_(0, y, c_m) / cnt)[y]
    r_h = c_h - (torch.zeros(n_classes, device=y.device).index_add_(0, y, c_h) / cnt)[y]
    return (r_m * r_h).sum() / torch.sqrt((r_m * r_m).sum() * (r_h * r_h).sum() + 1e-12)


def train(model, X, Y, *, epochs, lr, bs, seed, frame=None, delta=1.0, extra=None, wd=5e-4, schedule="onecycle",
          augmentation=True):
    """SGD with Nesterov momentum and a one-cycle schedule, as in silicon.py, mixed precision bf16.

    frame(y, gen) -> a or None draws the frame of each training image; extra(x, y, logits, idx) -> scalar adds a term
    to the loss (idx are the positions of the batch in X). All randomness (order, crops, flips, frames) comes from
    one generator seeded with `seed`. Returns the mean of the extra term over the last epoch, if any.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    per_epoch = len(X) // bs
    if schedule == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch)
    else:                                                       # short fine-tuning: cosine from lr to zero
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * per_epoch)
    model.train()
    last = []
    for ep in range(epochs):
        order = torch.randperm(len(X), device=DEV, generator=gen)
        for k in range(per_epoch):
            b = order[k * bs:(k + 1) * bs]
            x = augment(X[b], gen) if augmentation else X[b]
            y = Y[b]
            if frame is not None:
                x = apply_frame(x, frame(y, gen), delta)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            if extra is not None:
                term = extra(x, y, out, b)
                loss = loss + term
                if ep == epochs - 1:
                    last.append(term.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    return float(torch.stack(last).mean()) if last else None


@torch.no_grad()
def logits(model, X, a=None, delta=1.0, bs=2000) -> torch.Tensor:
    """Logits in eval mode and full precision (TF32 convolutions), optionally with the frame a * delta."""
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        x = X[k:k + bs]
        if a is not None:
            x = apply_frame(x, a[k:k + bs], delta)
        out.append(model(x.contiguous(memory_format=torch.channels_last)).float())
    return torch.cat(out)


def log_odds(lg: torch.Tensor) -> np.ndarray:
    """z = logit of class 1 minus logit of class 0."""
    return (lg[:, 1] - lg[:, 0]).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- bookkeeping

def versions() -> dict:
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0)}


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False))
    os.replace(tmp, path)


def save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)
