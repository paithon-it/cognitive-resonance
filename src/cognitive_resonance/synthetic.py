"""Synthetic raw files with the shapes of § 11, to test the readers and dry-run the analyses.

They are not data. The cues come from a Gaussian toy in which each network reads the image
with its own noise (shared in part between networks: the hard images are hard for all)
and, from mu > 0, also the frame a, with a sign that depends on its bias. Nothing here is
used to produce results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TASKS = ("cat_dog", "deer_horse", "automobile_truck", "airplane_ship", "bird_frog")
MUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
LAMBDAS_REPHASE = (0.0, 0.3, 1.0, 3.0, 10.0)
LAMBDAS_PEOPLE = (0.0, 1.0, 3.0, 10.0)
MACHINES_BIS = ("M0", "Mpair", "Mframe")


def _nets(gen, n_images, presentations, d, mu, signs, rho_img=0.6, scale=1.3):
    y_img = (np.arange(n_images) >= n_images // 2).astype(int)
    image = np.repeat(np.arange(n_images), presentations)
    y = y_img[image]
    s = 2.0 * y - 1.0
    a = gen.normal(0.0, math.sqrt(1.0 + mu * mu), image.size)
    common = gen.standard_normal(n_images)
    beta = 0.8 * mu
    g = 1.0 / (1.0 + 0.3 * mu)
    out = []
    for sign in signs:
        own = rho_img ** 0.5 * common + (1 - rho_img) ** 0.5 * gen.standard_normal(n_images)
        x = s * d * g / 2.0 + own[image] + 0.4 * gen.standard_normal(image.size) + sign * beta * a
        out.append(scale * x + 0.1)
    return y, image, a, out


def write_bench(folder, tasks=TASKS, mus=MUS, seeds=(0, 1, 2), n_images=2000, presentations=5,
                n_val=1000, seed=7) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for ti, task in enumerate(tasks):
        d = 1.6 + 0.25 * ti
        for mu in mus:
            for sd in seeds:
                gen = np.random.default_rng([seed, ti, int(mu * 100), sd])
                y, image, a, (zp, zm, zp2) = _nets(gen, n_images, presentations, d, mu, (1, -1, 1))
                vy, vimage, va, (vzp, vzm, vzp2) = _nets(gen, n_val, 1, d, mu, (1, -1, 1))
                np.savez(folder / f"{task}_mu{mu:g}_s{sd}.npz", y=y, image=image, a=a,
                         z_plus=zp, z_minus=zm, z_plus2=zp2, val_y=vy, val_image=vimage, val_a=va,
                         val_z_plus=vzp, val_z_minus=vzm, val_z_plus2=vzp2)
    (folder / "manifest.json").write_text(json.dumps({"synthetic": True}))


def _rephase_cues(gen, n_images, presentations, d, lambdas):
    y_img = (np.arange(n_images) >= n_images // 2).astype(int)
    image = np.repeat(np.arange(n_images), presentations)
    y = y_img[image]
    s = 2.0 * y - 1.0
    a = gen.normal(0.0, math.sqrt(2.0), image.size)
    common = gen.standard_normal(n_images)
    eh = 0.6 ** 0.5 * common + 0.4 ** 0.5 * gen.standard_normal(n_images)
    eh2 = 0.6 ** 0.5 * common + 0.4 ** 0.5 * gen.standard_normal(n_images)
    zh = s * d / 2 + eh[image] + 0.4 * gen.standard_normal(image.size) + 0.8 * a
    zh2 = s * d / 2 + eh2[image] + 0.4 * gen.standard_normal(image.size) + 0.8 * a
    zm = []
    for lam in lambdas:
        r = 0.6 / (1.0 + lam)
        em = r ** 0.5 * common + (1 - r) ** 0.5 * gen.standard_normal(n_images)
        gamma = -0.9 * lam / (1.0 + lam)
        dm = d * (1.0 - 0.1 * lam / (1.0 + lam))
        zm.append(s * dm / 2 + em[image] + 0.4 * gen.standard_normal(image.size) + gamma * a)
    return y, image, a, zh, zh2, np.array(zm)


def write_rephase(folder, tasks=TASKS, seeds=(0, 1), lambdas=LAMBDAS_REPHASE, n_images=2000,
                  presentations=5, n_val=1000, seed=8) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for ti, task in enumerate(tasks):
        d = 1.6 + 0.25 * ti
        for sd in seeds:
            gen = np.random.default_rng([seed, ti, sd])
            y, image, a, zh, zh2, zm = _rephase_cues(gen, n_images, presentations, d, lambdas)
            vy, vimage, va, vzh, vzh2, vzm = _rephase_cues(gen, n_val, 1, d, lambdas)
            np.savez(folder / f"{task}_s{sd}.npz", y=y, image=image, a=a, z_H=zh, z_H2=zh2, z_M=zm,
                     lambdas=np.array(lambdas), val_y=vy, val_image=vimage, val_a=va, val_z_H=vzh,
                     val_z_H2=vzh2, val_z_M=vzm)
    (folder / "manifest.json").write_text(json.dumps({"synthetic": True}))


def _rephase_bis_cues(gen, n_images, presentations, d):
    """H as in E2; M0 reads the image, Mpair also the frame against H, Mframe only the frame."""
    y_img = (np.arange(n_images) >= n_images // 2).astype(int)
    image = np.repeat(np.arange(n_images), presentations)
    y = y_img[image]
    s = 2.0 * y - 1.0
    a = gen.normal(0.0, math.sqrt(2.0), image.size)
    common = gen.standard_normal(n_images)
    noise = lambda r: (r ** 0.5 * common + (1 - r) ** 0.5 * gen.standard_normal(n_images))[image]  # noqa: E731
    zh = s * d / 2 + noise(0.6) + 0.4 * gen.standard_normal(image.size) + 0.8 * a
    m0 = s * d / 2 + noise(0.6) + 0.4 * gen.standard_normal(image.size)
    mpair = s * 0.9 * d / 2 + noise(0.4) + 0.4 * gen.standard_normal(image.size) - 0.7 * a
    mframe = 0.6 * gen.standard_normal(image.size) - 0.8 * a
    return y, image, a, zh, np.array([m0, mpair, mframe])


def write_rephase_bis(folder, tasks=TASKS, seeds=(0, 1), n_images=2000, presentations=5, n_val=1000,
                      seed=10) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for ti, task in enumerate(tasks):
        d = 1.6 + 0.25 * ti
        for sd in seeds:
            gen = np.random.default_rng([seed, ti, sd])
            y, image, a, zh, zm = _rephase_bis_cues(gen, n_images, presentations, d)
            vy, vimage, va, vzh, vzm = _rephase_bis_cues(gen, n_val, 1, d)
            np.savez(folder / f"{task}_s{sd}.npz", y=y, image=image, a=a, z_H=zh, z_M=zm,
                     machines=np.array(MACHINES_BIS), val_y=vy, val_image=vimage, val_a=va, val_z_H=vzh,
                     val_z_M=vzm)
    (folder / "manifest.json").write_text(json.dumps({"synthetic": True}))


def people_world(n=10_000, seed=9):
    """Labels, CIFAR-10H-like counts and a hidden 'confusable' class per image."""
    gen = np.random.default_rng(seed)
    y = np.repeat(np.arange(10), n // 10)
    other = (y + gen.integers(1, 10, n)) % 10
    hard = gen.beta(0.6, 8.0, n)                        # share of people who pick the other class
    votes = gen.integers(47, 64, n)
    wrong = gen.binomial(votes, hard)
    counts = np.zeros((n, 10), int)
    counts[np.arange(n), y] = votes - wrong
    counts[np.arange(n), other] += wrong
    return y, other, hard, counts


def _logits(gen, y, other, hard, follow):
    n = y.size
    lg = gen.normal(0.0, 1.0, (n, 10))
    lg[np.arange(n), y] += 4.0
    lg[np.arange(n), other] += 6.0 * hard * follow + gen.normal(0, 0.5, n)
    return lg


def write_people(folder, counts_path, seeds=(0, 1), lambdas=LAMBDAS_PEOPLE, seed=9) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    y, other, hard, counts = people_world(seed=seed)
    np.save(counts_path, counts)
    n = y.size
    for sd in seeds:
        gen = np.random.default_rng([seed, sd])
        np.savez(folder / f"base_s{sd}.npz", image=np.arange(n), y=y, logits=_logits(gen, y, other, hard, 1.0))
        fold_of = np.zeros(n, int)
        for c in range(10):
            members = np.flatnonzero(y == c)
            fold_of[gen.permutation(members)[1::2]] = 1
        for f in (0, 1):
            img = np.flatnonzero(fold_of == f)
            lg = np.stack([_logits(gen, y[img], other[img], hard[img], 1.0 / (1.0 + lam)) for lam in lambdas])
            np.savez(folder / f"fold{f}_s{sd}.npz", image=img, y=y[img], logits=lg, lambdas=np.array(lambdas))
    (folder / "manifest.json").write_text(json.dumps({"synthetic": True}))
