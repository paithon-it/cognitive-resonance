"""Data generated from the model of § 2, with the design of the bench (E0).

Each image is presented several times. The noise of each judge has a part that belongs
to the image (the same in all its presentations) and a part that belongs to the
presentation, with the same correlation rho in both: the marginal model is exactly that
of § 2, and the presentations of an image are not independent, which is what the
cluster bootstrap must handle.
"""
from __future__ import annotations

import math

import numpy as np

IMAGE_SHARE = 0.5          # share of the noise variance that belongs to the image


def _pair_noise(gen, n, rho):
    a = gen.standard_normal((n, 2))
    return a[:, 0], rho * a[:, 0] + math.sqrt(1.0 - rho * rho) * a[:, 1]


def model_cues(gen, n_images: int, presentations: int, k: float, rho: float, dh: float = 1.0,
               image_share: float = IMAGE_SHARE) -> dict:
    """y, image and the two cues as log-odds toward class 1 (dH x_H and dM x_M)."""
    dm = k * dh
    y_img = (np.arange(n_images) >= n_images // 2).astype(int)
    image = np.repeat(np.arange(n_images), presentations)
    y = y_img[image]
    s = 2.0 * y - 1.0
    ih, im = _pair_noise(gen, n_images, rho)
    ph, pm = _pair_noise(gen, image.size, rho)
    nh = math.sqrt(image_share) * ih[image] + math.sqrt(1.0 - image_share) * ph
    nm = math.sqrt(image_share) * im[image] + math.sqrt(1.0 - image_share) * pm
    xh = s * dh / 2.0 + nh
    xm = s * dm / 2.0 + nm
    return {"y": y, "image": image, "z_h": dh * xh, "z_m": dm * xm}


def model_experiment(gen, k: float, rho: float, dh: float = 1.0, n_images: int = 2000,
                     presentations: int = 5, n_val: int = 1000) -> dict:
    """A test set with the bench's size and a validation set for the standardization."""
    test = model_cues(gen, n_images, presentations, k, rho, dh)
    val = model_cues(gen, n_val, presentations, k, rho, dh)
    return {**test, "val_y": val["y"], "val_z_h": val["z_h"], "val_z_m": val["z_m"]}
