"""The estimators of the protocol, section 3, for judges whose whole cue is known.

Every function that takes `W` accepts either nothing (each presentation counts once), a
vector of weights per presentation, or a matrix (replicates x presentations) of integer
weights from the bootstrap: the presentations of an image all carry the count of their
image. Sensitivities come from the AUC through ranks, so a resample needs no new sort.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import owens_t
from scipy.stats import norm, spearmanr

from . import model

SQRT2 = math.sqrt(2.0)
SPLIT_SEED = 1110          # the halves of the tuned turing, fixed before any run
BOOT_SEED = 2026
RHO_GRID = np.r_[np.linspace(-0.995, -0.905, 19), np.linspace(-0.9, 0.89, 180),
                 np.linspace(0.9, 0.995, 20)]                 # dense where the tetrachoric is steep


# --- standardization ------------------------------------------------------------

def within_class_sd(z, y) -> float:
    """Pooled within-class standard deviation (residuals from the class mean)."""
    z, y = np.asarray(z, float), np.asarray(y)
    resid = np.empty_like(z)
    for c in (0, 1):
        s = y == c
        resid[s] = z[s] - z[s].mean()
    return float(math.sqrt(np.sum(resid ** 2) / (z.size - 2)))


def standardize(z, y, val_z, val_y) -> np.ndarray:
    """Divide the cue by its within-class standard deviation on the validation set."""
    sd = within_class_sd(val_z, val_y)
    if not sd > 0:
        raise ValueError("validation cue without spread")
    return np.asarray(z, float) / sd


# --- AUC through ranks ------------------------------------------------------------

def _weights(W, n):
    if W is None:
        return np.ones(n)
    return np.asarray(W, float)


class SortedScore:
    """A score sorted once; its weighted AUC for any weights is then two cumulative sums.

    Ties count one half, as in the Mann-Whitney statistic.
    """

    def __init__(self, score, y):
        score = np.asarray(score, float)
        y = np.asarray(y)
        order = np.argsort(score, kind="mergesort")
        s = score[order]
        pos = y[order] == 1
        self.order = order
        self.n = score.size
        self.neg_mask = (~pos).astype(float)
        self.pos_at = np.flatnonzero(pos)
        new = np.r_[True, s[1:] != s[:-1]]
        self.ties = not bool(new.all())
        if self.ties:
            gid = np.cumsum(new) - 1
            starts = np.flatnonzero(new)
            ends = np.r_[starts[1:], s.size] - 1
            self.start_of = starts[gid][self.pos_at]
            self.end_of = ends[gid][self.pos_at]

    def auc(self, W=None):
        """AUC, total weight of class 1, total weight of class 0 (each per replicate)."""
        W = _weights(W, self.n)
        Ws = W[..., self.order]
        neg = Ws * self.neg_mask
        cum = np.cumsum(neg, axis=-1)
        wpos = Ws[..., self.pos_at]
        if self.ties:
            below = cum[..., self.start_of] - neg[..., self.start_of]
            within = cum[..., self.end_of] - below
            num = np.sum(wpos * (below + 0.5 * within), axis=-1)
        else:
            num = np.sum(wpos * cum[..., self.pos_at], axis=-1)
        w1 = wpos.sum(axis=-1)
        w0 = cum[..., -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / (w1 * w0), w1, w0


def d_a_from_auc(auc, w1, w0):
    """d_a = sqrt(2) z(AUC). A perfect separation is moved half a pair from the edge."""
    auc, w1, w0 = (np.asarray(x, float) for x in (auc, w1, w0))
    eps = 0.5 / (w1 * w0)
    return SQRT2 * norm.ppf(np.clip(auc, eps, 1.0 - eps))


def auc(score, y, W=None):
    return SortedScore(score, y).auc(W)[0]


def d_a(score, y, W=None):
    """Sensitivity without a threshold, from the AUC."""
    return d_a_from_auc(*SortedScore(score, y).auc(W))


def at_edge(score, y, W=None):
    """True where the AUC is 0 or 1: d_a there is a bound, not a measure."""
    a = SortedScore(score, y).auc(W)[0]
    return (a <= 0.0) | (a >= 1.0)


# --- yes/no ------------------------------------------------------------------------

def d_yesno(z, y, W=None):
    """z(hit) + z(correct rejections) at threshold zero, with the log-linear correction of ½."""
    z, y = np.asarray(z, float), np.asarray(y)
    W = _weights(W, z.size)
    one, zero = (y == 1).astype(float), (y == 0).astype(float)
    hit = one * (z > 0)
    cr = zero * (z < 0)
    n1, n0 = W @ one, W @ zero
    p1 = (W @ hit + 0.5) / (n1 + 1.0)
    p0 = (W @ cr + 0.5) / (n0 + 1.0)
    return norm.ppf(p1) + norm.ppf(p0)


# --- within-class moments ------------------------------------------------------------

def _pooled(a, b, y, W):
    """Pooled within-class co-moment of a and b: sum over classes of sum w (a - mean)(b - mean)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    W = _weights(W, a.size)
    total = 0.0
    for c in (0, 1):
        s = np.asarray(y) == c
        Wc = W[..., s]
        sw = Wc.sum(axis=-1)
        ma = (Wc @ a[s]) / sw
        mb = (Wc @ b[s]) / sw
        total = total + (Wc @ (a[s] * b[s])) - sw * ma * mb
    return total


def pearson_within(zh, zm, y, W=None):
    """Correlation of the cues within the true class, pooled over the two classes."""
    shm = _pooled(zh, zm, y, W)
    shh = _pooled(zh, zh, y, W)
    smm = _pooled(zm, zm, y, W)
    return shm / np.sqrt(shh * smm)


def slope_within(z, a, y, W=None):
    """Within-class regression slope of the cue z on the frame a."""
    return _pooled(z, a, y, W) / _pooled(a, a, y, W)


# --- tetrachoric ----------------------------------------------------------------------

def bvn(h, k, r):
    """P(X < h, Y < k) for standard normals with correlation r (Owen 1956), as in turing-meter."""
    h, k, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (h, k, r)))
    h = np.where(h == 0, 1e-12, h)
    k = np.where(k == 0, 1e-12, k)
    s = np.sqrt((1.0 - r) * (1.0 + r))
    beta = np.where(h * k > 0, 0.0, 0.5)
    return (0.5 * norm.cdf(h) + 0.5 * norm.cdf(k)
            - owens_t(h, (k - r * h) / (h * s)) - owens_t(k, (h - r * k) / (k * s)) - beta)


def tetrachoric(p1, p2, p11):
    """Latent correlation of two binary variables; vectorized over replicates."""
    p1, p2, p11 = (np.atleast_1d(np.asarray(x, float)) for x in (p1, p2, p11))
    z1 = norm.ppf(np.clip(p1, 1e-4, 1 - 1e-4))
    z2 = norm.ppf(np.clip(p2, 1e-4, 1 - 1e-4))
    expected = bvn(z1[:, None], z2[:, None], RHO_GRID[None, :])
    expected = np.maximum.accumulate(expected, axis=1)
    out = np.array([np.interp(p11[i], expected[i], RHO_GRID) for i in range(p11.size)])
    return out


def tetrachoric_within(zh, zm, y, W=None):
    """Tetrachoric correlation of the correct answers (threshold zero) within class.

    Computed in each class and averaged with the class weights, as in turing-meter.
    """
    zh, zm, y = np.asarray(zh, float), np.asarray(zm, float), np.asarray(y)
    W = _weights(W, zh.size)
    ok_h = ((zh > 0) == (y == 1)).astype(float)
    ok_m = ((zm > 0) == (y == 1)).astype(float)
    total, weight = 0.0, 0.0
    for c in (0, 1):
        s = y == c
        Wc = W[..., s]
        sw = Wc.sum(axis=-1)
        r = tetrachoric(Wc @ ok_h[s] / sw, Wc @ ok_m[s] / sw, Wc @ (ok_h[s] * ok_m[s]) / sw)
        total = total + r * sw
        weight = weight + sw
    out = total / weight
    return float(out[0]) if np.ndim(W) == 1 else out


# --- vectorized model quantities ---------------------------------------------------------

def tu_app_vec(k, rho):
    k, rho = np.broadcast_arrays(np.asarray(k, float), np.asarray(rho, float))
    inside = (k > 0) & (rho < np.minimum(k, 1.0 / np.where(k > 0, k, 1.0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.sqrt((1.0 + k * k - 2.0 * rho * k) / (1.0 - rho * rho))
    return np.where(inside, val, np.maximum(1.0, k))


# --- the tuned turing --------------------------------------------------------------------

def split_halves(image, y, seed: int = SPLIT_SEED) -> np.ndarray:
    """Half (0 or 1) of each presentation: images split in two, class by class."""
    image, y = np.asarray(image), np.asarray(y)
    ids, first = np.unique(image, return_index=True)
    cls = y[first]
    gen = np.random.default_rng(seed)
    half_of = np.zeros(ids.size, int)
    for c in np.unique(cls):
        members = np.flatnonzero(cls == c)
        perm = gen.permutation(members)
        half_of[perm[1::2]] = 1
    return half_of[np.searchsorted(ids, image)]


def tu_curve(zh, zm, y, grid=model.GRID, W=None):
    """Tu(w) = d_a(z_H + w z_M) / d_a(z_H) on the grid; also d_a(z_H) and the pair's d_a."""
    dh = d_a(zh, y, W)
    dp = np.array([d_a(zh + w * zm, y, W) for w in grid])
    return dp / dh, dh, dp


def tuned_turing(zh, zm, y, half, grid=model.GRID) -> dict:
    """The weight chosen on one half of the images, the turing read on the other, then swapped."""
    zh, zm, y, half = (np.asarray(x) for x in (zh, zm, y, half))
    tus, pairs, ws, dhs, yes = [], [], [], [], []
    for a, b in ((0, 1), (1, 0)):
        ma, mb = half == a, half == b
        t, _, _ = tu_curve(zh[ma], zm[ma], y[ma], grid)
        w = float(grid[int(np.argmax(t))])
        dh_b = float(d_a(zh[mb], y[mb]))
        dp_b = float(d_a(zh[mb] + w * zm[mb], y[mb]))
        tus.append(dp_b / dh_b)
        pairs.append(dp_b)
        ws.append(w)
        dhs.append(dh_b)
        yes.append(float(d_yesno(zh[mb] + w * zm[mb], y[mb])) / float(d_yesno(zh[mb], y[mb])))
    return {"tu_tuned": float(np.mean(tus)), "d_pair_tuned": float(np.mean(pairs)),
            "tu_tuned_halves": tus, "w_tuned_halves": ws, "tu_tuned_yesno": float(np.mean(yes))}


# --- the band -----------------------------------------------------------------------------

def band_from_curve(grid, tu_values) -> dict:
    """Edges where Tu^2 - 1 crosses half its maximum on the grid, linear in ln w.

    An edge that the grid does not reach is open (None), and so is the width.
    """
    grid = np.asarray(grid, float)
    v = np.asarray(tu_values, float) ** 2 - 1.0
    lw = np.log(grid)
    i = int(np.argmax(v))
    vmax = float(v[i])
    out = {"peak_w": float(grid[i]), "peak_tu2m1": vmax, "low": None, "high": None,
           "width": None, "log_width": None}
    if not vmax > 0:
        return out
    half = vmax / 2.0
    for j in range(i - 1, -1, -1):
        if v[j] < half:
            t = (half - v[j]) / (v[j + 1] - v[j])
            out["low"] = float(math.exp(lw[j] + t * (lw[j + 1] - lw[j])))
            break
    for j in range(i + 1, v.size):
        if v[j] < half:
            t = (v[j - 1] - half) / (v[j - 1] - v[j])
            out["high"] = float(math.exp(lw[j - 1] + t * (lw[j] - lw[j - 1])))
            break
    if out["low"] is not None and out["high"] is not None:
        out["width"] = out["high"] / out["low"]
        out["log_width"] = math.log(out["width"])
    return out


def band_predicted(k, rho) -> dict:
    if not model.has_interior_peak(k, rho):
        return {"low": None, "high": None, "width": None, "log_width": None}
    low, high = model.band_edges(k, rho)
    finite = math.isfinite(high)
    return {"low": low, "high": high if finite else None,
            "width": high / low if finite else None,
            "log_width": math.log(high / low) if finite else None}


# --- one pair, point estimates ------------------------------------------------------------

def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def measure_pair(zh, zm, y, half, grid=model.GRID) -> dict:
    """All the protocol's quantities for one pair (person zh, machine zm), cues standardized."""
    zh, zm, y = np.asarray(zh, float), np.asarray(zm, float), np.asarray(y)
    dh, dm = float(d_a(zh, y)), float(d_a(zm, y))
    k = dm / dh
    rho = float(pearson_within(zh, zm, y))
    rho_t = float(tetrachoric_within(zh, zm, y))
    phi, phi_t = float(model.angle_deg(rho)), float(model.angle_deg(rho_t))
    tu_meas, _, dp = tu_curve(zh, zm, y, grid)
    tu_pred = model.curve(k, rho, grid) if k > 0 and abs(rho) < 1 else np.full(grid.size, np.nan)
    tuned = tuned_turing(zh, zm, y, half, grid)
    dyh, dym = float(d_yesno(zh, y)), float(d_yesno(zm, y))
    app = model.tu_app(k, rho) if k > 0 else math.nan
    out = {
        "n": int(y.size), "d_h": dh, "d_m": dm, "k": k, "d_h_yesno": dyh, "d_m_yesno": dym,
        "k_yesno": dym / dyh if dyh > 0 else None,
        "edge_h": bool(at_edge(zh, y)), "edge_m": bool(at_edge(zm, y)),
        "rho": rho, "rho_tetrachoric": rho_t, "phi": phi, "phi_tetrachoric": phi_t,
        "phi_gap": phi - phi_t, "q": float(model.q_factor(phi)),
        "tu_app": app, "pythagoras": model.pythagoras(k),
        "w_star": _f(model.w_star(k, rho)) if k > 0 else None,
        "useful_angle": model.useful_angle(k) if k > 0 else None,
        **tuned,
        "yield_on_machine": tuned["d_pair_tuned"] / dm if dm > 0 else None,
        "tu_max_same_data": float(np.max(tu_meas)), "w_max_same_data": float(grid[int(np.argmax(tu_meas))]),
        "curve_measured": [float(x) for x in tu_meas], "curve_predicted": [float(x) for x in tu_pred],
        "band_measured": band_from_curve(grid, tu_meas), "band_predicted": band_predicted(k, rho),
    }
    out["ratio_tuned_app"] = out["tu_tuned"] / app if app and np.isfinite(app) else None
    return out


# --- bootstrap ------------------------------------------------------------------------------

@dataclass
class Design:
    """Cues of several judges on the same presentations, already standardized."""
    y: np.ndarray
    image: np.ndarray
    cues: dict
    a: np.ndarray | None = None
    half: np.ndarray | None = None
    img_index: np.ndarray = field(init=False)
    n_images: int = field(init=False)

    def __post_init__(self):
        self.y = np.asarray(self.y)
        self.image = np.asarray(self.image)
        ids, self.img_index = np.unique(self.image, return_inverse=True)
        self.n_images = ids.size
        if self.half is None:
            self.half = split_halves(self.image, self.y)


def resample_counts(gen, n_images: int, reps: int) -> np.ndarray:
    """reps x n_images integer counts: images drawn with replacement."""
    draws = gen.integers(0, n_images, size=(reps, n_images))
    flat = draws + (np.arange(reps)[:, None] * n_images)
    return np.bincount(flat.ravel(), minlength=reps * n_images).reshape(reps, n_images).astype(float)


def bootstrap(design: Design, pairs: dict, reps: int = 1000, seed: int = BOOT_SEED,
              chunk: int = 100, grid=model.COARSE, fixed_weights: dict | None = None,
              slopes: bool = False) -> dict:
    """Cluster bootstrap over images, the same resamples for every pair of the design.

    `pairs` maps a name to (person, machine), names of cues. In each resample the tuning
    weight is chosen again on `grid` (the 13 weights of § 3), unless `fixed_weights`
    gives, per pair, the two weights chosen on the whole data. Returns, per pair and per
    judge, arrays with one value per resample.
    """
    y = design.y
    halves = [np.flatnonzero(design.half == h) for h in (0, 1)]
    judges = sorted({j for p in pairs.values() for j in p})
    all_sorted = {j: SortedScore(design.cues[j], y) for j in judges}
    persons = sorted({p[0] for p in pairs.values()})
    half_sorted = {(j, h): SortedScore(design.cues[j][idx], y[idx]) for j in persons for h, idx in enumerate(halves)}
    pair_sorted = {}
    for name, (ph, pm) in pairs.items():
        zh, zm = design.cues[ph], design.cues[pm]
        for h, idx in enumerate(halves):
            # with fixed weights, half h is read only at the weight chosen on the other half
            ws = grid if fixed_weights is None else [fixed_weights[name][1 - h]]
            pair_sorted[(name, h)] = [SortedScore(zh[idx] + w * zm[idx], y[idx]) for w in ws]
    gen = np.random.default_rng(seed)
    out = {}

    def put(key, val):
        out.setdefault(key, []).append(np.asarray(val, float))

    done = 0
    while done < reps:
        bc = min(chunk, reps - done)
        counts = resample_counts(gen, design.n_images, bc)
        W = counts[:, design.img_index]
        Wh = [W[:, idx] for idx in halves]
        da = {j: d_a_from_auc(*all_sorted[j].auc(W)) for j in judges}
        dy = {j: d_yesno(design.cues[j], y, W) for j in judges}
        for j in judges:
            put((j, "d_a"), da[j])
            put((j, "d_yesno"), dy[j])
            if slopes and design.a is not None:
                put((j, "slope"), slope_within(design.cues[j], design.a, y, W))
        dh_half = {(j, h): d_a_from_auc(*half_sorted[(j, h)].auc(Wh[h])) for j in persons for h in (0, 1)}
        for name, (ph, pm) in pairs.items():
            zh, zm = design.cues[ph], design.cues[pm]
            k = da[pm] / da[ph]
            rho = pearson_within(zh, zm, y, W)
            rho_t = tetrachoric_within(zh, zm, y, W)
            phi = model.angle_deg(rho)
            app = tu_app_vec(k, rho)
            dcomb = {h: np.stack([d_a_from_auc(*s.auc(Wh[h])) for s in pair_sorted[(name, h)]])
                     for h in (0, 1)}
            tus, dps = [], []
            cols = np.arange(bc)
            for a_, b_ in ((0, 1), (1, 0)):
                if fixed_weights is None:
                    pick = np.argmax(dcomb[a_] / dh_half[(ph, a_)], axis=0)
                    dp = dcomb[b_][pick, cols]
                else:
                    dp = dcomb[b_][0]
                tus.append(dp / dh_half[(ph, b_)])
                dps.append(dp)
            tuned = (tus[0] + tus[1]) / 2.0
            dpair = (dps[0] + dps[1]) / 2.0
            for key, val in (("k", k), ("k_yesno", dy[pm] / dy[ph]), ("rho", rho), ("rho_tetrachoric", rho_t),
                             ("phi", phi), ("phi_tetrachoric", model.angle_deg(rho_t)),
                             ("q", model.q_factor(phi)), ("tu_app", app), ("tu_tuned", tuned),
                             ("d_pair_tuned", dpair), ("ratio_tuned_app", tuned / app),
                             ("pythagoras", np.sqrt(1.0 + k * k)), ("yield_on_machine", dpair / da[pm]),
                             ("d_h", da[ph]), ("d_m", da[pm])):
                put((name, key), val)
        done += bc
    return {key: np.concatenate(v) for key, v in out.items()}


def ci(sample) -> list | None:
    s = np.asarray(sample, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def attach_intervals(point: dict, reps: dict, name: str) -> dict:
    """Add `<key>_ci` to the point estimates of a pair from its bootstrap arrays."""
    for (n, key), sample in reps.items():
        if n == name:
            point[key + "_ci"] = ci(sample)
    return point


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return None
    return float(spearmanr(x[keep], y[keep]).statistic)


# --- the signed turing and the null (§ 8 bis) -----------------------------------------------
#
# d_a keeps its sign: below an AUC of 0.5 it is negative. The curve is read on the extended
# grid, model.GRID_SIGNED (the 49 weights of § 3 with both signs).

def signed_curve(zh, zm, y, grid=model.GRID_SIGNED, W=None):
    """Signed Tu(w) = d_a(z_H + w z_M) / d_a(z_H) on the extended grid."""
    return tu_curve(zh, zm, y, grid, W)[0]


def _interp_zero(w1, t1, w2, t2):
    return w1 + (0.0 - t1) * (w2 - w1) / (t2 - t1)


def null_from_curve(grid, tu_values, side: int = -1):
    """First weight, going out from w = 0 on one side, where the signed turing reaches zero.

    Linear interpolation in w between the two neighbouring grid points; None if the curve
    never reaches zero on that side of the grid.
    """
    grid, t = np.asarray(grid, float), np.asarray(tu_values, float)
    idx = np.flatnonzero(grid < 0)[::-1] if side < 0 else np.flatnonzero(grid > 0)
    for a, b in zip(idx[:-1], idx[1:]):
        if t[a] > 0 and t[b] <= 0:
            return float(_interp_zero(grid[a], t[a], grid[b], t[b]))
    return None


def null_width_from_curve(grid, tu_values, w0, level: float = 0.5) -> dict:
    """The interval around the measured null where |Tu| < level, by linear interpolation in w.

    Width as the ratio far / near of the edges (the convention of the band) and as a
    difference; an edge outside the grid leaves the width open (None).
    """
    out = {"near": None, "far": None, "width": None, "log_width": None, "width_diff": None}
    if w0 is None:
        return out
    grid, a = np.asarray(grid, float), np.abs(np.asarray(tu_values, float))
    keep = grid * w0 > 0
    order = np.argsort(np.abs(grid[keep]))
    side, vals = grid[keep][order], a[keep][order]      # from near zero outward
    j = int(np.searchsorted(np.abs(side), abs(w0)))
    pts = np.r_[side[:j], w0, side[j:]]                 # the null itself, where |Tu| = 0
    vals = np.r_[vals[:j], 0.0, vals[j:]]
    for i in range(j, 0, -1):                           # toward zero
        if vals[i] < level <= vals[i - 1]:
            t = (level - vals[i]) / (vals[i - 1] - vals[i])
            out["near"] = float(pts[i] + t * (pts[i - 1] - pts[i]))
            break
    for i in range(j, pts.size - 1):                    # away from zero
        if vals[i] < level <= vals[i + 1]:
            t = (level - vals[i]) / (vals[i + 1] - vals[i])
            out["far"] = float(pts[i] + t * (pts[i + 1] - pts[i]))
            break
    if out["near"] is not None and out["far"] is not None:
        out["width"] = out["far"] / out["near"]
        out["log_width"] = math.log(out["width"])
        out["width_diff"] = abs(out["far"] - out["near"])
    return out


def null_predicted(k, rho) -> dict:
    """Null and its width from the measured k and rho."""
    if k == 0 or not abs(rho) < 1:
        return {"w0": None, "near": None, "far": None, "width": None, "log_width": None, "width_diff": None}
    near, far = model.null_edges(k, rho)
    finite = math.isfinite(far)
    return {"w0": model.null_weight(k), "near": near, "far": far if finite else None,
            "width": far / near if finite else None, "log_width": math.log(far / near) if finite else None,
            "width_diff": abs(far - near) if finite else None}


def null_depth_split(zh, zm, y, half) -> dict:
    """|Tu| at the predicted null -1/k, with k from one half and the turing read on the other."""
    zh, zm, y, half = (np.asarray(x) for x in (zh, zm, y, half))
    vals, ws = [], []
    for a, b in ((0, 1), (1, 0)):
        ma, mb = half == a, half == b
        k_a = float(d_a(zm[ma], y[ma])) / float(d_a(zh[ma], y[ma]))
        w = -1.0 / k_a
        t = float(d_a(zh[mb] + w * zm[mb], y[mb])) / float(d_a(zh[mb], y[mb]))
        vals.append(abs(t))
        ws.append(w)
    return {"abs_tu_at_null": float(np.mean(vals)), "abs_tu_at_null_halves": vals, "w0_halves": ws}


def measure_null(zh, zm, y, half, grid=model.GRID_SIGNED, side: int = -1) -> dict:
    """The quantities of E5 for one pair: measured and predicted null, widths, the whole curve."""
    zh, zm, y = np.asarray(zh, float), np.asarray(zm, float), np.asarray(y)
    dh, dm = float(d_a(zh, y)), float(d_a(zm, y))
    k = dm / dh
    rho = float(pearson_within(zh, zm, y))
    phi = float(model.angle_deg(rho))
    t = signed_curve(zh, zm, y, grid)
    pred = model.tu_signed(grid, k, rho)
    peak = model.tu_peak_signed(k, rho)
    w0 = null_from_curve(grid, t, side)
    out = {"n": int(y.size), "d_h": dh, "d_m": dm, "k": k, "rho": rho, "phi": phi,
           "q": float(model.fano_q(k, phi)) if k != 0 else None,
           "tu_peak_signed": peak, "w0": w0, "w0_k": w0 * k if w0 is not None else None,
           "null_measured": null_width_from_curve(grid, t, w0), "null_predicted": null_predicted(k, rho),
           "mae_over_peak": float(np.mean(np.abs(t - pred)) / peak),
           "curve_signed_measured": [float(x) for x in t], "curve_signed_predicted": [float(x) for x in pred]}
    if w0 is not None and k != 0:
        out["w0_ratio"] = w0 / model.null_weight(k)
    out.update(null_depth_split(zh, zm, y, half))
    return out
