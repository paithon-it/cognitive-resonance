"""E13 a second time: the same analysis written again from the protocol alone (a check added after the first run).

This script was written by a separate agent that was given only § 19 of the protocol and the choices fixed before
the confirmation run, without the numbers seen on the exploration files and without access to
analysis/people_bias.py or to cognitive_resonance.readable. It is kept as it was written, apart from this header and
the path of the data. It does not compute the 2,000 resamplings nor the criterion P3. Its result is in
results/e13_second_implementation.json, and tests/test_v2.py compares it with results/e13_people_bias.json.

    python analysis/people_bias_second.py [DATA_DIR]
"""
import csv, glob, os, hashlib, json, math, sys
import numpy as np
from scipy.stats import norm, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "confidence_db")).rstrip("/") + "/"
OUT = os.path.join(ROOT, "results")

def canon(s):
    s = s.strip()
    if s == "" or s.lower() in ("nan", "na", "null", "none"):
        return None
    try:
        f = float(s)
        if math.isnan(f): return None
        return ("n", f)
    except ValueError:
        return ("s", s)

def read_file(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        hdr = [h.strip().lstrip("﻿").lower() for h in hdr]
        need = ["subj_idx", "stimulus", "response", "confidence"]
        if any(c not in hdr for c in need):
            return None
        ix = [hdr.index(c) for c in need]
        rows = []
        for r in rd:
            if len(r) <= max(ix): continue
            s, st, rs, cf = (r[i] for i in ix)
            s, st, rs = canon(s), canon(st), canon(rs)
            try:
                c = float(cf)
            except ValueError:
                continue
            if s is None or st is None or rs is None or not math.isfinite(c):
                continue
            rows.append((s, st, rs, c))
    return rows

def dprime_auc(score, cls):
    # cls 1 = second class
    n1 = cls.sum(); n0 = len(cls) - n1
    if n1 == 0 or n0 == 0: return np.nan
    rk = rankdata(score)
    auc = (rk[cls == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc = min(max(auc, 1e-9), 1 - 1e-9)
    return math.sqrt(2) * norm.ppf(auc)

def dprime_mom(score, cls):
    a = score[cls == 1]; b = score[cls == 0]
    if len(a) < 2 or len(b) < 2: return np.nan
    sw = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if sw == 0: return np.nan
    return (a.mean() - b.mean()) / sw

def traces(sign, y):
    # rows for trials 1..n-1 ; columns: prev sign, prev judgment, mean of up to 20 prev judgments
    n = len(y)
    T = np.empty((n - 1, 3))
    cs = np.concatenate([[0], np.cumsum(y)])
    for t in range(1, n):
        lo = max(0, t - 20)
        T[t - 1] = (sign[t - 1], y[t - 1], (cs[t] - cs[lo]) / (t - lo))
    return T

def design(cls, T):
    return np.column_stack([cls == 0, cls == 1, T]).astype(float)

def fit_slopes(y, cls, T):
    X = design(cls, T)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b[2:]

def B_of(y, cls, T):
    n = len(y)
    X0 = design(cls, T[:, :0]); X1 = design(cls, T)
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    s0 = r0 @ r0; s1 = r1 @ r1
    R2 = (s0 - s1) / s0
    R2a = 1 - (1 - R2) * (n - 2) / (n - 5)
    return R2a / (1 - R2a)

def person_prep(st, rs, cf, labels, order):
    # labels: (first, second); order: index permutation (identity or shuffled)
    sgn = np.where(rs == labels[1], 1.0, -1.0)
    # dense rank of confidence (0-based) among the person's values
    vals = np.unique(cf)
    rank = np.searchsorted(vals, cf)
    key = sgn * (1 + rank)
    n = len(key)
    y = norm.ppf((rankdata(key) - 0.5) / n)
    cls = (st == labels[1]).astype(int)
    sgn, y, cls = sgn[order], y[order], cls[order]
    return sgn, y, cls

def halves(m):
    h = m // 2
    return h

def gains(y, cls, T, everyone_slopes=None):
    # y, cls, T are trials 1..n-1
    m = len(y); h = m // 2
    out = {}
    b = fit_slopes(y[:h], cls[:h], T[:h])
    y2, c2, T2 = y[h:], cls[h:], T[h:]
    d_raw = dprime_auc(y2, c2); m_raw = dprime_mom(y2, c2)
    # B1, predicted G2
    out["B1"] = B_of(y[:h], cls[:h], T[:h])
    ok = d_raw >= 0.2
    okm = m_raw >= 0.2
    out["guard_auc"] = not ok
    corr = y2 - T2 @ b
    out["G"] = dprime_auc(corr, c2) / d_raw if ok else np.nan
    out["Gm"] = dprime_mom(corr, c2) / m_raw if okm else np.nan
    if everyone_slopes is not None:
        corr2 = y2 - T2 @ everyone_slopes
        out["Ge"] = dprime_auc(corr2, c2) / d_raw if ok else np.nan
    return out, (y, cls, T)

def centre(y, cls, T):
    y = y.copy(); T = T.copy()
    for c in (0, 1):
        k = cls == c
        if k.any():
            y[k] -= y[k].mean(); T[k] -= T[k].mean(0)
    return y, T

def analyse_file(name, rows, shuffle_rng_seed):
    arr_s = [r[0] for r in rows]
    st = np.array([hash_label(r[1]) for r in rows]); rs = np.array([hash_label(r[2]) for r in rows])
    cf = np.array([r[3] for r in rows])
    info = {"name": name}
    labs = sorted(set(st)), sorted(set(rs))
    if len(labs[0]) != 2 or labs[0] != labs[1]:
        info["why"] = "not two-answer"; return None, info
    labels = labs[0]
    if len(np.unique(cf)) < 3: info["why"] = "conf<3 values"; return None, info
    corr = rs == st
    if not (cf[corr].mean() > cf[~corr].mean()) if (corr.any() and (~corr).any()) else True:
        info["why"] = "conf not oriented"; return None, info
    subj = np.array(arr_s, dtype=object)
    # group by subject keeping file order
    idxs = {}
    for i, s in enumerate(arr_s):
        idxs.setdefault(s, []).append(i)
    people = []
    for s, ii in idxs.items():
        ii = np.array(ii)
        n = len(ii)
        if n < 400: continue
        acc = (rs[ii] == st[ii]).mean()
        if not (0.55 <= acc <= 0.95): continue
        if min((st[ii] == l).mean() for l in labels) < 0.2: continue
        if len(np.unique(cf[ii])) < 3: continue
        people.append((s, ii))
    if len(people) < 10:
        info["why"] = f"only {len(people)} people"; info["npeople"] = len(people); return None, info
    P = []
    for s, ii in people:
        sg, y, cls = person_prep(st[ii], rs[ii], cf[ii], labels, np.arange(len(ii)))
        T = traces(sg, y)
        P.append(dict(sg=sg, y=y, cls=cls, T=T, subj=s, ii=ii))
    # everyone slopes: from first halves, centred
    firsts = []
    for p in P:
        y, cls, T = p["y"][1:], p["cls"][1:], p["T"]
        h = len(y) // 2
        yc, Tc = centre(y[:h], cls[:h], T[:h])
        firsts.append((yc, Tc))
    res = []
    for k, p in enumerate(P):
        y, cls, T = p["y"][1:], p["cls"][1:], p["T"]
        Y = np.concatenate([f[0] for j, f in enumerate(firsts) if j != k])
        TT = np.vstack([f[1] for j, f in enumerate(firsts) if j != k])
        be = np.linalg.lstsq(TT, Y, rcond=None)[0]
        r = {"B": B_of(y, cls, T)}
        g, _ = gains(y, cls, T, be)
        r.update(g)
        # shuffled control
        seed = int(hashlib.sha1((name.lower() + "|" + str(p["subj"])).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(p["ii"]))
        sg, ys, cs = person_prep(st[p["ii"]], rs[p["ii"]], cf[p["ii"]], labels, order)
        Ts = traces(sg, ys)
        r["Bs"] = B_of(ys[1:], cs[1:], Ts)
        gs, _ = gains(ys[1:], cs[1:], Ts)
        r["Gs"] = gs["G"]
        r["n"] = len(y)
        res.append(r)
    info["npeople"] = len(res)
    # diagnostics
    st01 = (st == labels[1]).astype(float)
    info["diag"] = dict(nconf=int(len(np.unique(cf))), cmin=float(cf.min()), cmax=float(cf.max()),
                        stim_lag1=float(np.corrcoef(st01[:-1], st01[1:])[0, 1]))
    return res, info

_lab = {}
def hash_label(l):
    return l[1] if l[0] == "n" else l[1]
# labels: numbers as floats, strings as strings; mixed types are not comparable -> use str
def hash_label(l):
    return str(l[1])

def pct(a, q): return float(np.nanpercentile(a, q))
def med(a): return float(np.nanmedian(a))

def summarise(results, names):
    allp = [r for f in names for r in results[f]]
    g = lambda k: np.array([r[k] for r in allp], float)
    out = {"n_people": len(allp)}
    for k, lab in [("B", "B"), ("Bs", "B_shuffled"), ("G", "G_personal_auc"), ("Gm", "G_personal_moments"),
                   ("Ge", "G_everyone"), ("Gs", "G_personal_shuffled")]:
        a = g(k)
        out["median_" + lab] = med(a)
        out["p10_" + lab] = pct(a, 10); out["p90_" + lab] = pct(a, 90)
        out["n_nan_" + lab] = int(np.isnan(a).sum())
    out["guard_excluded_auc"] = int(sum(r["guard_auc"] for r in allp))
    G = g("G"); Ge = g("Ge"); k = ~np.isnan(G) & ~np.isnan(Ge)
    out["share_personal_gt_everyone"] = float((G[k] > Ge[k]).mean()) if k.any() else None
    out["n_compared"] = int(k.sum())
    out["median_diff_personal_minus_everyone"] = float(np.median(G[k] - Ge[k])) if k.any() else None
    # bootstrap files
    rng = np.random.default_rng(12345)
    fl = [np.array([r["B"] for r in results[f]]) for f in names]
    fs = [np.array([r["Bs"] for r in results[f]]) for f in names]
    if fl:
        bm, bs = [], []
        for _ in range(1000):
            ch = rng.integers(0, len(fl), len(fl))
            bm.append(np.median(np.concatenate([fl[i] for i in ch])))
            bs.append(np.median(np.concatenate([fs[i] for i in ch])))
        out["median_B_CI95"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        out["median_B_shuffled_CI95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out

def main():
    files = sorted(glob.glob(D + "data_*.csv"))
    res = {}; incl = {"conf": [], "expl": []}; excl = {}; diag = {}; read = {"conf": 0, "expl": 0}
    for p in files:
        name = os.path.basename(p)
        grp = "expl" if hashlib.sha1(name.lower().encode()).hexdigest()[0] in "012" else "conf"
        read[grp] += 1
        rows = read_file(p)
        if rows is None:
            excl[name] = "missing columns"; continue
        r, info = analyse_file(name, rows, 0)
        if r is None:
            excl[name] = info.get("why"); continue
        res[name] = r; incl[grp].append(name); diag[name] = info["diag"]
    out = {"counts": {g: {"files_read": read[g], "files_included": len(incl[g]),
                          "people": sum(len(res[f]) for f in incl[g])} for g in read}}
    for g in read:
        out[g] = summarise(res, incl[g])
        out[g]["files"] = {f: len(res[f]) for f in incl[g]}
    out["excluded_files"] = {f: w for f, w in excl.items()}
    out["diag"] = diag
    json.dump(out, open(os.path.join(OUT, "e13_second_implementation.json"), "w"), indent=1)
    print(json.dumps({g: {k: out[g][k] for k in ("n_people", "median_B", "median_G_personal_auc")} for g in read}, indent=1))

main()
