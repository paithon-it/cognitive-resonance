"""What a shortcut costs twice: a reading of the files of E1 and E8, added after the results (not a test).

In E1 the frame says something about the class while a network is trained (strength mu) and nothing when it is
tested. At test the frame is a readable bias: removing it (the slope of the cue on the frame, at equal class) gives
back what the bias was covering. What it does not give back is what the network did not learn from the image while it
leaned on the frame. For every network: the sensitivity as it is, the sensitivity with the frame removed, and the
sensitivity of the network of the same task, seed and role trained with no shortcut (mu = 0).

    python analysis/shortcut.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, med, write_json  # noqa: E402

from cognitive_resonance import estimators as est  # noqa: E402
from cognitive_resonance import raw  # noqa: E402

ROLES = ("z_plus", "z_minus", "z_plus2")


def network(d: dict, role: str) -> dict:
    y, a = d["y"], np.asarray(d["a"], float)
    z = est.standardize(d[role], y, d["val_" + role], d["val_y"])
    slope = float(est.slope_within(z, a, y))
    return {"d_as_it_is": float(est.d_a(z, y)), "d_frame_removed": float(est.d_a(z - slope * a, y)),
            "b": slope ** 2 * float(np.var(a)) / float(est.within_class_sd(z - slope * a, y)) ** 2}


def main() -> None:
    units = []
    for path in sorted((ROOT / "results" / "raw" / "bench").glob("*.npz")):
        d = raw.read_bench(path)
        for role in ROLES:
            if role in d and "val_" + role in d:
                units.append({"task": d["task"], "seed": int(d["seed"]), "mu": float(d["mu"]), "role": role,
                              **network(d, role)})
    reference = {(u["task"], u["seed"], u["role"]): u["d_frame_removed"] for u in units if u["mu"] == 0.0}
    for u in units:
        ref = reference.get((u["task"], u["seed"], u["role"]))
        u["d_without_shortcut"] = ref
        if ref:
            u["kept_as_it_is"] = u["d_as_it_is"] / ref
            u["kept_frame_removed"] = u["d_frame_removed"] / ref
            lost = ref - u["d_as_it_is"]
            u["given_back"] = (u["d_frame_removed"] - u["d_as_it_is"]) / lost if lost > 0.05 * ref else None
    by_mu = []
    for mu in sorted({u["mu"] for u in units}):
        us = [u for u in units if u["mu"] == mu and u.get("d_without_shortcut")]
        by_mu.append({"mu": mu, "networks": len(us), "b": med([u["b"] for u in us]),
                      "d_as_it_is": med([u["d_as_it_is"] for u in us]),
                      "d_frame_removed": med([u["d_frame_removed"] for u in us]),
                      "kept_as_it_is": med([u["kept_as_it_is"] for u in us]),
                      "kept_frame_removed": med([u["kept_frame_removed"] for u in us]),
                      "given_back": med([u["given_back"] for u in us if u["given_back"] is not None])
                      if any(u["given_back"] is not None for u in us) else None})
    by_task = []
    for task in sorted({u["task"] for u in units}):
        mus = sorted({u["mu"] for u in units})
        kept = [med([u["kept_frame_removed"] for u in units if u["task"] == task and u["mu"] == m
                     and u.get("d_without_shortcut")]) for m in mus]
        by_task.append({"task": task, "kept_frame_removed_at_strongest": kept[-1],
                        "rank_correlation_with_mu": float(spearmanr(mus, kept).statistic)})
    out = {"what": "reading added after the results; no criterion", "by_mu": by_mu, "by_task": by_task,
           "tasks_where_what_is_learned_falls_with_the_shortcut": sum(t["rank_correlation_with_mu"] <= -0.9 for t in by_task),
           "tasks": len(by_task), "units": units}
    write_json(ROOT / "results" / "shortcut.json", out)
    for r in by_mu:
        print(r["mu"], r["networks"], "B", round(r["b"], 2), "as it is", round(r["kept_as_it_is"], 3), "frame removed",
              round(r["kept_frame_removed"], 3), "given back", None if r["given_back"] is None else round(r["given_back"], 2))
    print([(t["task"], round(t["kept_frame_removed_at_strongest"], 2), round(t["rank_correlation_with_mu"], 2)) for t in by_task])


if __name__ == "__main__":
    main()
