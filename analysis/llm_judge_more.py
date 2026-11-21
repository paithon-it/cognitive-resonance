"""E16, readings added after a fourth review (not tests; the files of the confirmation had been seen).

1. Where the loss comes from. The judgment under an opinion is the middle of the two judgments plus or minus half
   their distance: the middle is what the sentence with the opinion does whichever opinion it is, the half distance
   is the push. Three judges that do not exist are built from the same numbers: the judge with no opinion plus the
   push alone, the middle alone, and the two together (the judge as it is).
2. A corrector that knows more: the trace and the judgment it corrects (`opinion.state_corrector`). It still knows
   nothing of the question.
3. The shape of the push: by how far the judgment with no opinion is from the doubt, the width tau of the bell of
   the simulation (`analysis/nonadditive.py`) that fits it, and what that simulation says a linear corrector gives
   back for a push of that size and width, against what it gives back on the judge with the push alone.
4. Before or after: the same judge asked with no opinion, against the judge corrected afterwards.

    python analysis/llm_judge_more.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nonadditive  # noqa: E402
from common import ROOT, write_json  # noqa: E402
from llm_judge import PRIMARY  # noqa: E402

from cognitive_resonance import opinion as op  # noqa: E402
from cognitive_resonance.estimators import d_a  # noqa: E402


def judge(path: Path) -> dict:
    d = np.load(path)
    y, a = d["y"].astype(int), d["a"].astype(float)
    g0, ga, gb = (d[k].astype(float) for k in ("g_none", "g_a", "g_b"))
    n = y.size
    learn, test = np.arange(n // 2), np.arange(n // 2, n)
    half, middle = 0.5 * (ga - gb), 0.5 * (ga + gb)
    g = op.met(ga, gb, a)
    clean = float(d_a(g0[test], y[test]))
    kept = lambda z: float(d_a(z[test], y[test])) / clean                                          # noqa: E731

    def linear(z):
        beta, _ = op.readable_bias(z[learn], a[learn], y[learn])
        return z - beta * a

    push_only = g0 + a * half
    as_it_is, lin = kept(g), kept(linear(g))
    state = op.state_corrector(g, a, y, learn)
    cut0, cut_state = op.best_threshold(g0[learn], y[learn]), op.best_threshold(state[learn], y[learn])
    back = lambda k, base=None: (k - (as_it_is if base is None else base)) / (1.0 - (as_it_is if base is None else base))  # noqa: E731
    shape = op.shape_of_the_push(g0, ga, gb, y, cut0)
    sim = nonadditive.one(shape["rms_push"], shape["tau"], 2026, d=clean, two_valued=True)
    return {"questions": int(n),
            "kept": {"as_it_is": as_it_is, "middle_alone": kept(middle), "push_alone": kept(push_only),
                     "linear": lin, "linear_on_push_alone": kept(linear(push_only)), "state": kept(state)},
            "gives_back": {"linear": back(lin), "state": back(kept(state)),
                           "linear_on_push_alone": back(kept(linear(push_only)), kept(push_only))},
            "right": {"no_opinion": op.right(g0[test], y[test], cut0), "state": op.right(state[test], y[test], cut_state)},
            "shape": shape,
            "simulation_at_that_shape": {"kept_as_it_is": sim["as_it_is"]["kept"], "linear_gives_back": sim["linear_gives_back"]}}


def main() -> None:
    files = sorted((ROOT / "results" / "raw" / "judge").glob("*_test.npz"))
    judges = {f.stem[:-5]: judge(f) for f in files}
    judges = {k: judges[k] for k in sorted(judges, key=lambda k: (k != PRIMARY, k))}
    write_json(ROOT / "results" / "e16_llm_judge_more.json",
               {"what": "readings added after a fourth review; no criterion", "judges": judges})
    for name, j in judges.items():
        k, b, s = j["kept"], j["gives_back"], j["shape"]
        print(name, "| kept: as it is", round(k["as_it_is"], 3), "middle alone", round(k["middle_alone"], 3), "push alone",
              round(k["push_alone"], 3), "| linear", round(k["linear"], 3), "state", round(k["state"], 3),
              "| gives back: linear", round(b["linear"], 2), "state", round(b["state"], 2), "| on push alone: linear",
              round(b["linear_on_push_alone"], 2), "simulation", round(j["simulation_at_that_shape"]["linear_gives_back"], 2),
              "| tau", s["tau"], "rms push", round(s["rms_push"], 2), "bell explains", round(s["variance_explained_by_the_bell"], 2),
              "| right: state", round(100 * j["right"]["state"], 1), "no opinion", round(100 * j["right"]["no_opinion"], 1))
        print("   push by fifth of distance from the doubt:", [(round(p["distance"], 2), round(p["push"], 2)) for p in s["profile"]])


if __name__ == "__main__":
    main()
