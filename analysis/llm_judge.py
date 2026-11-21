"""E16, a judge with a bias nobody planted (protocol § 22).

For every judge of gpu/judge.py (results/raw/judge/<model>_<split>.npz): the bias readable from the opinion it meets,
what a linear corrector learned on half of the questions gives back on the other half against the same judge asked
with no opinion, the push question by question, and how many judgments the corrector needs.

    python analysis/llm_judge.py [validation|test]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance import opinion as op  # noqa: E402

PRIMARY = "qwen2.5-1.5b-instruct"
LARGE, BACK, DOUBT = 0.2, (0.5, 0.95), 1.2
SIZES = (10, 20, 50, 100, 500)
RESAMPLES, DRAWS = 1000, 200


def interval(values) -> dict:
    v = np.array([x for x in values if x is not None], float)
    return {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}


def judge(path: Path) -> dict:
    d = np.load(path)
    y, a = d["y"].astype(int), d["a"].astype(float)
    g0, ga, gb = (d[k].astype(float) for k in ("g_none", "g_a", "g_b"))
    n = y.size
    learn, test = np.arange(n // 2), np.arange(n // 2, n)
    g = op.met(ga, gb, a)
    beta, b = op.readable_bias(g, a, y)
    out = {"questions": int(n), "beta": beta, "b": b, **op.measure(y, g0, ga, gb, a, learn, test), "push": op.push(g0, ga, gb)}
    out["predicted_gain_if_additive"] = float(np.sqrt(1.0 + op.readable_bias(g[test], a[test], y[test])[1]))
    # the threshold: a preference for one letter is not a bias in the sense of the article, it moves the criterion
    cut = op.best_threshold(g0[learn], y[learn])
    out["threshold"] = {"learned": cut, "right_clean_at_zero": op.right(g0[test], y[test]),
                        "right_clean_at_learned": op.right(g0[test], y[test], cut), "mean_of_the_judgment": float(g0.mean())}
    rng = np.random.default_rng(16)
    boot = {"b": [], "kept_as_it_is": [], "kept_corrected": [], "gives_back": [], "right_gain": [],
            "right_gain_with_threshold": []}
    for _ in range(RESAMPLES):
        i = rng.integers(0, n, n)                            # questions resampled; the corrector learns on the first half
        m = op.measure(y[i], g0[i], ga[i], gb[i], a[i], learn, test)
        boot["b"].append(op.readable_bias(op.met(ga[i], gb[i], a[i]), a[i], y[i])[1])
        for k in ("kept_as_it_is", "kept_corrected", "gives_back"):
            boot[k].append(m[k])
        boot["right_gain"].append(m["right_corrected"] - m["right_as_it_is"])
        boot["right_gain_with_threshold"].append(m["right_corrected_with_threshold"] - m["right_as_it_is_with_threshold"])
    out["intervals"] = {k: interval(v) for k, v in boot.items()}
    few = []
    for size in SIZES:
        kept, back = [], []
        for _ in range(DRAWS):
            some = rng.choice(learn, size, replace=False)
            while not 0 < y[some].sum() < size:              # both answers among the judgments the corrector learns from
                some = rng.choice(learn, size, replace=False)
            m = op.measure(y, g0, ga, gb, a, some, test)
            kept.append(m["kept_corrected"])
            back.append(m["gives_back"])
        few.append({"judgments": size, "kept_corrected": float(np.median(kept)), "gives_back": float(np.median(back)),
                    "kept_corrected_p10": float(np.quantile(kept, 0.1))})
    out["with_few_judgments"] = few
    return out


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    files = sorted((ROOT / "results" / "raw" / "judge").glob(f"*_{split}.npz"))
    judges = {f.stem[: -len(split) - 1]: judge(f) for f in files}
    judges = {k: judges[k] for k in sorted(judges, key=lambda k: (k != PRIMARY, k))}            # the primary judge first
    out = {"protocol": "§ 22, E16", "split": split, "judges": judges}
    if PRIMARY in judges and split == "test":
        p = judges[PRIMARY]
        out["criteria"] = {
            "J1": {"outcome": "pass" if p["b"] >= LARGE else "fail", "b": p["b"],
                   "rule": f"for the primary judge the bias readable from the opinion it meets is at least {LARGE}"},
            "J2": {"outcome": "pass" if p["gives_back"] is not None and BACK[0] <= p["gives_back"] <= BACK[1] else "fail",
                   "gives_back": p["gives_back"],
                   "rule": f"a linear corrector learned on half of the questions gives back between {BACK[0]} and {BACK[1]} of "
                           "the sensitivity lost to the opinion, on the other half"},
            "J3": {"outcome": "pass" if p["push"]["most_doubtful_over_clearest"] >= DOUBT else "fail",
                   "ratio": p["push"]["most_doubtful_over_clearest"],
                   "rule": f"the mean push in the most doubtful third of the questions is at least {DOUBT} times that in the "
                           "clearest third"}}
    write_json(ROOT / "results" / ("e16_llm_judge.json" if split == "test" else f"e16_llm_judge_{split}.json"), out)
    for name, j in judges.items():
        print(name, "n", j["questions"], "| B", round(j["b"], 3), "| d clean", round(j["d_clean"], 2), "kept as it is", round(j["kept_as_it_is"], 3),
              "corrected", round(j["kept_corrected"], 3), "gives back", None if j["gives_back"] is None else round(j["gives_back"], 3),
              "| right", round(100 * j["right_clean"], 1), round(100 * j["right_as_it_is"], 1), round(100 * j["right_corrected"], 1),
              "with threshold", round(100 * j["right_clean_with_threshold"], 1), round(100 * j["right_as_it_is_with_threshold"], 1),
              round(100 * j["right_corrected_with_threshold"], 1),
              "| push by doubt", [round(x, 2) for x in j["push"]["mean_by_third_of_doubt"]])
    if "criteria" in out:
        print({k: v["outcome"] for k, v in out["criteria"].items()})


if __name__ == "__main__":
    main()
