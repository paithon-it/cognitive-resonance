"""What a readable bias is worth in right answers, said beforehand by B and measured (a reading added after a fifth
review; not a test). One row for the people of E13 and one for each language model of E16.

Predicted: the right answers a Gaussian judge gains when a bias B is removed, starting from the share of right
answers it has as it is (`chart.right_answers_gained`); for the people also with the cost of learning the correction
from half of their trials with three traces. Measured: the right answers with the corrector less those without, on
trials or questions the corrector has not seen. Beside them, B as the literature implies it for the people: the
extra probability of repeating the last answer in Bocheva, Xue and Rahnev (2026), at three quarters of right answers.

    python analysis/stakes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance import chart  # noqa: E402


def main() -> None:
    res = ROOT / "results"
    e13 = json.loads((res / "e13_people_bias.json").read_text())["confirmation"]
    e16 = json.loads((res / "e16_llm_judge.json").read_text())["judges"]
    latent = json.loads((res / "e13_people_latent.json").read_text())["confirmation"]
    rows = [{"who": "people", "b": e13["b"]["median"], "right_as_it_is": e13["accuracy_median"],
             "predicted": chart.right_answers_gained(e13["accuracy_median"], e13["b"]["median"]),
             "predicted_with_learning": chart.right_answers_gained(e13["accuracy_median"], e13["b"]["median"], 3, e13["trials_median"] / 2),
             "measured": e13["accuracy_change"]["median"], "measured_mean": e13["accuracy_change_mean"]}]
    for name, j in e16.items():
        rows.append({"who": name, "b": j["b"], "right_as_it_is": j["right_as_it_is_with_threshold"],
                     "predicted": chart.right_answers_gained(j["right_as_it_is_with_threshold"], j["b"]),
                     "measured": j["right_corrected_with_threshold"] - j["right_as_it_is_with_threshold"],
                     "not_shown_the_opinion": j["right_clean_with_threshold"] - j["right_as_it_is_with_threshold"]})
    rows.sort(key=lambda r: r["b"])
    out = {"what": "a reading added after a fifth review; no criterion", "rows": rows,
           "b_of_the_people_three_ways": {
               "linear_on_the_graded_judgment": {k: e13["b"][k] for k in ("median", "lo", "hi")},
               "latent_noise_of_a_logistic_model": {k: latent["b_latent"][k] for k in ("median", "lo", "hi")},
               "implied_by_the_repetition_in_the_literature": [chart.from_repetition(0.04, 0.75), chart.from_repetition(0.05, 0.75)]},
           "specialisation_example": chart.specialisation(0.95, 0.55)}
    write_json(res / "stakes.json", out)
    for r in rows:
        print(r["who"], "B", round(r["b"], 3), "| predicted", round(100 * r["predicted"], 2), "| measured", round(100 * r["measured"], 2))


if __name__ == "__main__":
    main()
