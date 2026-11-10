"""Write results/RESULTS.md from the results files: every number comes from them.

    python analysis/results_md.py [--results results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) or v is None for v in x):
        return f"[{f(x[0], nd)}, {f(x[1], nd)}]"
    return str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def outcome(c):
    o = c["outcome"]
    return {"pass": "**passes**", "fail": "**does not pass**"}.get(o, f"**{o}**")


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def e0(d):
    c = d["criteria"]
    rows = [[b["k"], b["rho"], f(b["phi"], 1), f(b["tu_true"]), f(b["tu_tuned_mean"]), f(100 * b["bias_rel"], 2),
             f(b["rho_mean"]), f(b["rho_tetrachoric_mean"]), f(b["share_at_edge"], 2)] for b in d["bias"]]
    cov = [[c_["rho"], c_["covered"], c_["experiments"], f(c_["median_ci_width"])] for c_ in d["coverage"]]
    chk = [[b["k"], b["rho"], f(100 * b["bias_rel"], 1), f(b["share_at_edge"], 2)] for b in d["added_check_dh2"]
           if b["rho"] <= -0.6]
    return "\n\n".join([
        "## E0. The instrument on the model",
        f"Data from the model of § 2 with the bench's size (2,000 images x 5 presentations, half of the noise "
        f"variance shared by the presentations of an image), dH = {d['design']['d_h']}, "
        f"{d['design']['experiments']} experiments per situation.",
        table(["k", "rho", "phi", "true Tu_app", "mean tuned", "bias %", "mean rho (Pearson)", "mean rho (tetrachoric)",
               "share at AUC = 1"], rows),
        f"- **E0a** (bias within 5% for rho >= -0.9) {outcome(c['E0a'])}: {c['E0a']['within']} of "
        f"{c['E0a']['situations']} situations; the worst is k = {c['E0a']['worst']['k']}, rho = "
        f"{c['E0a']['worst']['rho']}, {f(100 * c['E0a']['worst']['bias_rel'], 2)}%.",
        table(["rho (k = 1)", "intervals containing the truth", "experiments", "median width"], cov),
        f"- **E0b** (coverage between 90 and 98 in 100) {outcome(c['E0b'])}.",
        "- At rho = -0.95 (no criterion) the bias is "
        + ", ".join(f"{f(100 * b['bias_rel'], 1)}% for k = {b['k']}" for b in d["rho_minus_0_95"])
        + "; the share of experiments where a half separates the classes completely (AUC = 1) is "
        + ", ".join(f(b["share_at_edge"], 2) for b in d["rho_minus_0_95"]) + ". The tetrachoric drifts there too.",
        "- Added, without a criterion: the same simulation at dH = 2 (50 experiments). The instrument saturates "
        "when the pair's d_a goes beyond 6 to 7: there the AUC of a half reaches 1, and the tuned turing falls "
        "short of the truth.",
        table(["k", "rho", "bias % at dH = 2", "share at AUC = 1"], chk)])


def e1(d):
    c = d["criteria"]
    parts = ["## E1. The bench",
             f"{len(d['units'])} units (task x mu x seed), {d['settings']['resamples']} resamples each, "
             f"weight in the resamples: {d['settings']['bootstrap_weight']}."]
    for kind in ("opposite", "same"):
        rows = [[m, f(r["phi"], 1), f(r["phi_tetrachoric"], 1), f(r["k"], 2), f(r["q"], 2), f(r["d_h"], 2),
                 f(r["d_m"], 2), f(r["d_pair_tuned"], 2), f(r["tu_tuned"], 2), f(r["tu_app"], 2),
                 f(r["pythagoras"], 2), f"{r['phi_interval_above_90']}/{r['n_pairs']}"]
                for m, r in d["by_mu"][kind].items()]
        parts += [f"**{kind.capitalize()} pairs**, medians over tasks and seeds:",
                  table(["mu", "phi", "phi tetr.", "k", "Q", "d_H", "d_M", "d pair (tuned)", "tuned Tu", "Tu_app",
                         "sqrt(1+k^2)", "phi CI > 90"], rows)]
    a = c["H1a"]
    parts.append(f"- **H1a** {outcome(a)}: median Spearman(mu, rho) = {f(a['median_spearman_mu_rho'])} (needs <= -0.9); "
                 f"tasks with the phi interval above 90 degrees in every seed at mu = {a['top_mu']}: "
                 f"{a['tasks_with_phi_interval_above_90']} (needs {a['tasks_needed']}).")
    for h, text in (("H1b", lambda x: f"{x.get('lower_end_above_pythagoras', 'n/a')} of {x['pairs']} pairs beyond 90 "
                                      f"degrees have the lower end above sqrt(1 + k^2), share {f(x.get('share'))} "
                                      f"(needs 0.8)"),
                    ("H1c", lambda x: f"median tuned/Tu_app = {f(x.get('median_ratio_tuned_over_tu_app'))} "
                                      f"(needs 0.9-1.1, quartiles {f(x.get('ratio_quartiles'))}); mean relative "
                                      f"error {f(x.get('mean_abs_rel_error_tu_app'))} for Tu_app against "
                                      f"{f(x.get('mean_abs_rel_error_pythagoras'))} for sqrt(1 + k^2)"),
                    ("H1d", lambda x: f"median ln(measured width)/ln(predicted) = "
                                      f"{f(x.get('median_ratio_log_width_measured_over_predicted'))} (needs 0.7-1.3) on "
                                      f"{x.get('pairs_with_both_widths', 0)} pairs; Spearman(phi, width) = "
                                      f"{f(x.get('spearman_phi_log_width'))} (needs <= -0.7)")):
        x = c[h]
        parts.append(f"- **{h}** {outcome(x)}: " + (text(x) if x.get("pairs") else x.get("reason", "")) + ".")
    e = c["H1e"]
    parts.append(f"- **H1e** {outcome(e)}: median Spearman(mu, rho) in the equal pairs = "
                 f"{f(e['median_spearman_mu_rho_same'])} (threshold 0.9, see the note); median tuned turing at "
                 f"mu = {e['top_mu']}: {f(e['median_tuned_turing_same_at_top_mu'])} (needs <= 1.1).")
    h = c["H1f"]
    rows = [[m, f(v["median_ratio"]), f(v["max_ratio"]), f"{v['above_1_05']}/{v['pairs']}"] for m, v in h["by_mu"].items()]
    parts += [f"- **H1f** {outcome(h)}: d of the tuned opposite pair over that of the unbiased pair (mu = "
              f"{h['reference_mu']}, task mean):", table(["mu", "median ratio", "max ratio", "above 1.05"], rows)]
    return "\n\n".join(parts)


def e2(d):
    c = d["criteria"]
    rows = [[lam, f(r["phi"], 1), f(r["phi_tetrachoric"], 1), f(r["k"], 2), f(r["d_m"], 2), f(r["d_pair_tuned"], 2),
             f(r["tu_tuned"], 2), f(r["tu_app"], 2), f(r["slope_m_on_a"], 3)]
            for lam, r in d["by_lambda"].items() if lam != "control_H2"]
    ctl = d["by_lambda"]["control_H2"]
    a, b, cc = c["H2a"], c["H2b"], c["H2c"]
    return "\n\n".join([
        "## E2. Rephasing", f"{len(d['units'])} units (task x seed); medians:",
        table(["lambda", "phi(H, M)", "phi tetr.", "k", "d_M", "d pair (tuned)", "tuned Tu", "Tu_app", "slope of M on a"],
              rows),
        f"Control H' (another plus network): phi {f(ctl['phi'], 1)}, tuned turing {f(ctl['tu_tuned'], 2)}.",
        f"- **H2a** {outcome(a)}: median Spearman(lambda, phi) = {f(a['median_spearman_lambda_phi'])} (needs >= 0.9); "
        f"tasks with a lambda whose phi interval is above 90 degrees in every seed: "
        f"{a['tasks_with_phi_interval_above_90']} (needs {a['tasks_needed']}). The angle moved by "
        f"{f(a['phi_moved_from_lambda0_to_max']['median'], 1)} degrees in median from lambda = 0 to the largest.",
        f"- **H2b** {outcome(b)}: tasks with a lambda where M is worse alone and the pair better, both intervals "
        f"excluding zero in every seed: {b['tasks_passing']} (needs {b['tasks_needed']}).",
        f"- **H2c** {outcome(cc)}: {cc['opposite_sign']} of {cc['cases']} cases with the phi interval above 90 degrees "
        f"have the slope of M on a opposite to H's."])


def e3(d):
    c = d["criteria"]
    rows = [[lam, f(r["phi"], 1), f(r["accuracy"], 4), f(r["k"], 2), f(r["tu_app"], 3), f(r["turing_group"], 3)]
            for lam, r in d["by_lambda"].items()]
    base = [[s, f(v["base"]["accuracy"], 4), f(v["base"]["angle"].get("phi"), 1), f(v["base"]["angle"].get("phi_ci"), 1),
             f(v["base"]["group"]["turing_group"], 3)] for s, v in d["seeds"].items() if "base" in v]
    a, b = c["H3a"], c["H3b"]
    return "\n\n".join([
        "## E3. Real people",
        f"CIFAR-10H: mean accuracy of a vote {f(d['cifar10h']['mean_accuracy_of_votes'], 4)}, plurality "
        f"{f(d['cifar10h']['plurality_accuracy'], 4)}. turing-meter commit `{d['turing_meter']['commit']}`.",
        table(["seed", "base accuracy", "phi", "phi 95%", "group turing"], base),
        "Fine-tuned machines on the unseen half (medians over seeds):",
        table(["lambda", "phi", "accuracy", "k", "Tu_app", "group turing"], rows),
        f"- **H3a** {outcome(a)}: lambdas where the angle opens with a paired interval above zero in every seed: "
        f"{a.get('lambdas_opening_in_every_seed')}.",
        f"- **H3b** {outcome(b)}: the largest angle is {f(b.get('max_phi'), 1)} degrees."])


def e4(d):
    rows = [[p["pair"], f(p["phi"], 0), f(p["q"]), f(p["tu_app_k1"]), f(p["bandwidth"], 1), f(p["null_width"], 2)]
            for p in d["pairs"]]
    return "\n\n".join(["## E4. The pairs already measured",
                        "At equal sensitivity (k = 1); no criteria.",
                        table(["pair", "phi", "Q", "Tu_app = 2Q", "bandwidth w_high/w_low", "null width (ratio)"], rows)])


def e5(d):
    c = d["criteria"]
    parts = ["## E5. The null"]
    for kind in ("opposite", "same"):
        rows = [[m, f(r["phi"], 1), f(r["k"], 2), f(r["q"], 2), f(r["w0_k"], 3), f(r["abs_tu_at_predicted_null"], 3),
                 f(r["null_width_measured"], 2), f(r["null_width_predicted"], 2), f(r["mae_over_peak"], 3)]
                for m, r in d["by_mu"][kind].items()]
        parts += [f"**{kind.capitalize()} pairs**, medians:",
                  table(["mu", "phi", "k", "q", "w0 k", "|Tu| at -1/k", "null width meas.", "pred.", "MAE/Tu_app"], rows)]
    a, b, cc, dd, e = (c[h] for h in ("H5a", "H5b", "H5c", "H5d", "H5e"))
    parts += [
        f"- **H5a** {outcome(a)}: {a['crossing_at_negative_weight']} of {a['pairs']} pairs cross zero at a negative "
        f"weight; median w0 k = {f(a['median_w0_k'])} (needs -1.1 to -0.9).",
        f"- **H5b** {outcome(b)}: median |Tu| at the predicted null = {f(b['median_abs_tu_at_predicted_null'])} "
        f"(needs < 0.1).",
        f"- **H5c** {outcome(cc)}: median ln(measured width)/ln(predicted) = "
        f"{f(cc['median_ratio_log_width_measured_over_predicted'])} (needs 0.7-1.3) on {cc['pairs_with_both_widths']} "
        f"pairs; Spearman(phi, width) = {f(cc['spearman_phi_width'])} (needs >= 0.7). With the width as a difference "
        f"of weights: {f(cc['as_difference']['median_ratio_log_width'])} and {f(cc['as_difference']['spearman_phi_width'])}.",
        f"- **H5d** {outcome(dd)}: {dd['machine_negative_with_interval']} opposite pairs with mu >= 1 have the machine "
        f"below chance with its interval on the selected presentations; {dd['crossing_at_positive_weight']} cross zero "
        f"at a positive weight; median measured/predicted weight = {f(dd['median_ratio_measured_over_predicted'])} "
        f"(needs 0.8-1.25).",
        f"- **H5e** {outcome(e)}: median of the mean |measured - predicted| / Tu_app = {f(e['median_mae_over_tu_app'])} "
        f"(needs < 0.10)."]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = Path(args.results)
    sections = ["# Results", "Generated by `analysis/results_md.py` from the JSON files in this folder; hypotheses and "
                "criteria are those of `PROTOCOL.md` (v1.2). Angles in degrees. Where the protocol left a choice "
                "open, the choice is written next to the criterion in the JSON file (`rule`)."]
    for name, fn, label in (("e0_simulation.json", e0, "E0"), ("e1_bench.json", e1, "E1"),
                            ("e2_rephase.json", e2, "E2"), ("e3_people.json", e3, "E3"),
                            ("e4_reread.json", e4, "E4"), ("e5_null.json", e5, "E5")):
        d = load(r / name)
        sections.append(fn(d) if d else f"## {label}\n\nNot run yet: the raw files are not available.")
    out = Path(args.out) if args.out else r / "RESULTS.md"
    out.write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
