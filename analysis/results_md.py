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
             f(r["tu_tuned"], 2), f(r.get("tu_app_nonneg", r["tu_app"]), 2), f(r["slope_m_on_a"], 3)]
            for lam, r in d["by_lambda"].items() if lam != "control_H2"]
    ctl = d["by_lambda"]["control_H2"]
    a, b, cc = c["H2a"], c["H2b"], c["H2c"]
    return "\n\n".join([
        "## E2. Rephasing", f"{len(d['units'])} units (task x seed); medians:",
        table(["lambda", "phi(H, M)", "phi tetr.", "k", "d_M", "d pair (tuned)", "tuned Tu", "Tu_app (w >= 0)",
               "slope of M on a"],
              rows),
        f"Control H' (another plus network): phi {f(ctl['phi'], 1)}, tuned turing {f(ctl['tu_tuned'], 2)}.",
        f"- **H2a** {outcome(a)}: median Spearman(lambda, phi) = {f(a['median_spearman_lambda_phi'])} (needs >= 0.9); "
        f"tasks with a lambda whose phi interval is above 90 degrees in every seed: "
        f"{a['tasks_with_phi_interval_above_90']} (needs {a['tasks_needed']}). The angle moved by "
        f"{f(a['phi_moved_from_lambda0_to_max']['median'], 1)} degrees in median from lambda = 0 to the largest.",
        f"- **H2b** {outcome(b)}: tasks with a lambda where M is worse alone and the pair better, both intervals "
        f"excluding zero in every seed: {b['tasks_passing']} (needs {b['tasks_needed']}).",
        f"- **H2c** {outcome(cc)}: {cc['opposite_sign']} of {cc['cases']} cases with the phi interval above 90 degrees "
        f"have the slope of M on a opposite to H's."] + _e2_reading(d))


def _e2_reading(d):
    rows = d["by_lambda"]
    lam = [x for x in rows if x not in ("control_H2", "0")]
    out = []
    if lam:
        k_lo = min(rows[x]["k"] for x in lam)
        k_hi = max(rows[x]["k"] for x in lam)
        dm_lo = min(rows[x]["d_m"] for x in lam)
        dm_hi = max(rows[x]["d_m"] for x in lam)
        phi_lo = min(rows[x]["phi"] for x in lam)
        phi_hi = max(rows[x]["phi"] for x in lam)
        tu_lo = min(rows[x]["tu_tuned"] for x in lam)
        out.append(
            "**How to read E2.** With lambda > 0 the machines fall below chance: their median d_M is between "
            f"{f(dm_lo, 2)} and {f(dm_hi, 2)}, so k is negative (median {f(k_lo, 2)} to {f(k_hi, 2)}). The penalty was "
            "satisfied in the cheapest way: the machine turned its own cue upside down. A pair (k, rho) and the pair "
            "(-k, -rho) are the same pair, because one only has to believe the machine in reverse. Turned back, these "
            f"machines meet the person at {f(180 - phi_hi, 0)} to {f(180 - phi_lo, 0)} degrees, not at "
            f"{f(phi_lo, 0)} to {f(phi_hi, 0)}. So the angle above 90 degrees is not a gain there: the tuned turing "
            f"falls from {f(rows['0']['tu_tuned'], 2)} at lambda = 0 to {f(tu_lo, 2)}. H2a fails on the rank "
            "correlation (the angle jumps at the first lambda and then stays flat), not on the size of the angle; "
            "H2b fails because the pair with these machines is worse, not better.")
    e = d.get("exploratory_null_positive_weight")
    if e:
        out.append(
            "**Exploratory, not in the protocol: the null at positive weight.** For the machines below chance with "
            f"their interval ({e['pairs']} pairs), the signed turing of the pair should cross zero at w0 = 1/|k| "
            f"(§ 8 bis). It does in {e['crossing_at_positive_weight']} of {e['pairs']}; median w0 / (1/|k|) = "
            f"{f(e['median_ratio_w0_over_predicted'])} (quartiles {f(e['ratio_quartiles'])}); median |Tu| at the "
            f"predicted null, with k from the other half, {f(e['median_abs_tu_at_predicted_null'])}. The apparent "
            f"turing of (k, rho) and of the reversed pair (-k, -rho), measured again with the cue turned over, are "
            f"the same: median {f(e['median_tu_app'])} and {f(e['median_tu_app_reversed'])}, largest difference "
            f"{e['max_abs_difference_tu_app_reversed']:.1e}.")
    return out


def e3(d):
    c = d["criteria"]
    rows = [[lam, f(r["phi"], 1), f(r["accuracy"], 4), f(r["k"], 2), f(r["tu_app"], 3), f(r["turing_group"], 3)]
            for lam, r in d["by_lambda"].items()]
    base = [[s, f(v["base"]["accuracy"], 4), f(v["base"]["angle"].get("phi"), 1), f(v["base"]["angle"].get("phi_ci"), 1),
             f(v["base"]["angle"]["ideal_phi"] / v["base"]["angle"]["dh"], 3) if v["base"]["angle"].get("ideal_phi")
             else "n/a", f(v["base"]["group"]["turing_group"], 3)] for s, v in d["seeds"].items() if "base" in v]
    a, b = c["H3a"], c["H3b"]
    return "\n\n".join([
        "## E3. Real people",
        f"CIFAR-10H: mean accuracy of a vote {f(d['cifar10h']['mean_accuracy_of_votes'], 4)}, plurality "
        f"{f(d['cifar10h']['plurality_accuracy'], 4)}. turing-meter commit `{d['turing_meter']['commit']}`.",
        table(["seed", "base accuracy", "phi", "phi 95%", "Tu_app", "group turing"], base),
        "Fine-tuned machines on the unseen half (medians over seeds):",
        table(["lambda", "phi", "accuracy", "k", "Tu_app", "group turing"], rows),
        f"- **H3a** {outcome(a)}: lambdas where the angle opens with a paired interval above zero in every seed: "
        f"{a.get('lambdas_opening_in_every_seed')}.",
        f"- **H3b** {outcome(b)}: the largest angle is {f(b.get('max_phi'), 1)} degrees."] + _e3_reading(d))


def _e3_reading(d):
    r = d["by_lambda"]
    keys = list(r)
    if len(keys) < 3:
        return []
    first, mid, last = keys[0], keys[-2], keys[-1]
    return ["**How to read E3.** The angle between people and machine opens as lambda grows: "
            f"{f(r[first]['phi'], 0)} degrees at lambda = {first}, {f(r[mid]['phi'], 0)} at lambda = {mid}, and the "
            "paired intervals exclude zero. But the machine pays for it in accuracy "
            f"({f(r[first]['accuracy'], 3)} to {f(r[mid]['accuracy'], 3)}), so the apparent turing goes down, not up: "
            f"{f(r[first]['tu_app'], 2)} to {f(r[mid]['tu_app'], 2)}. The angle opens because the machine gets worse "
            "where people are right, not because it learns what they miss: on the images where at least 98% of the "
            f"votes are right its accuracy goes from {f(r[first].get('accuracy_where_people_right'), 3)} to "
            f"{f(r[mid].get('accuracy_where_people_right'), 3)}, and where fewer than 80% are it goes from "
            f"{f(r[first].get('accuracy_where_people_split'), 3)} to {f(r[mid].get('accuracy_where_people_split'), 3)} "
            f"(a description, not in the protocol). At lambda = {last} the network collapses: accuracy "
            f"{f(r[last]['accuracy'], 3)}, not far above the 0.1 of chance in ten classes, k = {f(r[last]['k'], 2)}. "
            f"The angle of {f(r[last]['phi'], 0)} degrees that makes H3b fail belongs to a machine that has stopped "
            "working. The group of people with the machine is worth at most "
            f"{f(max(x['turing_group'] for x in r.values()), 2)} turings at any lambda."]


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
        f"(needs 0.8-1.25). In {dd.get('person_at_auc_edge', 'n/a')} of these selections the person separates the "
        f"classes completely (AUC = 1), so its d_a is a bound and k = d_M / d_H is too: the ratio leans on it.",
        f"- **H5e** {outcome(e)}: median of the mean |measured - predicted| / Tu_app = {f(e['median_mae_over_tu_app'])} "
        f"(needs < 0.10)."]
    return "\n\n".join(parts)


def e2bis(d):
    c = d["criteria"]
    bm = d["by_machine"]
    rows = [[m, f(r["d_a"], 2), f(r["slope_on_a"], 3), f(r["phi"], 1), f(r["k"], 2), f(r["tu_tuned"], 3),
             f(r["tu_app_nonneg"], 3), f(r["inv_sin_phi"], 3), f(r["d_pair_tuned"], 2)]
            for m, r in bm.items() if m != "H"]
    lines = ["## E2 bis. The partner trained for the pair (exploratory)",
             "Added with protocol v1.3 after seeing E2 and E3; the expectations were written before this run. "
             f"{len(d['units'])} units; medians. H alone: d_a {f(bm['H']['d_a'], 2)}, slope on a "
             f"{f(bm['H']['slope_on_a'], 3)}.",
             table(["machine", "d_a alone", "slope on a", "phi", "k", "tuned Tu", "Tu_app (w >= 0)", "1/sin(phi)",
                    "d pair"], rows)]
    b1, b2, b3, b4, b5 = (c[h] for h in ("B1", "B2", "B3", "B4", "B5"))
    lines += [
        f"- **B1** {outcome(b1)}: of {b1['units']} units, Mpair above chance in {b1['mpair_above_chance']}, Mframe "
        f"at chance in {b1['mframe_contains_zero']}, slope opposite to H's in {b1['mpair_slope_opposite_to_h']} "
        f"(Mpair) and {b1['mframe_slope_opposite_to_h']} (Mframe).",
        f"- **B2** {outcome(b2)}: tasks where phi opens from M0 to Mpair in every seed: {b2['tasks_passing']} "
        f"(needs {b2['tasks_needed']}); median difference {f(b2['median_phi_difference'], 1)} degrees.",
        f"- **B3** {outcome(b3)}: tasks where Mpair is worse alone and the pair better, in every seed: "
        f"{b3['tasks_passing']} (needs {b3['tasks_needed']}); median differences {f(b3['median_solo_difference'], 2)} "
        f"alone and {f(b3['median_pair_difference'], 2)} in the pair.",
        f"- **B4** {outcome(b4)}: tasks where (H, Mframe) has its interval above 1 in every seed: "
        f"{b4['tasks_with_interval_above_1']} (needs {b4['tasks_needed']}); median tuned / (1/sin phi) = "
        f"{f(b4['median_ratio_tuned_over_inv_sin_phi'])} (needs 0.9-1.1).",
        f"- **B5** {outcome(b5)}: median tuned / Tu_app for Mpair = {f(b5['median_ratio_tuned_over_tu_app'])} "
        f"(needs 0.9-1.1)."]
    mp, mf, m0 = bm["Mpair"], bm["Mframe"], bm["M0"]
    outside = sum(1 for u in d["units"] for m in ("Mpair", "Mframe")
                  if not (u["differences"][m]["pair_ci"][0] <= u["differences"][m]["pair"] <= u["differences"][m]["pair_ci"][1]))
    lines.append(
        "**How to read E2 bis.** The machine trained on the error of the sum did not flip its cue: it learned the "
        f"frame against the person (slope {f(mp['slope_on_a'], 2)}, the person's {f(bm['H']['slope_on_a'], 2)}) and "
        f"almost nothing of the image (d_a {f(mp['d_a'], 2)}, k = {f(mp['k'], 3)}). It behaves like the machine that "
        f"sees only the frame (d_a {f(mf['d_a'], 3)}, phi {f(mf['phi'], 0)} against {f(mp['phi'], 0)} degrees). "
        f"Its angle with the person is a real one, {f(mp['phi'], 0)} degrees, and the model predicts the pair: tuned "
        f"turing {f(mp['tu_tuned'], 2)} against {f(mp['tu_app_nonneg'], 2)} from k and phi, and "
        f"{f(mf['tu_tuned'], 2)} against 1/sin(phi) = {f(mf['inv_sin_phi'], 2)} for the frame-only machine. But "
        f"a person worth {f(bm['H']['d_a'], 2)} gains more from the ordinary machine (pair {f(m0['d_pair_tuned'], 2)}) "
        f"than from the one trained for the pair ({f(mp['d_pair_tuned'], 2)}): here the best partner is still the best "
        "soloist, and B3 fails. B1 fails only because in "
        f"{b1['units'] - b1['mpair_above_chance']} units of {b1['units']} the interval of Mpair's d_a contains zero: there it is at chance, not below.")
    lines.append(
        f"A note on the intervals: in {outside} of {2 * len(d['units'])} paired differences of the pair's sensitivity "
        "the point estimate falls outside its own interval. The point estimate tunes the weight on 49 weights, the "
        "resamples on 13 (§ 3), and here the peak is narrow and falls between two of the 13: the intervals sit low. "
        "No outcome depends on it: the pair's differences are negative in the point estimates too, and an interval "
        "that sits low only makes B4 harder to pass.")
    return "\n\n".join(lines)


def summary(r):
    rows = []
    for name, label in (("e0_simulation.json", "E0"), ("e1_bench.json", "E1"), ("e5_null.json", "E5"),
                        ("e2_rephase.json", "E2"), ("e2bis_rephase.json", "E2 bis"), ("e3_people.json", "E3")):
        d = load(r / name)
        if d is None:
            rows.append([label, "not run yet", ""])
            continue
        passed = [h for h, v in d["criteria"].items() if v["outcome"] == "pass"]
        failed = [h for h, v in d["criteria"].items() if v["outcome"] == "fail"]
        other = [h for h, v in d["criteria"].items() if v["outcome"] not in ("pass", "fail")]
        rows.append([label, ", ".join(passed) or "none", ", ".join(failed + other) or "none"])
    return "## In short\n\n" + table(["experiment", "passes", "does not pass"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = Path(args.results)
    sections = ["# Results", "Generated by `analysis/results_md.py` from the JSON files in this folder; hypotheses and "
                "criteria are those of `PROTOCOL.md` (v1.3), with its deviations. Angles in degrees. Where the protocol left a choice "
                "open, the choice is written next to the criterion in the JSON file (`rule`)."]
    sections.append(summary(r))
    for name, fn, label in (("e0_simulation.json", e0, "E0"), ("e1_bench.json", e1, "E1"),
                            ("e2_rephase.json", e2, "E2"), ("e2bis_rephase.json", e2bis, "E2 bis"),
                            ("e3_people.json", e3, "E3"), ("e4_reread.json", e4, "E4"), ("e5_null.json", e5, "E5")):
        d = load(r / name)
        sections.append(fn(d) if d else f"## {label}\n\nNot run yet: the raw files are not available.")
    out = Path(args.out) if args.out else r / "RESULTS.md"
    out.write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
