"""The sections of results/RESULTS.md for protocol v2.1: E6 to E9, and the readings of old data.

Used by `results_md.py`; every number comes from the JSON files.
"""
from __future__ import annotations

import statistics

V2 = (("e6_trust.json", "E6"), ("e8_chart.json", "E8"), ("e7_views.json", "E7"), ("e9_shape.json", "E9"),
      ("e10_trace.json", "E10 and E11"), ("e12_yesno.json", "E12"))


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def outcome(c):
    o = c["outcome"]
    return {"pass": "**passes**", "fail": "**does not pass**"}.get(o, f"**{o}**")


def criteria(c):
    """One line per criterion: outcome, the rule as written in the JSON, and its scalar numbers."""
    out = []
    for h, v in c.items():
        shown = []
        for k, x in v.items():
            if k in ("outcome", "rule", "by_task", "detail", "spearman_by_unit", "reason") or isinstance(x, list):
                continue
            if isinstance(x, dict):
                if any(isinstance(y, (dict, list)) for y in x.values()):
                    flat = "; ".join(f"{a} " + ", ".join(f"{b}: {f(z)}" for b, z in y.items()) for a, y in x.items()
                                     if isinstance(y, dict))
                    shown.append(f"{k.replace('_', ' ')} ({flat})")
                else:
                    shown.append(f"{k.replace('_', ' ')} " + ", ".join(f"{a}: {f(b)}" for a, b in x.items()))
            else:
                shown.append(f"{k.replace('_', ' ')} {f(x)}")
        rule = v.get("rule") or v.get("reason") or ""
        out.append(f"- **{h}** {outcome(v)}: {rule}." + (f" Numbers: {'; '.join(shown)}." if shown else ""))
    return "\n\n".join(out)


def summary_rows(load, r) -> list:
    rows = []
    for name, label in V2:
        d = load(r / name)
        if d is None:
            rows.append([label, "registered, not run yet", ""])
            continue
        passed = [h for h, v in d["criteria"].items() if v["outcome"] == "pass"]
        failed = [h for h, v in d["criteria"].items() if v["outcome"] != "pass"]
        rows.append([label, ", ".join(passed) or "none", ", ".join(failed) or "none"])
    return rows


def identity(d):
    rows = [[f(r["mu"], 2), r["pairs"], f(r["phi"], 1), f(r["auc_gap_median"]), f(r["auc_gap_max_abs"]),
             f(r["excess_kurtosis_of_the_sum"], 2)] for r in d["by_mu"]]
    return "\n\n".join([
        "## E1 read again: how much of it is an identity",
        "Added with protocol v2.0, descriptive, no criterion. With sensitivity as a signal-to-noise ratio (difference "
        "of the class means over the pooled within-class standard deviation) the formula of § 2 is algebra on the "
        f"sample moments. On the {d['pairs']} opposite pairs of E1, at five weights, measured and predicted turings "
        f"differ at most by {d['snr_identity_max_relative_error']:.1e} in relative terms. With d_a from the AUC, as in "
        f"E1, the largest gap at weight 1 is {f(d['auc_gap_max_abs'])}: that gap is what H1c and H1d measure, that "
        "is how far the cues are from the Gaussian model.",
        table(["mu", "pairs", "phi", "AUC gap at w = 1 (median)", "largest |gap|", "excess kurtosis of the sum"], rows)])


def _trust_table(d):
    cols = ["d_a_alone", "accuracy_alone", "phi", "w_star_model", "raw_peak", "tu_tuned", "tu_std0.5", "tu_std1",
            "tu_std2", "tu_std8", "tu_face", "accuracy_face"]
    rows = [[r["machine"], f(r["built_for_weight"], 1) if r.get("built_for_weight") is not None else ""]
            + [f(r.get(c), 3 if c.startswith(("tu", "acc")) else 2) for c in cols] for r in d["by_machine"]]
    return table(["machine", "built for w", "d_a alone", "right alone", "phi", "w* (model)", "peak at raw w",
                  "tuned Tu", "Tu at w 0.5", "Tu at w 1", "Tu at w 2", "Tu at w 8", "Tu at face value",
                  "right at face value"], rows)


def e6_old(d):
    cross = sorted(c[0] for c in d["crossing_weight_partner_vs_soloist"] if c)
    return "\n\n".join([
        "## E2 ter read again at a fixed weight (exploratory)",
        "The files of E2 ter, read with `analysis/trust.py --ter`: the same pairs, at a weight fixed beforehand "
        "instead of the tuned one. This is where the hypotheses of E6 come from: it is not a test. Weights 0.5 to 8 "
        "are on the standardized scale of § 3; \"face value\" is the sum of the raw log-odds.",
        _trust_table(d), criteria({h: d["criteria"][h] for h in ("F1", "F2", "F3", "F4")}),
        "Standardized weight at which the pair with the branch and the pair with the soloist are worth the same: "
        f"median {f(cross[len(cross) // 2], 2)} (from {f(cross[0], 2)} to {f(cross[-1], 2)}, {len(cross)} units). "
        "Below it the built partner wins, above it the soloist."])


def e6(d):
    return "\n\n".join([
        "## E6. Trust made to measure",
        f"Protocol § 12 (v2.1), registered before the run. {len(d['units'])} units (task x seed), on seeds never used "
        "before. M0plus is the control soloist; Mramo_w is the frozen M0 plus a frame-only branch trained on the sum "
        "with H at weight w. The primary criteria are F1 and F5.",
        _trust_table(d), criteria(d["criteria"])])


def chart_old(d):
    rows = []
    for k, s in d["summary"].items():
        rows.append([k, s["units"], f(s["b"], 2), f(s["r"], 2), f(s["d_h_clean"], 2), f(s["d_m"], 2),
                     f(s["theorem_d1"], 2), f(s["measured_d1"], 2), f(s["theorem_d2"], 2),
                     f(s.get("measured_d2_partner"), 2), f(s.get("theorem_d_blind"), 2),
                     f(s.get("measured_d_blind"), 2), f(s["strength"], 2), f(s["blind_threshold"], 2),
                     f(s["gain_over_d1_percent"], 1)])
    return "\n\n".join([
        "## The theorem of the chart on E2 bis and E2 ter (a reading of old data)",
        "`analysis/chart.py --e2`: per unit, the slope of H's cue on the frame gives the bias B and what is left is "
        "the clean cue h; from h and the soloist come d_h, d_m and r, and the theorem (protocol § 14) says what the "
        "pair is worth with the soloist (D1), with the soloist plus something that reads the frame (D2) and with a "
        "blind corrector (d_h). Medians over the units. Not a test: these runs came before the theorem.",
        table(["runs", "units", "B", "r", "d_h clean", "d_m", "D1 theorem", "D1 measured", "D2 theorem",
               "pair with Mramo", "blind, theorem", "blind, measured", "d_m / d_h", "threshold", "D2 over D1, %"], rows)])


def _e8_right_answers(d):
    """The pairs of E8 in right answers at face value, with the differences unit by unit. A reading added later."""
    rows = []
    for v in sorted({u["v"] for u in d["units"]}, reverse=True):
        acc = [u["accuracy"] for u in d["units"] if u["v"] == v]
        med = lambda k: f(statistics.median(a[k] for a in acc), 3)
        gain = [a["H+Mramo"] - a["H+M"] for a in acc]
        over_blind = [a["H+Mramo"] - a["H+Mblind"] for a in acc]
        clean = [a["H0"] - a["H+Mramo"] for a in acc]
        rows.append([v, len(acc), med("H"), med("H+M"), med("H+Mblind"), med("H+Mramo"), med("H0"),
                     f"{f(statistics.median(gain), 3)} ({f(min(gain), 3)} to {f(max(gain), 3)})",
                     f"{sum(x > 0 for x in over_blind)} of {len(acc)} ({f(min(over_blind), 3)} to {f(max(over_blind), 3)})",
                     f"{sum(x > 0 for x in clean)} of {len(acc)}"])
    return table(["v", "units", "right, H", "right, H+M", "right, H+Mblind", "right, H+Mramo", "right, H0 alone",
                  "H+Mramo minus H+M, median (min to max)", "units where H+Mramo beats H+Mblind (min to max of the difference)",
                  "units where H0 alone beats H+Mramo"], rows)


def e8(d):
    cols = ["accuracy_h", "accuracy_m", "accuracy_mramo", "d_h", "d_m", "b", "r", "d1", "theorem_d1", "d2_partner",
            "d2_three_cues", "theorem_d2", "d_blind", "theorem_d_blind", "d_pair_without_bias", "strength",
            "blind_threshold"]
    rows = [[r["v"], r["units"]] + [f(r[c], 3 if c.startswith("acc") else 2) for c in cols] for r in d["by_v"]]
    return "\n\n".join([
        "## E8. The bench of the chart",
        "Protocol § 14 (v2.1), registered before the run. The judge H sees the whole image; the machine sees the "
        "frame and v per cent of the inside. M is the soloist, Mramo is M plus a frame-only branch trained on the sum "
        "with H, Mblind sees only the frame; H0 is the judge without the bias. \"Theorem\" columns are computed "
        "from quantities measured on H and M of the same unit. Medians over the units. The primary criteria are K2 "
        "and K4.",
        table(["v", "units", "right, H", "right, M", "right, Mramo", "d_a H", "d_a M", "B", "r", "pair with M (D1)",
               "D1 theorem", "pair with Mramo", "best linear use of 3 cues", "D2 theorem", "pair with Mblind",
               "blind, theorem", "pair (H0, M)", "d_m / d_h", "threshold"], rows),
        criteria(d["criteria"]),
        "**Right answers, pair by pair.** A reading added after a third review, with no criterion: the two cues "
        "summed at face value, medians over the units, and the paired differences unit by unit. With a quarter of "
        "the image the partner and the blind machine are level; with half or a quarter, the judge without the "
        "bias, alone, does better than the judge corrected by the partner in every unit.",
        _e8_right_answers(d)])


def e7(d):
    cols = ["accuracy_alone", "d_a_alone", "phi", "tu_tuned", "d_pair_tuned", "pair_minus_soloist_alone", "tu_face"]
    rows = [[r["overlap"], r["machine"]] + [f(r[c], 3 if c.startswith("acc") else 2) for c in cols]
            for r in d["by_overlap_and_machine"]]
    return "\n\n".join([
        "## E7. Different views",
        "Protocol § 13 (v2.1), registered before the run. Each judge sees half of the image; the overlap is the share "
        "of the machine's half that the judge H sees too. M0 is the soloist, Mramo is M0 plus a frame-only branch, "
        "Mpair is trained from scratch on the sum with H. Medians over the units. The primary criteria are V1 and V4.",
        table(["overlap %", "machine", "right alone", "d_a alone", "phi", "tuned Tu", "d pair", "d pair - d_a(M0)",
               "Tu at face value"], rows),
        criteria(d["criteria"])])


def e9(d):
    cols = ["phi", "d_h", "snr_h", "d_a_over_snr_single", "d_pair_tuned", "d_pair_predicted", "ratio_tuned_app",
            "d_a_over_snr_pair", "pair_over_gauss", "excess_kurtosis_single", "accuracy_single"]
    rows = [[f(r["mu"], 2), r["shape"], r["units"]] + [f(r[c], 1 if c == "phi" else 3) for c in cols]
            for r in d["by_mu_and_shape"]]
    return "\n\n".join([
        "## E9. The shape of the noise",
        "Protocol § 15 (v2.1), registered before the run. The two opposite networks of E1 on new seeds; at test the "
        "frame is drawn from four distributions with the same variance. \"Ratio\" is tuned turing over Tu_app, as in "
        "H1c. With the signal-to-noise ratio in place of d_a the formula holds for every shape: largest relative "
        f"gap {d['snr_identity_max_relative_error']:.1e}. Medians over the units.",
        table(["mu", "frame", "units", "phi", "d_a single", "SNR single", "d_a / SNR single", "pair measured",
               "pair predicted", "ratio", "d_a / SNR pair", "pair over pair (gauss)", "excess kurtosis single",
               "right, single"], rows),
        criteria(d["criteria"])])


def _design_tables(d):
    rows = [[f(r["fraction"], 3), f(r["raw_weight"], 2), r.get("partner", ""), f(r["control_turing"], 3),
             f(r.get("partner_turing"), 3), f(r.get("gain"), 3), f"{r.get('units_with_gain', '')}/{r.get('units', '')}",
             f(r.get("gain_if_the_fraction_is_of_its_own_due_weight"), 3)] for r in d["by_fraction"]]
    by_fraction = table(["share f of the weight due", "raw weight", "partner for f", "Tu with the soloist",
                         "Tu with the partner", "gain (median of the paired differences)", "units with a gain",
                         "gain if f is of the partner's own due weight"], rows)
    rows = [[name, f(c["built_for_weight"], 1), f(c["predicted"], 1), f(c["median"], 2),
             f"{c['closed_units']}/{c['units']}", f(c["on_median_curves"], 2)] for name, c in d["break_even"].items()]
    crossing = table(["branch", "built for w", "predicted break-even", "median break-even", "closed in units",
                      "on the median curves"], rows)
    ch = d["choice"]
    rows = [[name, f(r["k"], 2), "yes" if r["holds_floor"] else "no", f(r["worst"], 3), f(r["mean"], 3),
             f(r["minimum_on_grid"], 3)] for name, r in ch["table"].items()]
    choice = table(["machine", "k", "holds the floor", "worst Tu in the interval", "mean Tu in the interval",
                    "lowest Tu on the whole grid"], rows)
    return by_fraction, crossing, choice


def _design_text(d):
    by_fraction, crossing, choice = _design_tables(d)
    ch = d["choice"]
    return [f"Weight due to the control soloist (the peak weight of its pair, median over units): "
            f"{f(d['due_weight']['median'], 2)}. A person who gives the soloist a share f of it uses the raw weight "
            "f times that, with every machine.", by_fraction,
            "Break-even: going up from the weight a branch is built for, the first weight at which its pair falls "
            "below the pair with the control soloist. The model puts it at twice the built weight.", crossing,
            f"The rule of choice for the people between {f(ch['interval_of_fractions'][0], 3)} and "
            f"{f(ch['interval_of_fractions'][1], 0)} of the weight due (raw weights {f(ch['raw_weights'][0], 2)} to "
            f"{f(ch['raw_weights'][1], 2)}): among the machines with k >= 1, the highest worst case. Chosen: "
            f"**{ch['chosen']}**; the same rule on each unit alone makes the same choice in "
            f"{ch['units_with_the_same_choice']} units of {ch['units']}.", choice]


def design_old(d):
    return "\n\n".join(["## E2 ter read as a design problem (exploratory)",
                        "The files of E2 ter, read with `analysis/design.py --ter` (protocol § 12 bis). They hold one "
                        "branch, built for weight 1: the criteria G1 to G4 need the four branches of E6 and are not "
                        "judged here. Not a test.", *_design_text(d)])


def design(d):
    ch = d["choice"]
    met = {True: "met", False: "not met", None: "not evaluable"}[ch["expectation"]["met"]]
    return "\n\n".join(["## E6, the design reading: who a partner is worth it for",
                        "Protocol § 12 bis (v2.2), recorded when the runs of E6 had just started and before any of "
                        "their files was read. It takes no new measure: it reads `e6_trust.json`. The primary "
                        "criteria are G1 and G2.", *_design_text(d),
                        f"Expectation written before, not a criterion ({ch['expectation']['text']}): **{met}**.",
                        criteria(d["criteria"])])


def chart_e7(d):
    rows = []
    for s in d["by_overlap"]:
        rows.append([s["overlap"], s["units"], f(s["b"], 2), f(s["r"], 2), f(s["d_h"], 2), f(s["d_h_clean"], 2),
                     f(s["d_m"], 2), f(s["theorem_d1"], 2), f(s["measured_d1"], 2), f(s["theorem_d2"], 2),
                     f(s["measured_d2_partner"], 2), f(s["median_share_of_predicted_gain_taken"], 2),
                     f(s["measured_d_blind_two_cues"], 2), f"{s['units_where_partner_beats_blind']}/{s['units']}",
                     f"{s['units_where_partner_beats_soloist']}/{s['units']}",
                     f"{s['units_where_blind_beats_soloist']}/{s['units']}", f(s["strength"], 2),
                     f(s["blind_threshold"], 2)])
    return "\n\n".join([
        "## The theorem of the chart on E7 (a reading added after the results)",
        "`analysis/chart.py --views`: the theorem of protocol § 14 on the files of E7, a bench it was not registered "
        "for, where the machine too sees things the judge does not. Added on 20 November 2026, after the results of "
        "E7 were seen: not a test. Per unit, B, d_h, d_m and r come from the judge and the soloist M0 alone, as in "
        "E8. No blind network was trained in E7: the blind corrector here is the best linear use of the judge's cue "
        "and the frame, fitted on half of the images and read on the other half. Medians over the units; the share "
        "of the predicted gain is over the units where the theorem predicts a gain above 0.05.",
        table(["overlap %", "units", "B", "r", "d_a H", "d_h clean", "d_m", "D1 theorem", "D1 measured", "D2 theorem",
               "pair with Mramo", "share of predicted gain taken", "pair with an ideal blind corrector",
               "partner beats blind", "partner beats soloist", "blind beats soloist", "d_m / d_h", "threshold"], rows)])


def push(d):
    rows = []
    for name, s in d["by_branch"].items():
        same, control = s["break_even_against_same_machine"], s["break_even_against_control"]
        rows.append([name, f(s["built_for_weight"], 1), f(s["median_c"], 3), f(s["median_b_over_c"], 3),
                     f(100 * s["median_push_left_at_the_built_weight"], 1), f(100 * s["median_abs_push_left"], 1),
                     f(100 * s["largest_abs_push_left"], 1),
                     f(s["median_measured_peak"], 2), f(s["predicted_break_even"], 0),
                     f(same["median"], 2), f"{same['closed_units']}/{s['units']}",
                     f(control["median"], 2), f"{control['closed_units']}/{s['units']}"])
    return "\n\n".join([
        "## E6, the push of the branch (a reading added after the results)",
        "`analysis/push.py`: added on 20 November 2026, after the results of E6 were seen: not a test. The design "
        "reading found the break-even against the control soloist earlier than twice the built weight. Either the "
        "branch does not cancel the push exactly, or the control is not the same machine without the branch (M0plus "
        "is M0 trained ten more epochs). Per unit, in raw log-odds: b is the within-class slope of the judge's cue on "
        f"the frame (median {f(d['median_b'], 3)}), c is minus the slope of the branch; the pair should peak at b / c "
        "and beat the same machine without the branch up to 2 b / c. The slope of M0 on the frame is "
        f"{f(d['median_abs_slope_of_M0'], 3)} in absolute value. Medians over {d['units']} units.",
        table(["branch", "built for w", "c", "b / c", "push left at the built weight, %", "its size, median, %", "largest, %",
               "measured peak", "predicted break-even", "break-even against M0", "closed in units",
               "break-even against M0plus", "closed in units"], rows),
        "In the median the branch leaves less than half a per cent of the push: it neither under- nor over-corrects. "
        "Unit by unit what is left is a few per cent of the push, more for the branches built for the higher weights. "
        "Against the same machine without the branch the break-even is where the model puts it, a little later at the "
        "higher weights; the earlier break-even of the registered comparison comes from the control, which alone is a "
        "slightly better machine than M0."])


def judgments(d):
    ns = [str(n) for n in d["settings"]["ns"]]
    rows = []
    for s in d["by_v"]:
        rows.append([s["v"], s["units"], f(s["median_b"], 2), f(s["median_r"], 2)]
                    + [f"{f(s['by_n'][n]['share_kept'], 2)} ({f(s['by_n'][n]['predicted'], 2)})" for n in ns])
    return "\n\n".join([
        "## E8, how many judgments a correction needs (a reading added after the results)",
        "`analysis/judgments.py`: added on 20 November 2026, not a test. The slope of the judge's cue on the frame is "
        "estimated from n presentations of the validation set whose answer is known; the judge's test cue is corrected "
        "with it and put beside the soloist. Each cell: the share of the gain that is kept, median over the draws and "
        "the units, and in brackets the share predicted from a bias left of 1 / n. Units where the gain is below 0.2 "
        "are left out.",
        table(["v", "units", "B", "r"] + [f"n = {n}" for n in ns], rows)])


def informative(d):
    rows = []
    for s in d["by_v_and_d_a"]:
        rows.append([s["v"], f(s["target_d_a"], 1), s["units"], f(s["beta"], 2), f(s["beta_deserved"], 2),
                     f(s["excess_share_of_the_push"], 2), f(s["share_of_the_push_removed_by_new_branch"], 2),
                     f(s["theorem_d1"], 2), f(s["measured_d1"], 2), f(s["theorem_d2"], 2),
                     f(s["measured_d2_three_cues"], 2), f(s["pair_with_new_branch"], 2),
                     f(s["pair_with_partner_of_e8"], 2)])
    return "\n\n".join([
        "## E8, when the trace says something true (a reading added after the results)",
        "`analysis/informative.py`: added on 20 November 2026, not a test. The presentations of E8 are resampled so "
        "that the frame has a sensitivity of 0, 0.5 or 1 at test; the cues of the networks are untouched. The theorem "
        "for an informative trace (`cognitive_resonance.chart`) says that only the excess of the judge's weight over "
        "the deserved one is a bias. The new branch is linear in the frame and is trained on the error of the sum on "
        "the resampled validation set; the partner of E8 was trained when the frame said nothing. Medians over the "
        "units. With the whole view (v = 100) the judge has almost no private part and the deserved weight is badly "
        "determined.",
        table(["v", "d_a of the frame", "units", "weight the judge gives", "weight deserved", "excess, share of the push",
               "share of the push the new branch removes", "D1 theorem", "D1 measured", "D2 theorem",
               "best linear use of 3 cues", "pair with the new branch", "pair with the partner of E8"], rows)])


def private_people(d):
    rows = []
    for s in d["by_source"]:
        g = s["if_bias"]
        rows.append([s["source"], s["comparisons"], f(s["d_h"], 2), f(s["d_m"], 2), f(s["rho"], 2),
                     f(s["private_observed"], 2), f(100 * (s["pair_now_over_machine"] - 1), 1),
                     f(100 * (s["pair_now_over_best_alone"] - 1), 1)]
                    + [f"{f(100 * g[b]['median_gain_over_pair_now'], 2 if float(b) < 0.1 else 1)} ({g[b]['comparisons']})"
                       for b in g])
    doc = d["documented_bias"]
    return "\n\n".join([
        "## E4, the theorem on people already measured (a reading, with no measure of the bias)",
        "`analysis/private_people.py`: added on 20 November 2026, not a test. The 74 comparisons of people with a "
        f"language model of turing-meter (commit `{(d['source']['commit'] or '')[:7]}`), with d_H, d_m and rho as "
        "published: the people's biases are in them. P_obs is what the person adds to the machine today; the last "
        "columns say by how much the best pair would grow if a share B / (1 + B) of the person's noise were a bias a "
        "machine can read (in brackets, the comparisons where such a judge can exist). Nothing here measures B. "
        f"The two smallest values are the size of a documented bias: {doc['source']}, a shift of "
        f"{f(100 * doc['shift'], 1)} percentage points around a rate of {f(doc['base_rate'], 2)}, is B = "
        f"{f(doc['b'], 4)} when the noise is everything that makes the answer vary, and about five times as much "
        "for a judge whose own noise is a fifth of that. The larger values are scenarios: with B = 0.25 the person "
        "alone would already improve by 12 percent. Medians over the comparisons.",
        table(["source", "comparisons", "d_H", "d_m", "rho", "P_obs", "best pair over the machine alone, %",
               "best pair over the better of the two alone, %"] + [f"gain if B = {b}, %" for b in d["by_source"][0]["if_bias"]],
              rows)])


def informative_face(d):
    rows = [[s["v"], f(s["target_d_a"], 1), s["units"], f(s["d_soloist"], 2), f(s["d_old"], 2), f(s["d_new"], 2),
             f"{f(s['new_minus_old'], 3)} ({s['units_where_new_beats_old']})",
             f"{f(s['old_minus_soloist'], 3)} ({s['units_where_old_beats_soloist']})",
             f"{f(s['new_minus_soloist'], 3)} ({s['units_where_new_beats_soloist']})"] for s in d["by_v_and_d_a"]]
    return "\n\n".join([
        "## E8 resampled, the pairs at face value (a look at old data for § 17)",
        "`analysis/informative_face.py`: made on 20 November 2026 to size the expectations of protocol § 17, before "
        "the run of E10 to E12. Not a test. The files of E8 are resampled as above; the raw log-odds are summed at "
        "weight 1. \"Old\" is the partner of E8, which removes the whole push; \"new\" is a branch linear in the "
        "frame trained on the resampled validation set. Medians over the units; in brackets the units, of 25, "
        "where the first of the two is ahead.",
        table(["v", "d_a of the frame", "units", "pair with the soloist", "with the old partner", "with the new branch",
               "new - old", "old - soloist", "new - soloist"], rows)])


def trace(d):
    rows, right = [], []
    for s in d["by_v_and_level"]:
        rows.append([s["v"], f(s["level"], 1), s["units"], f(s["theorem_beta"], 2), f(s["theorem_beta_deserved"], 2),
                     f(s["theorem_excess"], 2), f(s["share_removed_new"], 2), f(s["share_removed_old"], 2),
                     f(s["tuned_M"], 2), f(s["tuned_new"], 2), f(s["tuned_old"], 2), f(s["theorem_d2"], 2),
                     f(s["tuned_calibrated_judge_with_soloist"], 2), f(s["share_of_predicted_gain"], 2)])
        right.append([s["v"], f(s["level"], 1), f(s["face_M"], 2), f(s["face_new"], 2), f(s["face_old"], 2),
                      f"{f(s['face_new_minus_old'], 3)} ({s['units_where_new_beats_old_at_face_value']})",
                      f"{f(s['face_old_minus_soloist'], 3)} ({s['units_where_old_beats_soloist_at_face_value']})",
                      f"{f(s['face_new_minus_soloist'], 3)} ({s['units_where_new_beats_soloist_at_face_value']})",
                      f(s["right_H"], 3), f(s["right_C"], 3), f(s["right_H+M"], 3), f(s["right_H+new"], 3),
                      f(s["right_H+old"], 3)])
    return "\n\n".join([
        "## E10 and E11. The chart that tells the truth",
        "Protocol §§ 16-17 (v2.3), registered before the run. At test the frame has a sensitivity of its own (the "
        "level). The judge H was trained where it was worth 2. M is the soloist and does not see the frame; \"new\" "
        "is M plus the branch built for the level of the test, \"old\" is M plus the branch built at level 0, the "
        "partner of E8, which removes the whole push; C is the judge trained with frames of the level. \"Excess\" is "
        "1 - beta_0 / beta of the theorem with the quantities of the unit; \"removed\" is the slope of the branch on "
        "the frame over the slope of the judge. Medians over the units. The primary criteria are T2, X1 and X2.",
        "Tuned pairs (§ 3), and what the branches remove:",
        table(["v", "level", "units", "weight the judge gives", "weight deserved", "excess", "removed by new",
               "removed by old", "pair with M", "pair with new", "pair with old", "D2 theorem", "pair (C, M)",
               "share of the predicted gain taken by new"], rows),
        "At face value (raw log-odds summed, weight 1): sensitivity of the three sums, paired differences (in "
        "brackets the units where the first of the two is ahead) and right answers:",
        table(["v", "level", "H + M", "H + new", "H + old", "new - old", "old - M", "new - M", "right, H", "right, C",
               "right, H + M", "right, H + new", "right, H + old"], right),
        criteria(d["criteria"])])


def yesno(d):
    s = d["summary"]
    rows = [[n, f(x["graded"], 2), f(x["predicted_graded"], 2), f(x["yes_no"], 2), f(x["predicted_yes_no"], 2),
             f(x["predicted_yes_no_corrected"], 2), f"{f(x['yes_no_range'][0], 2)} to {f(x['yes_no_range'][1], 2)}"]
            for n, x in s["by_n"].items()]
    late = s["added_after_the_results"]
    return "\n\n".join([
        "## E12. The push estimated from yes and no",
        "Protocol § 18 (v2.3), registered before the run, on the files of E10 with the machine seeing a quarter of "
        "the image and the frame saying nothing (level 0). The same draws of n validation presentations feed two "
        "estimators of the judge's push: the slope of the graded cue, and a probit regression of the yes-or-no "
        "answer. Share of the gain kept by the corrected judge beside the soloist, median over the draws and over "
        f"the {s['units']} units with a gain to keep. Predicted: a bias left of 1 / n with the graded cue, and of "
        f"kappa / n with yes and no, kappa from the Fisher information (median {f(s['median_kappa'], 2)}, from "
        f"{f(s['kappa_range'][0], 2)} to {f(s['kappa_range'][1], 2)}). The primary criteria are Y1 and Y2.",
        table(["judgments", "kept, graded cue", "predicted", "kept, yes and no", "predicted (registered)",
               "predicted (corrected afterwards)", "yes and no, over the units"], rows),
        criteria(d["criteria"]),
        "**Added after the results.** The registered prediction was too pessimistic: a factor was missing. The probit "
        "slope is in units of the clean noise, and bringing it to the units of the judge's whole cue with the same "
        "estimate shrinks its error; the bias left is kappa / (n (1 + B)^2). That gives a factor of "
        f"{f(late['median_kappa_corrected'], 2)} (median), where the measured one is {f(late['median_factor_measured'], 2)} "
        "at a target of 0.90 and " + ", ".join(f"{f(v, 2)} at {k}" for k, v in late["factor_by_target"].items() if k != "0.9")
        + ". The criteria above use the registered prediction and are untouched."])


def trace_left(d):
    rows = [[r["v"], f(r["level"], 1), r["units"], f(r["push_raw"], 2), f(r["share_removed"], 2), f(r["excess_of_the_theorem"], 2),
             f"{f(r['left_to_the_frame'], 2)} ({f(r['left_range'][0], 2)} to {f(r['left_range'][1], 2)})", f(r["frame_own_slope"], 2),
             r["units_where_old_pair_is_below_the_judge_alone"]] for r in d["by_v_and_level"]]
    return "\n\n".join([
        "## E10, what the branch leaves to the frame (a reading added after the results)",
        "`analysis/trace_left.py`: added on 20 November 2026 after the results of E10, not a test. The branch built "
        "for a level removes the same share of the push at the two views, and what it leaves to the frame, in raw "
        "log-odds per unit of frame, is close to the slope of the frame's own log-likelihood ratio. The excess of the "
        "theorem is that of the best linear use of the three cues and depends on the machine: the two coincide where "
        "the machine sees a quarter of the image, not where it sees everything. Last column: the units, of 25, where "
        "the pair with the old partner at face value gives fewer right answers than the judge alone.",
        table(["v", "level", "units", "push of the judge", "share removed", "excess of the theorem",
               "left to the frame (min to max)", "the frame's own slope", "old pair below the judge alone"], rows)])


def pair_factors(d):
    name = {"curve": "the curve (E1)", "chart": "the chart (E8)", "trace": "the chart, new seeds (E10, level 0)"}
    pct = lambda x, nd: f(100.0 * x, nd) + "%"                                                # noqa: E731
    rows = []
    for s in d["by_bench_and_condition"]:
        g = lambda k: f(s.get(k), 2) if s.get(k) is not None else "n/a"                      # noqa: E731
        rows.append([name[s["bench"]], s["condition"], s["units"], g("resonance"), g("strength"), g("private"),
                     f"{g('tu_partner')} / {g('measured_tu_partner')}", f"{g('tu_blind')} / {g('measured_tu_blind')}",
                     f"{g('tu_soloist')} / {g('measured_tu_soloist')}",
                     f"{f(s['angle_partner'], 0)} / {f(s['measured_angle_partner'], 0)}",
                     g("share_of_log_turing_from_resonance")])
    return "\n\n".join([
        "## The turing of the pair in three factors (a reading added after the results)",
        "`analysis/factors.py`: added on 20 November 2026, not a test. The theorem of the three machines divided by "
        "what the judge is worth alone: with the partner Tu^2 = (1 + B)(q^2 + p), with q = d_m / d_h and "
        "p = (1 - r q)^2 / (1 - r^2) (`cognitive_resonance.chart.pair_factors`). 1 + B is what cancelling the readable "
        "bias is worth, q^2 what the machine knows of the task, p what the judge knows and the machine does not; with "
        "the soloist p shrinks by (1 - r^2) / (1 - r^2 + B), with the blind machine Tu^2 = 1 + B. Each cell gives the "
        "turing from the equation and the measured one (tuned pair over the judge alone); medians over the units. On "
        "the bench of the curve the second network carries the opposite push itself and is read as a partner. Last "
        "column: the share of ln Tu^2 that comes from 1 + B.",
        table(["bench", "condition", "units", "1 + B", "q^2", "p", "partner: equation / measured",
               "blind machine: equation / measured", "soloist: equation / measured", "angle with the partner: equation / measured",
               "share of the turing from 1 + B"], rows),
        "The same pairs seen from the two sides, and the judge that never learned the bias (H0 in E8, C_0 in E10: the same "
        "network trained where the frame says nothing). `q^2 + p` is what the two know together in units of the judge "
        "without its bias: the turing the equation leaves once the bias is gone. What the judge adds to the machine is the "
        "pair with the partner over the soloist alone, minus one. The last columns compare two ways of removing the bias: "
        "in the machine (judge + partner) or in the judge (judge without bias + soloist); right answers of the raw sums, "
        "and in how many units the second is above the first.",
        table(["bench", "condition", "q^2 + p", "judge adds to the machine: equation / measured",
               "judge without bias alone, over the judge", "turing of the judge without bias with the soloist",
               "that pair over the pair judge + partner", "right answers: judge / judge without bias",
               "judge + partner / judge without bias + soloist", "units where the second is above"],
              [[name[s["bench"]], s["condition"], f(s["together"], 2),
                f"{pct(s['judge_adds_to_machine'], 0)} / {pct(s['measured_judge_adds_to_machine'], 0)}",
                f(s["measured_judge_without_bias_over_judge"], 2), f(s["measured_tu_without_bias"], 2),
                f(s["measured_pair_without_bias_over_pair_partner"], 3),
                f"{pct(s['measured_accuracy_judge'], 1)} / {pct(s['measured_accuracy_judge_without_bias'], 1)}",
                f"{pct(s['measured_accuracy_judge_plus_partner'], 1)} / {pct(s['measured_accuracy_without_bias_plus_soloist'], 1)}",
                f"{s['units_without_bias_plus_soloist_above_partner']} of {s['units']}"]
               for s in d["by_bench_and_condition"] if s["bench"] != "curve"])])


SECTIONS = (("identity.json", identity), ("e6_trust_from_e2ter.json", e6_old),
            ("e6_design_from_e2ter.json", design_old), ("e6_trust.json", e6), ("e6_design.json", design),
            ("chart_from_e2.json", chart_old), ("e6_push.json", push), ("e8_chart.json", e8), ("e8_judgments.json", judgments), ("e8_informative.json", informative),
            ("e7_views.json", e7), ("chart_from_e7.json", chart_e7), ("e4_private.json", private_people),
            ("e9_shape.json", e9), ("e8_informative_face.json", informative_face), ("e10_trace.json", trace), ("e10_trace_left.json", trace_left),
            ("e12_yesno.json", yesno), ("pair_factors.json", pair_factors))
