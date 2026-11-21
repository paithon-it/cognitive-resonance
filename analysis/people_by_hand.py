"""E13 for one person, step by step, so that a human can redo it by hand or in a spreadsheet.

It prints every intermediate quantity for one person of one file of the Confidence Database: the steps of the
judgment with their normal scores, the first trials with their traces, the sums of squares of the two regressions,
R2, the adjusted R2 and B; then the slopes learned on the first half and what they do on the second. Nothing here is
used by the analysis: it is a worked example to check analysis/people_bias.py against.

    python analysis/people_by_hand.py data/confidence_db/data_Maniscalco_2017_expt1.csv [SUBJECT] [--csv OUT.csv]

With --csv it also writes the trials of that person, one row per trial, with the judgment and the three traces.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people_bias as pb  # noqa: E402

from cognitive_resonance import readable  # noqa: E402
from cognitive_resonance.estimators import auc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("subject", nargs="?")
    ap.add_argument("--csv")
    args = ap.parse_args()
    people, status = pb.read_file(Path(args.file))
    if people is None:
        raise SystemExit(f"file not read: {status}")
    keys = [k for k in people if pb.included(people[k])]
    who = args.subject or keys[0]
    p = people[who]
    s, r, c = p["stimulus"], p["response"], p["confidence"]
    print(f"file {Path(args.file).name}, person {who}: {s.size} valid trials, {100 * (s == r).mean():.1f}% right, "
          f"class 1 in {100 * s.mean():.1f}% of the trials, {np.unique(c).size} values of confidence")
    g = readable.graded_judgment(r, c)
    print("\n1. The steps of the judgment, from the surest answer for class 0 to the surest for class 1")
    print("   answer  confidence  trials  mean rank  normal score")
    for score in np.unique(g):
        on = g == score
        rank = 0.5 + (g < score).sum() + (on.sum() - 1) / 2 + 0.5
        print(f"   {int(r[on][0]):6d}  {c[on][0]:10g}  {on.sum():6d}  {rank:9.1f}  {score:+.4f}")
    a = readable.history(r, g)
    y, gg = s[1:], g[1:]
    print("\n2. The first twelve trials (the first one has no history and stays out of the regression)")
    print("   trial  class  answer  judgment  previous answer  previous judgment  mean of the last 20")
    print(f"   {1:5d}  {int(s[0]):5d}  {int(r[0]):6d}  {g[0]:+8.4f}")
    for t in range(11):
        print(f"   {t + 2:5d}  {int(y[t]):5d}  {int(r[t + 1]):6d}  {gg[t]:+8.4f}  {a[t, 0]:+15.0f}  {a[t, 1]:+17.4f}  {a[t, 2]:+19.4f}")
    xtx, xty, yty, n = readable.moments(gg, a, y)
    beta = readable.slopes(xtx, xty)
    explained = float(beta @ xty)
    r2 = explained / yty
    adj = 1 - (1 - r2) * (n - 2) / (n - 5)
    print(f"\n3. All the trials but the first: n = {n}")
    print(f"   sum of squares of the judgment around the mean of its class        {yty:.4f}")
    print(f"   slopes on the three traces, at equal class                         {beta[0]:+.5f}  {beta[1]:+.5f}  {beta[2]:+.5f}")
    print(f"   sum of squares explained by the traces                             {explained:.4f}")
    print(f"   R2 = {r2:.5f};  adjusted = 1 - (1 - R2)(n - 2)/(n - 5) = {adj:.5f};  B = adjusted / (1 - adjusted) = {adj / (1 - adj):.5f}")
    half = n // 2
    m1 = readable.moments(gg[:half], a[:half], y[:half])
    b1 = readable.slopes(m1[0], m1[1])
    before, after = gg[half:], gg[half:] - a[half:] @ b1
    a0, a1 = float(auc(before, y[half:])), float(auc(after, y[half:]))
    print(f"\n4. Slopes from the first {half} trials: {b1[0]:+.5f}  {b1[1]:+.5f}  {b1[2]:+.5f}")
    print(f"   area under the curve on the other {n - half}: {a0:.5f} as it is, {a1:.5f} corrected")
    m = readable.person(s, r, c)
    print(f"   sensitivity {m['d_second']:.4f} as it is; gain {m['gain_personal']:.4f}; right answers "
          f"{100 * m['accuracy_before']:.2f}% before, {100 * m['accuracy_after']:.2f}% after")
    print(f"\n   what analysis/people_bias.py gives for this person: B = {m['b']:.5f}, gain = {m['gain_personal']:.4f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "class", "answer", "confidence", "judgment", "previous_answer", "previous_judgment", "mean_of_last_20"])
            w.writerow([1, int(s[0]), int(r[0]), c[0], f"{g[0]:.6f}", "", "", ""])
            for t in range(n):
                w.writerow([t + 2, int(y[t]), int(r[t + 1]), c[t + 1], f"{gg[t]:.6f}", f"{a[t, 0]:.0f}", f"{a[t, 1]:.6f}", f"{a[t, 2]:.6f}"])
        print(f"\n   trials written to {args.csv}")


if __name__ == "__main__":
    main()
