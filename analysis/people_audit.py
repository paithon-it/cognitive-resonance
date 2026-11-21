"""E13, two checks file by file, added after the confirmation run (not tests).

1. Is the order of the rows the order of the trials? Where a file has a column that counts the trials, we count how
   often, inside a person, that counter goes back without starting again from its minimum.
2. Is the confidence column a confidence? Where one value alone covers more than four fifths of the trials it is
   more likely a code for "not asked" than a step of a scale.
3. Does the history say something about the stimulus about to arrive? For every person the three traces of E13
   predict the class (slopes from the first half, area under the curve on the second), and we count how often the
   stimulus repeats. In a randomised task the area is one half and the stimulus repeats half of the time.

Then the medians of E13 are read again without the files that fail a check.

    python analysis/people_audit.py [--data data/confidence_db]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people_bias as pb  # noqa: E402
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance import readable  # noqa: E402
from cognitive_resonance.estimators import auc  # noqa: E402

COUNTER = re.compile(r"^n?trials?(_?(num|number|idx|index|no|count|in_block|id|n|order))?$")
AUC_BAND, REPEAT_BAND, BACK_SHARE, ONE_VALUE = 0.02, 0.03, 0.01, 0.80


def counters(path: Path) -> dict:
    """For every person the values of the columns that count the trials, on the rows E13 keeps."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        names = {(c or "").strip().lower(): c for c in reader.fieldnames or []}
        subject, stimulus, response, confidence = (names[c.lower()] for c in pb.COLUMNS)
        found = [c for low, c in names.items() if COUNTER.match(low)]
        out = {}
        for row in reader:
            who = (row[subject] or "").strip()
            values = (pb._label(row[stimulus]), pb._label(row[response]), pb._number(row[confidence]))
            if who == "" or any(v is None for v in values):
                continue
            out.setdefault(who, {c: [] for c in found})
            for c in found:
                out[who][c].append(pb._number(row[c]))
    return out


def goes_back(values: list) -> float | None:
    """Share of the steps in which the counter falls without starting again from its minimum."""
    v = np.array([np.nan if x is None else x for x in values], float)
    if np.isnan(v).any() or v.size < 3:
        return None
    steps = np.diff(v)
    return float(((steps < 0) & (v[1:] > v.min() + 1)).mean())


def person_checks(p: dict) -> dict:
    g_all = readable.graded_judgment(p["response"], p["confidence"])
    a, y = readable.history(p["response"], g_all), p["stimulus"][1:]
    half = y.size // 2
    first, second = slice(0, half), slice(half, None)
    centred = y[first] - y[first].mean()
    slopes = np.linalg.lstsq(a[first] - a[first].mean(axis=0), centred, rcond=None)[0]
    area = float(auc(a[second] @ slopes, y[second])) if 0 < y[second].mean() < 1 else None
    return {"auc": area, "repeats": float((p["stimulus"][1:] == p["stimulus"][:-1]).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "confidence_db"))
    args = ap.parse_args()
    e13 = json.loads((ROOT / "results" / "e13_people_bias.json").read_text())
    files = []
    for entry in e13["files"]:
        if entry["status"] != "included":
            continue
        path = Path(args.data) / entry["file"]
        people, _ = pb.read_file(path)
        keys = [k for k in people if pb.included(people[k])]
        checks = [person_checks(people[k]) for k in keys]
        cols = counters(path)
        back = {}
        for c in next(iter(cols.values())) if cols else []:
            shares = [goes_back(cols[k][c]) for k in keys if k in cols]
            shares = [s for s in shares if s is not None]
            if shares:
                back[c] = {"people": len(shares), "median_share_going_back": float(np.median(shares)),
                           "people_above_one_per_cent": int(sum(s > BACK_SHARE for s in shares))}
        areas = [c["auc"] for c in checks if c["auc"] is not None]
        conf = np.concatenate([people[k]["confidence"] for k in keys])
        values, counts = np.unique(conf, return_counts=True)
        row = {"file": entry["file"], "split": entry["split"], "people": len(keys), "counters": back,
               "auc_of_history_for_stimulus": float(np.median(areas)),
               "stimulus_repeats": float(np.median([c["repeats"] for c in checks]))}
        row["confidence_values"] = int(values.size)
        row["confidence_most_frequent_share"] = float(counts.max() / counts.sum())
        row["confidence_negative_share"] = float((conf < 0).mean())
        row["confidence_doubtful"] = bool(row["confidence_most_frequent_share"] > ONE_VALUE)
        row["order_doubtful"] = bool(back) and all(v["median_share_going_back"] > BACK_SHARE for v in back.values())
        row["history_says_something"] = bool(abs(row["auc_of_history_for_stimulus"] - 0.5) > AUC_BAND
                                             or abs(row["stimulus_repeats"] - 0.5) > REPEAT_BAND)
        files.append(row)
    out = {"what": "checks added after the confirmation run; no criterion", "bands": {"auc": AUC_BAND, "repeats": REPEAT_BAND,
                                                                                      "going_back": BACK_SHARE},
           "files": files}
    for part in ("confirmation", "exploration"):
        rows = [f for f in files if f["split"] == part]
        records = [r for r in e13["people"] if r["split"] == part]
        doubt = {f["file"] for f in rows if f["order_doubtful"] or f["history_says_something"] or f["confidence_doubtful"]}
        kept = [r for r in records if r["file"] not in doubt]
        gains = [r["gain_personal"] for r in kept if r["gain_personal"] is not None]

        def without(flag):
            rs = [r for r in records if r["file"] not in {f["file"] for f in rows if f[flag]}]
            gs = [r["gain_personal"] for r in rs if r["gain_personal"] is not None]
            return {"people": len(rs), "b_median": float(np.median([r["b"] for r in rs])) if rs else None,
                    "gain_median": float(np.median(gs)) if gs else None}
        out[part] = {"files": len(rows), "files_with_a_counter": sum(bool(f["counters"]) for f in rows),
                     "files_order_doubtful": sum(f["order_doubtful"] for f in rows),
                     "files_history_says_something": sum(f["history_says_something"] for f in rows),
                     "files_confidence_doubtful": sum(f["confidence_doubtful"] for f in rows),
                     "files_confidence_negative": sum(f["confidence_negative_share"] > 0 for f in rows),
                     "without_history_files": without("history_says_something"),
                     "without_confidence_files": without("confidence_doubtful"),
                     "files_left": len(rows) - len(doubt), "people_left": len(kept),
                     "auc_median_of_files": float(np.median([f["auc_of_history_for_stimulus"] for f in rows])) if rows else None,
                     "repeats_median_of_files": float(np.median([f["stimulus_repeats"] for f in rows])) if rows else None,
                     "b_median_all": float(np.median([r["b"] for r in records])) if records else None,
                     "b_median_left": float(np.median([r["b"] for r in kept])) if kept else None,
                     "gain_median_left": float(np.median(gains)) if gains else None}
    write_json(ROOT / "results" / "e13_people_audit.json", out)
    for part in ("confirmation", "exploration"):
        print(part, out[part])
    for f in files:
        if f["order_doubtful"] or f["history_says_something"] or f["confidence_doubtful"]:
            print("  ", f["split"][:4], f["file"], f["people"], "auc", round(f["auc_of_history_for_stimulus"], 3), "repeats",
                  round(f["stimulus_repeats"], 3), "order doubtful" if f["order_doubtful"] else "",
                  "confidence doubtful" if f["confidence_doubtful"] else "")


if __name__ == "__main__":
    main()
