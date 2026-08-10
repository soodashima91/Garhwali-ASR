#!/usr/bin/env python3
"""
aggregate_class_errors.py  (v2 -- adds per-seed dump + noaug scan)
Aggregates per-phonological-category error rates from class_errors.json.
Scans BOTH results/runs/objective/* and results/runs/noaug/*.
Skips gamma/lambda ablation dirs (seed*_g* / seed*_l*).

Writes:
  results/aggregate/class_errors_summary.csv   (mean+/-std per cat/obj/condition)
  results/aggregate/class_errors_perseed.csv   (one row per seed/cat/obj/condition)
Stdlib + numpy only. Incremental-safe.
"""
import os, re, json, csv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "results", "runs")
OUT_DIR = os.path.join(ROOT, "results", "aggregate")
CONDITIONS = ["objective", "noaug"]
OBJECTIVES = ["standard", "focal", "matra"]
SEED_RE = re.compile(r"^seed\d+$")
CAT_ORDER = ["matra", "indep_vowel", "virama", "nasal",
             "aspirated", "retroflex", "nukta", "_overall"]


def seed_dirs(condition, objective):
    base = os.path.join(RUNS, condition, objective)
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        if SEED_RE.match(name):
            p = os.path.join(base, name, "class_errors.json")
            if os.path.isfile(p):
                out.append((name, p))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    perseed_rows, summary_rows = [], []

    for cond in CONDITIONS:
        for obj in OBJECTIVES:
            dirs = seed_dirs(cond, obj)
            if not dirs:
                continue
            bycat = {}
            for seedname, path in dirs:
                with open(path) as f:
                    d = json.load(f)
                for cat, v in d.items():
                    if not isinstance(v, dict) or "error_rate" not in v:
                        continue
                    rate = float(v["error_rate"]) * 100.0
                    perseed_rows.append({
                        "condition": cond, "objective": obj, "seed": seedname,
                        "category": cat, "error_rate_pct": round(rate, 3),
                        "errors": v.get("errors"), "ref_count": v.get("ref_count"),
                        "insertions": v.get("insertions"),
                    })
                    bycat.setdefault(cat, {"rates": [], "errs": [], "refs": []})
                    bycat[cat]["rates"].append(rate)
                    bycat[cat]["errs"].append(float(v.get("errors", 0)))
                    bycat[cat]["refs"].append(float(v.get("ref_count", 0)))

            present = [c for c in CAT_ORDER if c in bycat] + \
                      [c for c in bycat if c not in CAT_ORDER]
            print(f"\n=== {cond}/{obj}  (n={len(dirs)} seeds) ===")
            for cat in present:
                a = bycat[cat]
                r = np.array(a["rates"])
                mean = float(r.mean())
                std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
                pooled = (sum(a["errs"]) / sum(a["refs"]) * 100.0) if sum(a["refs"]) else float("nan")
                print(f"  {cat:<14} n={len(r)}  {mean:6.2f} +/-{std:5.2f}  pooled {pooled:6.2f}")
                summary_rows.append({
                    "condition": cond, "objective": obj, "category": cat,
                    "n_seeds": len(r), "mean_pct": round(mean, 3),
                    "std_pct": round(std, 3), "pooled_pct": round(pooled, 3)})

    sp = os.path.join(OUT_DIR, "class_errors_summary.csv")
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "objective", "category",
                                          "n_seeds", "mean_pct", "std_pct", "pooled_pct"])
        w.writeheader(); w.writerows(summary_rows)
    pp = os.path.join(OUT_DIR, "class_errors_perseed.csv")
    with open(pp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "objective", "seed", "category",
                                          "error_rate_pct", "errors", "ref_count", "insertions"])
        w.writeheader(); w.writerows(perseed_rows)
    print(f"\n-> {sp}\n-> {pp}")


if __name__ == "__main__":
    main()
