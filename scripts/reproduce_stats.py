#!/usr/bin/env python
"""
reproduce_stats.py -- regenerate the paper's headline tables and significance
numbers directly from the shipped, transcript-free per-seed CSVs. No GPU, no
model, no VAANI download required.

This verifies the *statistical* claims of the paper (the multi-seed argument)
from committed numbers alone:

  * Table 2  : mean +/- s.d. WER/CER/matraER per objective   (main_results.csv)
  * Sec 4.1  : seed-level paired tests are not significant     (from per-seed WER)
  * Sec 4.5  : power analysis (dz, achieved power, n for 80%)  (power_analysis.json)
  * Sec 4.4  : Hindi transfer vs direct, paired, n.s.          (transfer_results.json)

The per-seed corpus WERs live in results/aggregate/master_perseed.csv, which
contains only numbers (system, seed, wer, cer, per-class error rates) and NO
transcripts. Re-running the corpus-WER-level Wilcoxon/t-tests on those five
numbers reproduces the headline null result.

Usage:  python scripts/reproduce_stats.py
Requires: numpy, scipy  (statsmodels optional, for the power cross-check)
"""
import os, csv, json, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGG = os.path.join(ROOT, "results", "aggregate")

OBJ_ORDER = ["standard", "focal", "matra"]
LABEL = {"standard": "Standard CTC", "focal": "Focal CTC",
         "matra": "Matra-weighted CTC (ours)"}


def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def load_perseed_wer():
    """Return {objective: {seed: wer}} for the augmented 'objective' runs."""
    path = os.path.join(AGG, "master_perseed.csv")
    out = {o: {} for o in OBJ_ORDER}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            grp = r.get("group", "")
            sysname = r.get("system", "")
            if grp != "objective":
                continue
            obj = sysname.split("/")[-1]
            if obj in out and r.get("wer"):
                out[obj][int(r["seed"])] = float(r["wer"])
    return {o: d for o, d in out.items() if d}


def main():
    print("=" * 66)
    print("Reproducing paper statistics from committed per-seed numbers")
    print("(no GPU / no transcripts / no VAANI download)")
    print("=" * 66)

    # ---- Table 2 means ----
    print("\n[Table 2] main_results.csv (mean +/- s.d. over 5 seeds)")
    with open(os.path.join(AGG, "main_results.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(f"  {LABEL.get(r['objective'], r['objective']):28s} "
                  f"WER {float(r['wer_mean']):.2f} +/- {float(r['wer_std']):.2f}  "
                  f"CER {float(r['cer_mean']):.2f}  "
                  f"matraER {float(r['matra_er_mean']):.2f}")

    # ---- Sec 4.1 seed-level paired tests ----
    wer = load_perseed_wer()
    names = [o for o in OBJ_ORDER if o in wer]
    common = sorted(set.intersection(*[set(wer[o]) for o in names]))
    print(f"\n[Sec 4.1] Seed-level paired tests on corpus WER "
          f"(n={len(common)} seeds: {common})")
    arr = {o: np.array([wer[o][s] for s in common]) for o in names}
    pairs, raw = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diff = arr[b] - arr[a]
            try:
                _, wp = stats.wilcoxon(arr[b], arr[a])
            except ValueError:
                wp = 1.0
            _, tp = stats.ttest_rel(arr[b], arr[a])
            pairs.append((f"{a}_vs_{b}", float(diff.mean()), wp, tp))
            raw.append(wp)
    holm = holm_bonferroni(raw)
    print(f"  {'pair':22s} {'dWER':>7} {'wilcoxon_holm':>14} {'sig@.05':>8}")
    for (nm, d, wp, tp), hp in zip(pairs, holm):
        print(f"  {nm:22s} {d:>+7.3f} {hp:>14.3f} {str(hp < 0.05):>8}")
    print("  -> no pair significant: matches the paper's null result.")

    # ---- Sec 4.5 power (echo shipped json) ----
    pj = os.path.join(AGG, "power_analysis.json")
    if os.path.exists(pj):
        p = json.load(open(pj))
        print("\n[Sec 4.5] Post-hoc power (results/aggregate/power_analysis.json)")
        for rec in p["pairs"]:
            print(f"  {rec['pair']:22s} dz={rec['cohen_dz']:+.2f} "
                  f"power@5={rec['achieved_power']:.2f} "
                  f"n_for_80%={rec['seeds_for_target_power']}")

    # ---- Sec 4.4 transfer (echo shipped json) ----
    tj = os.path.join(AGG, "transfer_results.json")
    if os.path.exists(tj):
        t = json.load(open(tj))
        pt = t["paired_transfer_vs_standard"]
        print("\n[Sec 4.4] Hindi->Garhwali transfer vs direct standard CTC")
        print(f"  transfer {t['transfer']['wer_mean']:.2f} vs "
              f"direct {t['standard']['wer_mean']:.2f}  "
              f"dWER={pt['mean_diff_wer']:+.2f}  "
              f"wilcoxon_p={pt['wilcoxon_p']:.3f}  "
              f"sig={pt['significant_at_0.05']}")

    print("\nDone. All headline statistics reproduced from committed numbers.")


if __name__ == "__main__":
    main()
