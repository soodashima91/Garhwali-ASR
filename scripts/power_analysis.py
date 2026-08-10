#!/usr/bin/env python
"""
power_analysis.py - post-hoc power & required-seed analysis for the objective comparison.

Answers, honestly, "how sure are we that the objective differences are not significant?"
For each pair of objectives it reports, on PER-SEED corpus WER (the seed is the randomised
unit, matching paired_tests_multiseed):
  - mean WER difference
  - Cohen's dz (paired effect size)
  - achieved (post-hoc) power at the current number of seeds
  - the number of seeds that WOULD be needed for 80% power at the observed effect size

This converts a bare "not significant" into a quantified statement: e.g. "the gap is real-sized
(dz~1) but n=5 gives only ~0.4 power; ~10 seeds would be needed to detect it." That is the
defensible, reviewer-proof framing for a Findings paper -- it neither over-claims significance
nor pretends the effect is zero.

Reads the same per-seed result.json files as aggregate.py.
Usage:  python power_analysis.py --exp objective [--alpha 0.05 --power 0.80]
Writes: results/aggregate/power_analysis.json  (+ console table)

Requires: numpy, scipy, statsmodels.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
try:
    from garhwali_asr import config as C
    RUNS_DIR = C.RUNS_DIR; AGG_DIR = C.AGG_DIR
except Exception:
    RUNS_DIR = os.path.join("results", "runs"); AGG_DIR = os.path.join("results", "aggregate")

OBJ_ORDER = ["standard", "focal", "matra"]

def per_seed_wer(exp, obj):
    """Corpus WER per seed, read straight from result.json (skips variant dirs)."""
    wers = []
    for rj in sorted(glob.glob(os.path.join(RUNS_DIR, exp, obj, "seed*", "result.json"))):
        d = os.path.basename(os.path.dirname(rj))
        if "_" in d.replace("seed", "", 1):
            continue
        try:
            wers.append(json.load(open(rj))["wer"])
        except Exception:
            pass
    return np.array(wers, dtype=float)

def cohen_dz(diff):
    s = diff.std(ddof=1)
    return float(diff.mean() / s) if s > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="objective")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    a = ap.parse_args()

    from statsmodels.stats.power import TTestPower
    tp = TTestPower()

    wer = {o: per_seed_wer(a.exp, o) for o in OBJ_ORDER}
    wer = {o: v for o, v in wer.items() if len(v) >= 2}
    names = list(wer.keys())
    if len(names) < 2:
        print("[power] need >=2 objectives with >=2 seeds"); return

    out = {"exp": a.exp, "alpha": a.alpha, "target_power": a.power,
           "n_seeds": {o: int(len(v)) for o, v in wer.items()}, "pairs": []}

    print(f"\n=== POWER ANALYSIS (per-seed corpus WER, exp={a.exp}, alpha={a.alpha}) ===")
    print(f"{'pair':20s} {'meanDiff':>9} {'dz':>6} {'p(t)':>7} "
          f"{'power@n':>8} {'n_for_'+str(int(a.power*100))+'%':>9}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            ax, by = wer[x], wer[y]
            n = min(len(ax), len(by))
            ax, by = ax[:n], by[:n]
            diff = by - ax
            dz = cohen_dz(diff)
            _, p_t = stats.ttest_rel(by, ax)
            pow_now = float(tp.power(effect_size=abs(dz), nobs=n,
                                     alpha=a.alpha, alternative="two-sided")) if dz != 0 else 0.0
            try:
                n_need = float(tp.solve_power(effect_size=abs(dz), power=a.power,
                                              alpha=a.alpha, alternative="two-sided")) if dz != 0 else float("inf")
            except Exception:
                n_need = float("inf")
            n_need_r = int(np.ceil(n_need)) if np.isfinite(n_need) else None
            rec = {"pair": f"{x}_vs_{y}", "n_seeds": int(n),
                   "mean_diff_wer": float(diff.mean()), "cohen_dz": dz,
                   "paired_t_p": float(p_t),
                   "achieved_power": pow_now,
                   "seeds_for_target_power": n_need_r}
            out["pairs"].append(rec)
            nn = str(n_need_r) if n_need_r is not None else "inf"
            print(f"{rec['pair']:20s} {diff.mean():>+9.3f} {dz:>+6.2f} "
                  f"{p_t:>7.3f} {pow_now:>8.2f} {nn:>9}")

    # one-line honest summary
    big = max(out["pairs"], key=lambda r: abs(r["cohen_dz"]))
    out["summary"] = (
        f"Largest objective gap ({big['pair']}, {big['mean_diff_wer']:+.2f} WER, dz={big['cohen_dz']:+.2f}) "
        f"has achieved power {big['achieved_power']:.2f} at n={big['n_seeds']} seeds; "
        f"~{big['seeds_for_target_power']} seeds would be needed for {int(a.power*100)}% power. "
        f"Differences are therefore reported as not reliably established rather than absent.")
    print("\n" + out["summary"])

    os.makedirs(AGG_DIR, exist_ok=True)
    p = os.path.join(AGG_DIR, "power_analysis.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n-> {p}")

if __name__ == "__main__":
    main()
