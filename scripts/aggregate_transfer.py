#!/usr/bin/env python
"""
aggregate_transfer.py - aggregate the 5-seed Hindi->Garhwali transfer runs and
test them against standard CTC.

The master aggregator (aggregate_all.py) walks objective/ noaug/ baselines/ and does
NOT pick up transfer/, because transfer runs live under transfer/phase2_seed{S}/ with
a two-phase layout. This script handles that layout.

For transfer we read transfer/phase2_seed*/result.json (phase-2 = the Garhwali-finetuned
stage, which is what is comparable to standard CTC). It reports:
  - transfer 5-seed mean +/- std WER/CER
  - standard CTC 5-seed mean +/- std (for reference)
  - paired test (transfer vs standard) on per-seed corpus WER, IF seeds align
  - honest verdict string

IMPORTANT framing (matches the paper):
  This is a curriculum/initialisation question (Hindi fine-tune THEN Garhwali fine-tune),
  NOT the zero-shot or data-selection question studied by Dhasmana et al. (2026).
  Given n=5 and the project's power analysis, expect the transfer-vs-standard gap to be
  small and likely NOT significant; a null here is consistent with prior work that the
  related standard language (Hindi) adds little beyond the dialect data itself.

Usage:  python aggregate_transfer.py
Writes: results/aggregate/transfer_results.json (+ console)
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
try:
    from garhwali_asr import config as C
    RUNS_DIR = C.RUNS_DIR; AGG_DIR = C.AGG_DIR
except Exception:
    RUNS_DIR = os.path.join("results", "runs"); AGG_DIR = os.path.join("results", "aggregate")

SEEDS = [42, 123, 777, 2025, 1234]

def _read(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

def transfer_by_seed():
    """phase-2 WER/CER per seed, keyed by seed int."""
    out = {}
    for rj in glob.glob(os.path.join(RUNS_DIR, "transfer", "phase2_seed*", "result.json")):
        d = _read(rj)
        if not d:
            continue
        # seed from dir name phase2_seed{S}
        dirn = os.path.basename(os.path.dirname(rj))
        try:
            s = int(dirn.replace("phase2_seed", ""))
        except ValueError:
            s = d.get("seed")
        out[s] = {"wer": d.get("wer"), "cer": d.get("cer")}
    return out

def standard_by_seed():
    out = {}
    for rj in glob.glob(os.path.join(RUNS_DIR, "objective", "standard", "seed*", "result.json")):
        dirn = os.path.basename(os.path.dirname(rj))
        if "_" in dirn.replace("seed", "", 1):   # skip ablation variant dirs
            continue
        d = _read(rj)
        if not d:
            continue
        try:
            s = int(dirn.replace("seed", ""))
        except ValueError:
            s = d.get("seed")
        out[s] = {"wer": d.get("wer"), "cer": d.get("cer")}
    return out

def msd(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)

def main():
    tr = transfer_by_seed()
    st = standard_by_seed()

    tr_wer = [tr[s]["wer"] for s in sorted(tr)]
    tr_cer = [tr[s]["cer"] for s in sorted(tr)]
    st_wer = [st[s]["wer"] for s in sorted(st)]

    tw_m, tw_s, tw_n = msd(tr_wer)
    tc_m, tc_s, tc_n = msd(tr_cer)
    sw_m, sw_s, sw_n = msd(st_wer)

    print("=== TRANSFER (Hindi FLEURS -> Garhwali, phase-2) ===")
    if tw_n == 0:
        print("  no transfer phase-2 results found yet.")
    else:
        print(f"  WER {tw_m:.2f} +/- {tw_s:.2f}  CER {tc_m:.2f} +/- {tc_s:.2f}  (n={tw_n} seeds)")
        print("  per-seed WER:", {s: round(tr[s]['wer'], 2) for s in sorted(tr)})
    print(f"=== STANDARD CTC (reference) ===")
    print(f"  WER {sw_m:.2f} +/- {sw_s:.2f}  (n={sw_n} seeds)" if sw_n else "  none found")

    result = {"transfer": {"wer_mean": tw_m, "wer_std": tw_s, "n": tw_n,
                           "cer_mean": tc_m, "cer_std": tc_s,
                           "per_seed": {str(s): tr[s] for s in sorted(tr)}},
              "standard": {"wer_mean": sw_m, "wer_std": sw_s, "n": sw_n}}

    # paired test only if we have matched seeds for both, n>=2
    common = sorted(set(tr) & set(st))
    if len(common) >= 2:
        a = np.array([st[s]["wer"] for s in common])   # standard
        b = np.array([tr[s]["wer"] for s in common])   # transfer
        diff = b - a                                   # negative => transfer better
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
        try:
            w_stat, w_p = stats.wilcoxon(b, a)
        except Exception:
            w_p = float("nan")
        t_stat, t_p = stats.ttest_rel(b, a)
        result["paired_transfer_vs_standard"] = {
            "n_common_seeds": len(common), "common_seeds": common,
            "mean_diff_wer": float(diff.mean()), "cohen_dz": dz,
            "wilcoxon_p": float(w_p), "ttest_p": float(t_p),
            "significant_at_0.05": bool(t_p < 0.05)}
        print(f"\n=== PAIRED: transfer vs standard (n={len(common)} matched seeds) ===")
        print(f"  mean_diff={diff.mean():+.3f} WER (neg = transfer better)  dz={dz:+.2f}")
        print(f"  wilcoxon p={w_p:.3f}  ttest p={t_p:.3f}  sig={t_p < 0.05}")
        if t_p >= 0.05:
            verdict = ("No significant difference between Hindi->Garhwali transfer and "
                       "Garhwali-only standard CTC. Consistent with prior work that the related "
                       "standard language adds little beyond dialect data.")
        elif diff.mean() < 0:
            verdict = ("Transfer significantly LOWER WER than standard CTC -- a real transfer benefit.")
        else:
            verdict = ("Transfer significantly HIGHER WER than standard CTC -- the Hindi stage hurts.")
        result["verdict"] = verdict
        print("\nVERDICT:", verdict)
    else:
        print(f"\n[paired test skipped: only {len(common)} matched seed(s); need >=2]")
        result["verdict"] = "insufficient matched seeds for paired test"

    os.makedirs(AGG_DIR, exist_ok=True)
    p = os.path.join(AGG_DIR, "transfer_results.json")
    json.dump(result, open(p, "w"), indent=2)
    print(f"\n-> {p}")

if __name__ == "__main__":
    main()