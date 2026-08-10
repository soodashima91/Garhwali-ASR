#!/usr/bin/env python
"""Aggregate all runs/<exp>/<obj>/seed*/result.json into tables + significance.
Safe to run on partial results.
    python aggregate.py
Outputs to results/aggregate/: main_results.csv/.tex, significance.json
"""
import os, sys, json, glob, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from garhwali_asr import config as C
from garhwali_asr.analyze import stats as S

OBJ_ORDER = ["standard", "focal", "matra"]
OBJ_LABEL = {"standard": "Standard CTC", "focal": "Focal CTC",
             "matra": "Matra-weighted CTC (ours)"}

def load_results(exp):
    out = {}
    for obj in OBJ_ORDER:
        rows = []
        for rj in sorted(glob.glob(os.path.join(C.RUNS_DIR, exp, obj, "seed*", "result.json"))):
            d = os.path.basename(os.path.dirname(rj))
            if "_" in d.replace("seed", "", 1):
                continue
            try:
                rows.append(json.load(open(rj)))
            except Exception as e:
                print(f"  [warn] could not read {rj}: {e}")
        out[obj] = rows
    return out

def msd(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0: return (None, None, 0)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="objective")
    a = ap.parse_args()
    os.makedirs(C.AGG_DIR, exist_ok=True)
    res = load_results(a.exp)

    table = []
    for obj in OBJ_ORDER:
        rows = res[obj]
        if not rows:
            print(f"[skip] {obj}: no finished seeds yet"); continue
        wer_m, wer_s, n = msd([r["wer"] for r in rows])
        cer_m, cer_s, _ = msd([r["cer"] for r in rows])
        mat_m, mat_s, _ = msd([r.get("matra_error_rate") for r in rows])
        asp_m, _, _ = msd([r.get("aspirated_error_rate") for r in rows])
        ret_m, _, _ = msd([r.get("retroflex_error_rate") for r in rows])
        table.append({"objective": obj, "label": OBJ_LABEL[obj], "n_seeds": n,
                      "wer_mean": wer_m, "wer_std": wer_s,
                      "cer_mean": cer_m, "cer_std": cer_s,
                      "matra_er_mean": (mat_m*100 if mat_m is not None else None),
                      "asp_er_mean": (asp_m*100 if asp_m is not None else None),
                      "ret_er_mean": (ret_m*100 if ret_m is not None else None)})

    print("\n=== MAIN RESULTS (mean +/- std over seeds) ===")
    print(f"{'system':28s} {'n':>2} {'WER':>14} {'CER':>14} {'matraER%':>9}")
    for t in table:
        print(f"{t['label']:28s} {t['n_seeds']:>2} "
              f"{t['wer_mean']:>6.2f} +/- {t['wer_std']:4.2f}  "
              f"{t['cer_mean']:>6.2f} +/- {t['cer_std']:4.2f}  "
              f"{(t['matra_er_mean'] or 0):>7.2f}")

    if table:
        with open(os.path.join(C.AGG_DIR, "main_results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader(); [w.writerow(t) for t in table]
        with open(os.path.join(C.AGG_DIR, "main_results.tex"), "w") as f:
            f.write("\\begin{tabular}{lrcc}\n\\toprule\n")
            f.write("System & Seeds & WER (\\%) & CER (\\%) \\\\\n\\midrule\n")
            for t in table:
                f.write(f"{t['label']} & {t['n_seeds']} & "
                        f"{t['wer_mean']:.2f} $\\pm$ {t['wer_std']:.2f} & "
                        f"{t['cer_mean']:.2f} $\\pm$ {t['cer_std']:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- significance ----
    # gather ALL seeds' predictions per objective (not just seed42)
    seed_preds = {}
    for obj in OBJ_ORDER:
        runs = sorted(glob.glob(os.path.join(C.RUNS_DIR, a.exp, obj, "seed*", "predictions.csv")))
        runs = [p for p in runs
                if "_" not in os.path.basename(os.path.dirname(p)).replace("seed", "", 1)]
        if runs:
            seed_preds[obj] = [S.load_predictions(p) for p in runs]

    sig = {}
    n_each = [len(v) for v in seed_preds.values()]
    if len(seed_preds) >= 2 and all(n > 0 for n in n_each):
        try:
            sig["multiseed"] = S.paired_tests_multiseed(seed_preds)
            ms = sig["multiseed"]
            n_seeds = len(next(iter(seed_preds.values())))
            print(f"\n=== MULTI-SEED SIGNIFICANCE (primary: seed-level, n={n_seeds}) ===")
            for pr in ms["primary_seed_level"]:
                print(f"  {pr['pair']:20s} mean_diff={pr['mean_diff']:+.3f}  "
                      f"wilcoxon_holm={pr['wilcoxon_holm']:.4f}  "
                      f"ttest_holm={pr['ttest_holm']:.4f}  sig={pr['wilcoxon_sig']}")
            print("  [secondary pooled per-utt -- overstates sig, cross-check only]")
            for pr in ms["secondary_pooled_per_utt"]:
                print(f"    {pr['pair']:20s} holm_p={pr['holm_p']:.4f} sig={pr['significant_0.05']}")
        except Exception as e:
            sig["multiseed_error"] = str(e)
            print(f"\n[multiseed] error: {e}")
    else:
        print("\n[significance] need >=2 objectives with >=1 seed; have",
              {k: len(v) for k, v in seed_preds.items()})

    # seed-42 per-class matra test + bootstrap (explicitly seed42, NOT lst[0])
    def _seed42(obj):
        p = os.path.join(C.RUNS_DIR, a.exp, obj, "seed42", "predictions.csv")
        return S.load_predictions(p) if os.path.exists(p) else None
    s42 = {obj: _seed42(obj) for obj in OBJ_ORDER if _seed42(obj) is not None}
    if "standard" in s42 and "matra" in s42:
        sig["matra_class_std_vs_matra_seed42"] = S.per_class_significance(
            s42["standard"], s42["matra"], "matra")
        mc = sig["matra_class_std_vs_matra_seed42"]
        print(f"  matra-class std vs matra (seed42): err {mc.get('mean_err_a')} -> "
              f"{mc.get('mean_err_b')} p={mc.get('p')}")
    sig["bootstrap_ci_seed42"] = {}
    for obj, (refs, hyps) in s42.items():
        m, lo, hi = S.bootstrap_ci(S.per_utt_wer(refs, hyps))
        sig["bootstrap_ci_seed42"][obj] = {"wer_mean": m, "ci95": [lo, hi]}

    with open(os.path.join(C.AGG_DIR, "significance.json"), "w") as f:
        json.dump(sig, f, indent=2)
    print(f"\nSaved to {C.AGG_DIR}")

if __name__ == "__main__":
    main()
