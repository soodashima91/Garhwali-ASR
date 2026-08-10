"""garhwali_asr.analyze.stats - bootstrap CIs, Holm-corrected paired tests, per-class significance."""
import os, csv, glob, json
import numpy as np
from jiwer import wer as _wer

def load_predictions(pred_csv):
    refs, hyps = [], []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            refs.append(str(r["reference"])); hyps.append(str(r["hypothesis"]))
    return refs, hyps

def per_utt_wer(refs, hyps):
    out = []
    for r, h in zip(refs, hyps):
        if not r.strip():
            continue
        out.append(_wer([r], [h]) * 100.0)
    return np.array(out)

def bootstrap_ci(per_utt_values, n_boot=2000, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    n = len(per_utt_values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = per_utt_values[idx].mean()
    lo, hi = np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])
    return float(per_utt_values.mean()), float(lo), float(hi)

def _wilcoxon_paired(a, b):
    from scipy import stats
    assert len(a) == len(b), "paired arrays must align (same utterances)"
    diff = b - a
    if np.allclose(diff, 0):
        return {"W": 0.0, "p": 1.0, "mean_diff": 0.0, "cohen_dz": 0.0, "boot_ci": [0.0, 0.0]}
    try:
        W, p = stats.wilcoxon(b, a)
    except ValueError:
        W, p = float("nan"), 1.0
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boot, [2.5, 97.5])
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return {"W": float(W), "p": float(p), "mean_diff": float(diff.mean()),
            "cohen_dz": float(dz), "boot_ci": [float(ci[0]), float(ci[1])]}

def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()

def paired_tests(system_preds):
    names = list(system_preds.keys())
    puw = {n: per_utt_wer(*system_preds[n]) for n in names}
    lens = {len(v) for v in puw.values()}
    if len(lens) != 1:
        raise ValueError(f"systems have differing #utterances {lens}; ensure identical test split and order.")
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            res = _wilcoxon_paired(puw[a], puw[b])
            res["pair"] = f"{a}_vs_{b}"; res["raw_p"] = res["p"]
            pairs.append(res)
    adj = holm_bonferroni([pr["raw_p"] for pr in pairs])
    for pr, ap in zip(pairs, adj):
        pr["holm_p"] = ap
        pr["significant_0.05"] = ap < 0.05
    return pairs

def _per_utt_class_error(refs, hyps, target_class):
    from .class_errors import _align
    from ..phonology import classes_of
    rates = []
    for r, h in zip(refs, hyps):
        ref_n = err_n = 0
        for op, rc, hc in _align(r, h):
            if rc is not None and target_class in classes_of(rc):
                ref_n += 1
                if op != "match":
                    err_n += 1
        rates.append((err_n / ref_n) if ref_n > 0 else np.nan)
    return np.array(rates)

def per_class_significance(preds_a, preds_b, target_class="matra"):
    from scipy import stats
    ra = _per_utt_class_error(*preds_a, target_class)
    rb = _per_utt_class_error(*preds_b, target_class)
    mask = ~(np.isnan(ra) | np.isnan(rb))
    a, b = ra[mask], rb[mask]
    if len(a) < 3:
        return {"class": target_class, "n_paired": int(mask.sum()),
                "note": "too few utterances with this class for a test"}
    diff = b - a
    try:
        W, p = stats.wilcoxon(b, a) if not np.allclose(diff, 0) else (0.0, 1.0)
    except ValueError:
        W, p = float("nan"), 1.0
    return {"class": target_class, "n_paired": int(mask.sum()),
            "mean_err_a": float(a.mean()), "mean_err_b": float(b.mean()),
            "mean_reduction_a_minus_b": float((a - b).mean()),
            "W": float(W), "p": float(p)}

def find_predictions(runs_dir, objective, exp="objective", seed=None):
    base = os.path.join(runs_dir, exp, objective)
    if seed is not None:
        p = os.path.join(base, f"seed{seed}", "predictions.csv")
        return [p] if os.path.exists(p) else []
    return sorted(glob.glob(os.path.join(base, "seed*", "predictions.csv")))

def paired_tests_multiseed(system_seed_preds):
    """
    Multi-seed significance, replacing the seed-42-only path.

    system_seed_preds: {objective: [(refs, hyps), ...one per seed...]}
      every (refs,hyps) must be the SAME test utterances in the SAME order
      (verified: all official-split runs share identical references).

    Returns dict with:
      - 'per_seed_corpus_wer': {obj: [wer per seed]}          (report this table)
      - 'primary_seed_level' : pairwise tests on the n_seed corpus WERs
                               (Wilcoxon + paired t, Holm-corrected). DEFENSIBLE
                               headline test: the seed is the randomised unit.
                               Low power at n=5 by design.
      - 'secondary_pooled_per_utt' : pairwise Wilcoxon on per-utterance WER
                               pooled across seeds. High power but treats
                               correlated (same-utterance, different-seed)
                               points as independent, so it OVERSTATES
                               significance. Directional cross-check only.
    """
    import numpy as np
    from scipy import stats

    names = list(system_seed_preds.keys())

    # alignment guard: all runs must share identical test references
    ref0 = system_seed_preds[names[0]][0][0]
    for nm in names:
        for refs, _ in system_seed_preds[nm]:
            if refs != ref0:
                raise ValueError("seed runs are not on identical aligned test refs")

    def corpus_wer(refs, hyps):
        pr = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        R, H = zip(*pr)
        return _wer(list(R), list(H)) * 100.0

    corp = {nm: np.array([corpus_wer(r, h) for (r, h) in system_seed_preds[nm]])
            for nm in names}

    # ---- primary: seed-level paired tests (n = n_seeds) ----
    prim = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            da, db = corp[a], corp[b]
            diff = db - da
            if np.allclose(diff, 0):
                wp = 1.0
            else:
                try:
                    _, wp = stats.wilcoxon(db, da)
                except ValueError:
                    wp = 1.0
            _, tp = stats.ttest_rel(db, da)
            prim.append({"pair": f"{a}_vs_{b}", "n_seeds": len(da),
                         "mean_diff": float(diff.mean()),
                         "wilcoxon_p": float(wp), "ttest_p": float(tp)})
    for key, adjkey in [("wilcoxon_p", "wilcoxon_holm"), ("ttest_p", "ttest_holm")]:
        adj = holm_bonferroni([p[key] for p in prim])
        for p, ap in zip(prim, adj):
            p[adjkey] = ap
            p[adjkey.replace("holm", "sig")] = ap < 0.05

    # ---- secondary: pooled per-utterance (caveated) ----
    puw = {nm: np.concatenate([per_utt_wer(r, h) for (r, h) in system_seed_preds[nm]])
           for nm in names}
    sec = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = _wilcoxon_paired(puw[a], puw[b])
            res["pair"] = f"{a}_vs_{b}"; res["raw_p"] = res["p"]
            sec.append(res)
    adj = holm_bonferroni([r["raw_p"] for r in sec])
    for r, ap in zip(sec, adj):
        r["holm_p"] = ap; r["significant_0.05"] = ap < 0.05

    return {
        "per_seed_corpus_wer": {nm: corp[nm].tolist() for nm in names},
        "primary_seed_level": prim,
        "secondary_pooled_per_utt": sec,
        "secondary_note": ("pooled test treats same-utterance/different-seed "
                           "points as independent; overstates significance, "
                           "use as directional cross-check only"),
    }
