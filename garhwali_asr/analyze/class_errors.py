"""
garhwali_asr.analyze.class_errors
==================================
Per-phonological-class error rates. This is the targeted metric that makes the
matra-weighted objective evaluable EVEN IF aggregate WER moves within seed noise:
if matra-weighted CTC reduces matra-class errors specifically, that is a clean,
mechanistically-honest result regardless of overall WER.

Method: character-level alignment (Levenshtein backtrace) between reference and
hypothesis; for each reference character we record whether it was correctly
matched. The per-class error rate is then:

    class_error_rate(C) = (# reference chars in class C not correctly matched)
                          / (# reference chars in class C)

We report this for each class in garhwali_asr.phonology.CLASS_SETS, plus an
overall CER for context. Insertions of class-C characters are reported
separately (they have no reference character to attribute to).
"""
from ..phonology import classes_of


def _align(ref, hyp):
    """
    Levenshtein alignment over characters. Returns a list of ops:
    ('match',r,h) / ('sub',r,h) / ('del',r,None) / ('ins',None,h).
    Standard DP with backtrace.
    """
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1] and dp[i][j] == dp[i-1][j-1]:
            ops.append(("match", ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1:
            ops.append(("sub", ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1:
            ops.append(("del", ref[i-1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j-1])); j -= 1
    ops.reverse()
    return ops


def class_error_rates(refs, hyps, classes=None):
    """
    refs, hyps: lists of strings (already normalised the same way as training).
    Returns dict: class -> {'ref_count','errors','error_rate','insertions'}.
    A reference char is an 'error' if it was deleted or substituted (not matched).
    """
    from ..phonology import CLASS_SETS
    classes = classes or list(CLASS_SETS.keys())
    ref_count = {c: 0 for c in classes}
    err_count = {c: 0 for c in classes}
    ins_count = {c: 0 for c in classes}
    overall_ref = 0; overall_err = 0

    for ref, hyp in zip(refs, hyps):
        for op, r, h in _align(ref, hyp):
            if r is not None and not r.isspace():
                overall_ref += 1
                rc = classes_of(r)
                is_err = (op != "match")
                if is_err: overall_err += 1
                for c in rc:
                    if c in ref_count:
                        ref_count[c] += 1
                        if is_err: err_count[c] += 1
            if op == "ins" and h is not None and not h.isspace():
                for c in classes_of(h):
                    if c in ins_count: ins_count[c] += 1

    out = {}
    for c in classes:
        rc = ref_count[c]
        out[c] = {"ref_count": rc, "errors": err_count[c],
                  "error_rate": (err_count[c]/rc if rc else None),
                  "insertions": ins_count[c]}
    out["_overall"] = {"ref_count": overall_ref, "errors": overall_err,
                       "error_rate": (overall_err/overall_ref if overall_ref else None),
                       "insertions": None}
    return out
