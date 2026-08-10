#!/usr/bin/env python
"""train_one.py — train a single (objective, seed) run and persist results."""
import os, sys, json, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garhwali_asr import config as C
from garhwali_asr.data.prepare import clean_text
from garhwali_asr.data.loaders import make_datasets
from garhwali_asr.models import w2vbert as M
from garhwali_asr.analyze.class_errors import class_error_rates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["standard", "focal", "matra"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--exp", default="objective")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-aug", action="store_true", dest="no_aug")
    # predictions.csv contains VAANI reference transcripts. It is required only
    # to re-run analyze/stats.py locally; it is NOT part of the public release
    # (do not redistribute it -- it reconstructs corpus text). Off by default.
    ap.add_argument("--dump-predictions", action="store_true", dest="dump_predictions",
                    help="write predictions.csv with reference+hypothesis text "
                         "(local analysis only; do NOT redistribute)")
    a = ap.parse_args()

    num_epochs = 1 if a.smoke else C.NUM_EPOCHS
    factors = [1.0] if a.no_aug else None
    if a.no_aug and a.exp == "objective":
        a.exp = "noaug"

    parts = [a.objective, f"seed{a.seed}"]
    if a.objective == "focal" and a.gamma is not None: parts.append(f"g{a.gamma}")
    if a.objective == "matra" and a.lam is not None:    parts.append(f"l{a.lam}")
    tag = "_".join(parts)
    suffix = "" if len(parts) == 2 else "_" + "_".join(parts[2:])
    runs_base = C.RUNS_DIR + ("_smoke" if a.smoke else "")
    out_dir = os.path.join(runs_base, a.exp, a.objective, f"seed{a.seed}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "DONE")
    if os.path.exists(done):
        print(f"[skip] {tag}: DONE exists at {out_dir}"); return

    print(f"[run] objective={a.objective} seed={a.seed} epochs={num_epochs} "
          f"gamma={a.gamma} lam={a.lam} smoke={a.smoke}")
    print(f"[run] out_dir={out_dir}")

    proc, train_ds, val_ds, test_ds, info = make_datasets(smoke=a.smoke, factors=factors)
    print(f"[data] {info}")
    if not info["aug_ok"]:
        print(f"[warn] aug train {info['n_train_aug']} != expected {info['expected_aug']}")

    from transformers import set_seed
    set_seed(a.seed)
    import torch
    model = M.load_model(proc)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    compute_metrics = M.make_metric(proc, clean_text)
    trainer = M.build_trainer(model, proc, train_ds, val_ds, compute_metrics,
                              objective=a.objective, out_dir=out_dir, seed=a.seed,
                              num_epochs=num_epochs, gamma=a.gamma, lam=a.lam,
                              temperature=a.temperature)
    trainer.train()

    with open(os.path.join(out_dir, "curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","step","train_loss","eval_loss","eval_wer","eval_cer"])
        w.writeheader()
        for e in trainer.state.log_history:
            w.writerow({"epoch": e.get("epoch"), "step": e.get("step"),
                        "train_loss": e.get("loss"), "eval_loss": e.get("eval_loss"),
                        "eval_wer": e.get("eval_wer"), "eval_cer": e.get("eval_cer")})

    wer, cer, refs, hyps = M.greedy_test(trainer.model, proc, test_ds, clean_text)

    if a.dump_predictions:
        with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["idx","reference","hypothesis"])
            for i, (r, h) in enumerate(zip(refs, hyps)): w.writerow([i, r, h])

    cls = class_error_rates(refs, hyps)
    with open(os.path.join(out_dir, "class_errors.json"), "w") as f:
        json.dump(cls, f, indent=2, ensure_ascii=False)

    result = {"objective": a.objective, "seed": a.seed, "tag": tag,
              "gamma": a.gamma, "lam": a.lam, "epochs": num_epochs, "smoke": a.smoke,
              "wer": wer, "cer": cer, "best_val_wer": trainer.state.best_metric,
              "data_info": info,
              "matra_error_rate": cls.get("matra", {}).get("error_rate"),
              "aspirated_error_rate": cls.get("aspirated", {}).get("error_rate"),
              "retroflex_error_rate": cls.get("retroflex", {}).get("error_rate")}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    open(done, "w").write("ok\n")
    print(f"[done] {tag}: WER {wer:.2f} CER {cer:.2f} "
          f"matraER {result['matra_error_rate']} -> {out_dir}")

if __name__ == "__main__":
    main()
