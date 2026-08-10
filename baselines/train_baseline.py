#!/usr/bin/env python
"""Train a wav2vec2-family / MMS baseline on official-split Garhwali, one (model, seed).
Writes result.json/predictions.csv/DONE to results/runs/baselines/<model>/seed<N>/."""
import os, sys, json, argparse, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from typing import Any
from garhwali_asr import config as C
from garhwali_asr.data.prepare import clean_text
from baselines.baseline_data import MODELS, make_baseline_datasets

@dataclass
class Collator:
    processor: Any
    fe: Any
    tok: Any
    def __call__(self, features):
        inp = [{"input_values": f["input_values"]} for f in features]
        lab = [{"input_ids": f["labels"]} for f in features]
        batch = self.fe.pad(inp, padding=True, return_tensors="pt")
        batch["input_values"] = batch["input_values"].float()  # double -> float32 for BF16 model
        lb = self.tok.pad(lab, padding=True, return_tensors="pt")
        batch["labels"] = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
        return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--aug", action="store_true", help="apply 3x speed aug")
    ap.add_argument("--smoke", action="store_true")
    # see note in scripts/train_one.py: predictions.csv holds reference text,
    # local analysis only, not for redistribution. Off by default.
    ap.add_argument("--dump-predictions", action="store_true", dest="dump_predictions")
    a = ap.parse_args()

    sub = f"seed{a.seed}" + ("_smoke" if a.smoke else "")
    out_dir = os.path.join(C.RUNS_DIR, "baselines", a.model, sub)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print(f"[skip] {a.model}/seed{a.seed} DONE"); return

    factors = C.SPEED_FACTORS if a.aug else None
    epochs = 1 if a.smoke else C.NUM_EPOCHS
    print(f"[run] model={a.model} seed={a.seed} aug={a.aug} epochs={epochs}")

    tok, fe, train_ds, val_ds, test_ds, info = make_baseline_datasets(factors=factors, smoke=a.smoke)
    print(f"[data] {info}")

    from transformers import (AutoModelForCTC, set_seed, TrainingArguments, Trainer,
                              EarlyStoppingCallback)
    set_seed(a.seed)
    ckpt = MODELS[a.model]
    load_kw = dict(ctc_loss_reduction="mean", pad_token_id=tok.pad_token_id,
                   vocab_size=len(tok), ignore_mismatched_sizes=True)
    # MMS: ignore its built-in adapter head, train a fresh CTC head on our vocab
    model = AutoModelForCTC.from_pretrained(ckpt, **load_kw)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    from jiwer import wer as _wer, cer as _cer
    def compute_metrics(pred):
        pid = np.argmax(pred.predictions, axis=-1)
        labs = pred.label_ids.copy(); labs[labs == -100] = tok.pad_token_id
        ps = tok.batch_decode(pid); ls = tok.batch_decode(labs, group_tokens=False)
        pn = [clean_text(p.replace("|", " ")) for p in ps]
        ln = [clean_text(l.replace("|", " ")) for l in ls]
        pr = [(p, l) for p, l in zip(pn, ln) if l.strip()]
        if not pr: return {"wer": 1.0, "cer": 1.0}
        P, L = zip(*pr); return {"wer": _wer(list(L), list(P)), "cer": _cer(list(L), list(P))}

    # dual learning rate: high LR on fresh head, low on pretrained encoder
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    head_params, enc_params = [], []
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (head_params if "lm_head" in n else enc_params).append(prm)
    optimizer = AdamW([
        {"params": head_params, "lr": 1e-3},
        {"params": enc_params,  "lr": 3e-5},
    ])
    steps_per_epoch = max(1, len(train_ds) // 32)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    args = TrainingArguments(
        output_dir=out_dir, seed=a.seed,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, num_train_epochs=epochs,
        warmup_ratio=C.WARMUP_RATIO,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=25,
        load_best_model_at_end=True, metric_for_best_model="wer",
        greater_is_better=False, save_total_limit=6, report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(None, fe, tok), compute_metrics=compute_metrics,
                      optimizers=(optimizer, scheduler),
                      callbacks=[EarlyStoppingCallback(C.EARLY_STOP_PATIENCE)])
    trainer.train()

    # greedy test
    model.eval(); dev = next(model.parameters()).device; refs, hyps = [], []
    for s in test_ds:
        iv = torch.tensor(np.array(s["input_values"]), dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            lg = model(iv).logits
        h = clean_text(tok.batch_decode(torch.argmax(lg, -1))[0].replace("|", " "))
        li = torch.tensor(s["labels"]); 
        r = clean_text(tok.decode(li, group_tokens=False).replace("|", " "))
        refs.append(r); hyps.append(h)
    pr = [(p, l) for p, l in zip(hyps, refs) if l.strip()]; P, L = zip(*pr)
    test_wer = _wer(list(L), list(P)) * 100; test_cer = _cer(list(L), list(P)) * 100

    if a.dump_predictions:
        with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["idx", "reference", "hypothesis"])
            for i, (r, h) in enumerate(zip(refs, hyps)): w.writerow([i, r, h])
    json.dump({"model": a.model, "seed": a.seed, "aug": a.aug, "epochs": epochs,
               "wer": test_wer, "cer": test_cer, "data_info": info,
               "best_val_wer": trainer.state.best_metric},
              open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    open(os.path.join(out_dir, "DONE"), "w").write("ok\n")
    print(f"[done] {a.model}/seed{a.seed}: WER {test_wer:.2f} CER {test_cer:.2f}")

if __name__ == "__main__":
    main()
