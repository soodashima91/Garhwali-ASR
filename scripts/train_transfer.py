#!/usr/bin/env python
"""Cross-lingual transfer (single-seed). Phase-1: w2v-BERT on FLEURS Hindi.
Phase-2: load Phase-1 encoder (fresh Garhwali head via vocab mismatch), train aug Garhwali.
Usage: python train_transfer.py --phase 1 --seed 42
       python train_transfer.py --phase 2 --seed 42"""
import os, sys, json, argparse, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from garhwali_asr import config as C
from garhwali_asr.data.prepare import clean_text, decode_audio
from garhwali_asr.data.loaders import make_datasets
from garhwali_asr.models import w2vbert as M

P1_DIR = lambda s: os.path.join(C.RUNS_DIR, "transfer", f"phase1_seed{s}")
P2_DIR = lambda s: os.path.join(C.RUNS_DIR, "transfer", f"phase2_seed{s}")

def fleurs_hindi(proc, smoke=False):
    from datasets import load_dataset, Audio, Dataset
    ds = load_dataset("google/fleurs", "hi_in", split="train+validation", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    if smoke: ds = ds.select(range(40))
    fe = proc.feature_extractor; feats, masks, labs = [], [], []
    for ex in ds:
        text = clean_text(ex.get("transcription") or ex.get("raw_transcription") or "")
        if not text.strip(): continue
        sp, sr = decode_audio(ex["audio"]); sp = np.asarray(sp, dtype=np.float32)
        ids = proc.tokenizer(text).input_ids
        if not ids: continue
        ff = np.asarray(fe(sp, sampling_rate=C.SAMPLE_RATE).input_features[0], dtype=np.float32)
        if ff.size == 0: continue
        am = np.ones(ff.shape[0], dtype=np.int64)
        feats.append(ff); masks.append(am); labs.append(ids)
    return Dataset.from_dict({"input_features": feats, "attention_mask": masks, "labels": labs})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1,2], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    # see note in scripts/train_one.py -- predictions.csv holds reference text,
    # for local analysis only, not for redistribution. Off by default.
    ap.add_argument("--dump-predictions", action="store_true", dest="dump_predictions")
    a = ap.parse_args()

    if a.phase == 1:
        out = P1_DIR(a.seed); os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out,"DONE")): print("[p1] DONE"); return
        proc, _, _, _, _ = make_datasets(smoke=True, factors=[1.0])  # just need proc; tiny
        print("[p1] loading FLEURS Hindi...")
        train_ds = fleurs_hindi(proc, smoke=a.smoke)
        # small val split off the Hindi train
        n = len(train_ds); val_ds = train_ds.select(range(max(1,int(0.95*n)), n))
        train_ds = train_ds.select(range(0, max(1,int(0.95*n))))
        print(f"[p1] Hindi train {len(train_ds)} val {len(val_ds)}")
        epochs = 1 if a.smoke else 15
        model = M.load_model(proc)
        cm = M.make_metric(proc, clean_text)
        trainer = M.build_trainer(model, proc, train_ds, val_ds, cm,
                                  objective="standard", out_dir=out, seed=a.seed,
                                  num_epochs=epochs)
        trainer.train()
        trainer.save_model(out)
        open(os.path.join(out,"DONE"),"w").write("ok\n")
        print(f"[p1 done] -> {out}")
    else:
        p1 = P1_DIR(a.seed)
        assert os.path.exists(os.path.join(p1,"DONE")), f"run --phase 1 first ({p1})"
        out = P2_DIR(a.seed); os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out,"DONE")): print("[p2] DONE"); return
        proc, train_ds, val_ds, test_ds, info = make_datasets(smoke=a.smoke, factors=C.SPEED_FACTORS)
        print(f"[p2] init encoder from Phase-1 {p1}")
        model = M.load_model(proc, init_path=p1)  # <-- the transfer
        num_epochs = 1 if a.smoke else C.NUM_EPOCHS
        cm = M.make_metric(proc, clean_text)
        trainer = M.build_trainer(model, proc, train_ds, val_ds, cm,
                                  objective="standard", out_dir=out, seed=a.seed,
                                  num_epochs=num_epochs)
        trainer.train()
        wer, cer, refs, hyps = M.greedy_test(trainer.model, proc, test_ds, clean_text)
        if a.dump_predictions:
            with open(os.path.join(out,"predictions.csv"),"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["idx","reference","hypothesis"])
                for i,(r,h) in enumerate(zip(refs,hyps)): w.writerow([i,r,h])
        json.dump({"system":"transfer","seed":a.seed,"wer":wer,"cer":cer,
                   "data_info":info,"best_val_wer":trainer.state.best_metric},
                  open(os.path.join(out,"result.json"),"w"), indent=2)
        open(os.path.join(out,"DONE"),"w").write("ok\n")
        print(f"[p2 done] transfer/seed{a.seed}: WER {wer:.2f} CER {cer:.2f}")

if __name__ == "__main__":
    main()
