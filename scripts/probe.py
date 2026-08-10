#!/usr/bin/env python
"""Layer-wise linear probing of w2v-BERT 2.0 (memory-safe).

Freeze the encoder, extract hidden states at each probed layer, train a
linear CTC probe on the frozen representations. Compares fine-tuned vs base.

MEMORY-SAFE REWRITE: the previous version held all probed layers x all
utterances of hidden states in CPU RAM simultaneously, which OOM-killed the
job (6 layers x ~3.8k utts x (T x 1024) float32). This version processes ONE
layer at a time -- extract that layer's features (as float16), train+eval the
probe, free the features, move on. Peak RAM is ~1/6 of before, halved again by
float16. Output JSON format is unchanged.

Usage: python probe.py --which finetuned --seed 42 [--objective standard]
       python probe.py --which base --seed 42
       (optional) --layers 4 8 12 16 20 24  --epochs 5  --cache
"""
import os, sys, json, argparse, gc, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from garhwali_asr import config as C
from garhwali_asr.data.prepare import clean_text
from garhwali_asr.data.loaders import build_processor, make_datasets

PROBE_LAYERS = [4, 8, 12, 16, 20, 24]
BASE_CKPT = "facebook/w2v-bert-2.0"

def get_ckpt(which, seed, objective):
    if which == "base":
        return BASE_CKPT
    run = os.path.join(C.RUNS_DIR, "objective", objective, f"seed{seed}")
    cks = sorted([d for d in os.listdir(run) if d.startswith("checkpoint-")])
    assert cks, f"no checkpoint in {run}"
    return os.path.join(run, cks[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["finetuned", "base"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    # headline system is standard CTC under the multi-seed reframing
    ap.add_argument("--objective", default="standard")
    ap.add_argument("--layers", type=int, nargs="+", default=PROBE_LAYERS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = build_processor()
    _, train_ds, val_ds, test_ds, info = make_datasets(smoke=a.smoke)
    blank = proc.tokenizer.pad_token_id
    V = len(proc.tokenizer)

    from transformers import Wav2Vec2BertModel
    ckpt = get_ckpt(a.which, a.seed, a.objective)
    print(f"[probe] which={a.which} objective={a.objective} ckpt={ckpt} layers={a.layers}")
    encoder = Wav2Vec2BertModel.from_pretrained(ckpt).to(dev).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- materialise labels once (small: just token id arrays) ----
    def labels_of(ds):
        labs = []
        for s in ds:
            li = np.array(s["labels"]); li = li[li != -100]
            labs.append(torch.tensor(li, dtype=torch.long))
        return labs
    ytr = labels_of(train_ds)
    yte = labels_of(test_ds)

    # ---- extract ONE layer's hidden states for a split, as float16 on CPU ----
    def extract_layer(ds, layer):
        feats = []
        for s in ds:
            fe = torch.tensor(np.array(s["input_features"]), dtype=torch.float32).unsqueeze(0).to(dev)
            am = torch.tensor(np.array(s["attention_mask"])).unsqueeze(0).to(dev)
            with torch.no_grad():
                out = encoder(fe, attention_mask=am, output_hidden_states=True)
                h = out.hidden_states[layer][0].to(torch.float16).cpu()  # (T,1024) fp16
            feats.append(h)
            del out, fe, am
        return feats

    from jiwer import wer as _wer, cer as _cer
    results = {}
    for L in a.layers:
        print(f"[probe] layer {L}: extracting train...", flush=True)
        Htr = extract_layer(train_ds, L)
        print(f"[probe] layer {L}: extracting test...", flush=True)
        Hte = extract_layer(test_ds, L)

        probe = nn.Linear(1024, V).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
        for ep in range(a.epochs):
            probe.train(); idx = np.random.permutation(len(Htr))
            for i in idx:
                h = Htr[i].to(dev).float()           # fp16 -> fp32 for the linear
                y = ytr[i].to(dev)
                logp = probe(h).log_softmax(-1).unsqueeze(1)  # (T,1,V)
                loss = ctc(logp, y.unsqueeze(0),
                           torch.tensor([h.shape[0]]), torch.tensor([len(y)]))
                opt.zero_grad(); loss.backward(); opt.step()

        probe.eval(); hyps, refs = [], []
        for i in range(len(Hte)):
            h = Hte[i].to(dev).float()
            with torch.no_grad():
                ids = probe(h).argmax(-1).cpu().numpy()
            out, prev = [], None
            for t in ids:
                if t != blank and t != prev: out.append(int(t))
                prev = t
            hyps.append(clean_text(proc.tokenizer.decode(out).replace("|", " ")))
            refs.append(clean_text(proc.tokenizer.decode(yte[i].numpy(), group_tokens=False).replace("|", " ")))
        pr = [(h, r) for h, r in zip(hyps, refs) if r.strip()]; P, R = zip(*pr)
        w, c = _wer(list(R), list(P))*100, _cer(list(R), list(P))*100
        results[L] = {"wer": w, "cer": c}
        print(f"  layer {L}: WER {w:.2f} CER {c:.2f}", flush=True)

        # free this layer's features before the next one
        del Htr, Hte, probe, opt
        gc.collect()
        if dev == "cuda": torch.cuda.empty_cache()

    out_dir = os.path.join(C.RESULTS_ROOT, "probing")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"probe_{a.which}_seed{a.seed}.json")
    json.dump({"which": a.which, "seed": a.seed, "objective": a.objective,
               "layers": results}, open(outp, "w"), indent=2)
    print(f"[done] -> {outp}")

if __name__ == "__main__":
    main()
