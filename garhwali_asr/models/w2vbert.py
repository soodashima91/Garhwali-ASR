"""garhwali_asr.models.w2vbert — model setup + trainer routing loss via combined_loss."""
import math, numpy as np, torch
from dataclasses import dataclass
from typing import Any
from .. import config as C
from ..losses import combined_loss
from ..phonology import build_token_weights

@dataclass
class Collator:
    processor: Any
    def __call__(self, features):
        inp = [{"input_features": f["input_features"], "attention_mask": f["attention_mask"]}
               for f in features]
        lab = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(inp, padding=True, return_tensors="pt")
        lb = self.processor.tokenizer.pad(lab, padding=True, return_tensors="pt")
        batch["labels"] = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
        return batch

def make_metric(processor, clean_text):
    from jiwer import wer as _wer, cer as _cer
    def compute_metrics(pred):
        pid = np.argmax(pred.predictions, axis=-1)
        labs = pred.label_ids.copy()
        labs[labs == -100] = processor.tokenizer.pad_token_id
        ps = processor.tokenizer.batch_decode(pid)
        ls = processor.tokenizer.batch_decode(labs, group_tokens=False)
        pn = [clean_text(p.replace("|", " ")) for p in ps]
        ln = [clean_text(l.replace("|", " ")) for l in ls]
        pr = [(p, l) for p, l in zip(pn, ln) if l.strip()]
        if not pr:
            return {"wer": 1.0, "cer": 1.0}
        P, L = zip(*pr)
        return {"wer": _wer(list(L), list(P)), "cer": _cer(list(L), list(P))}
    return compute_metrics

def load_model(processor, init_path=None):
    from transformers import Wav2Vec2BertForCTC
    model = Wav2Vec2BertForCTC.from_pretrained(
        init_path or C.BASE_MODEL,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        mask_time_prob=0.0, layerdrop=0.0,
        ctc_loss_reduction="none",
        add_adapter=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    for p in model.wav2vec2_bert.feature_projection.parameters():
        p.requires_grad = False
    def hook(mod, inp, out):
        ts = out if isinstance(out, tuple) else (out,)
        for t in ts:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad_(True)
    model.wav2vec2_bert.feature_projection.register_forward_hook(hook)
    model.gradient_checkpointing_enable()
    return model

def build_trainer(model, processor, train_ds, val_ds, compute_metrics,
                  objective, out_dir, seed, num_epochs,
                  gamma=None, lam=None, temperature=None):
    from transformers import (TrainingArguments, Trainer, EarlyStoppingCallback,
                              get_linear_schedule_with_warmup)
    from torch.optim import AdamW

    gamma = C.FOCAL_GAMMA if gamma is None else gamma
    lam = C.MATRA_LAMBDA if lam is None else lam
    temperature = C.FOCAL_TEMPERATURE if temperature is None else temperature

    vocab = processor.tokenizer.get_vocab()
    token_weights = build_token_weights(vocab, C.CLASS_WEIGHTS)
    blank = processor.tokenizer.pad_token_id

    class RoutedTrainer(Trainer):
        _logged = False
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs["labels"]
            out = model(**inputs)
            logits = out.logits
            attn = inputs.get("attention_mask")
            if attn is not None:
                il = model._get_feat_extract_output_lengths(attn.sum(-1)).to(torch.long)
            else:
                il = torch.full((logits.size(0),), logits.size(1),
                                dtype=torch.long, device=logits.device)
            tl = (labels >= 0).sum(-1)
            loss, info = combined_loss(
                logits, labels, il, tl, token_weights, blank,
                objective=objective, alpha=C.FOCAL_ALPHA, gamma=gamma,
                temperature=temperature, lam=lam,
                focal_on_base=C.MATRA_FOCAL_ON_BASE)
            if not RoutedTrainer._logged and info:
                print(f"[loss check] objective={objective} {info}")
                RoutedTrainer._logged = True
            return (loss, out) if return_outputs else loss

    spe = math.ceil(len(train_ds) / (C.PER_DEVICE_BATCH * C.GRAD_ACCUM))
    total = spe * num_epochs
    enc = [p for n, p in model.named_parameters() if "lm_head" not in n and p.requires_grad]
    hd  = [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad]
    opt = AdamW([{"params": enc, "lr": C.ENCODER_LR},
                 {"params": hd, "lr": C.HEAD_LR}])
    sch = get_linear_schedule_with_warmup(opt, int(C.WARMUP_RATIO * total), total)

    args = TrainingArguments(
        output_dir=out_dir, seed=seed, data_seed=seed,
        per_device_train_batch_size=C.PER_DEVICE_BATCH,
        per_device_eval_batch_size=C.PER_DEVICE_BATCH,
        gradient_accumulation_steps=C.GRAD_ACCUM,
        num_train_epochs=num_epochs, warmup_ratio=C.WARMUP_RATIO,
        bf16=True, tf32=True, gradient_checkpointing=True,
        dataloader_num_workers=2, group_by_length=False,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=25,
        load_best_model_at_end=True, metric_for_best_model="wer",
        greater_is_better=False, save_total_limit=6, report_to="none",
    )
    return RoutedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(processor), compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer, optimizers=(opt, sch),
        callbacks=[EarlyStoppingCallback(C.EARLY_STOP_PATIENCE)],
    )

def greedy_test(model, processor, test_ds, clean_text):
    from jiwer import wer as _wer, cer as _cer
    model.eval(); refs, hyps = [], []
    dev = next(model.parameters()).device
    for s in test_ds:
        fe = torch.tensor(s["input_features"]).unsqueeze(0).to(dev)
        am = torch.tensor(s["attention_mask"]).unsqueeze(0).to(dev)
        with torch.no_grad():
            lg = model(fe, attention_mask=am).logits
        h = clean_text(processor.tokenizer.batch_decode(torch.argmax(lg, -1))[0].replace("|", " "))
        li = torch.tensor(s["labels"]); li[li == -100] = processor.tokenizer.pad_token_id
        r = clean_text(processor.tokenizer.decode(li, group_tokens=False).replace("|", " "))
        refs.append(r); hyps.append(h)
    pr = [(p, l) for p, l in zip(hyps, refs) if l.strip()]
    P, L = zip(*pr)
    return _wer(list(L), list(P)) * 100, _cer(list(L), list(P)) * 100, refs, hyps
