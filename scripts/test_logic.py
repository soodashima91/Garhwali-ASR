import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from garhwali_asr.phonology import classes_of, char_weight, build_token_weights, CLASS_SETS
from garhwali_asr.losses import focal_ctc_loss, matra_weighted_aux, combined_loss, per_sample_ctc
from garhwali_asr.analyze.class_errors import class_error_rates
import torch.nn.functional as F

print("="*60); print("TEST 1: phonology classification")
# ा is a matra; ख is aspirated; ट is retroflex; ठ is BOTH aspirated+retroflex
assert "matra" in classes_of("\u093E"), "AA matra not classified"
assert "aspirated" in classes_of("\u0916"), "kha not aspirated"
assert "retroflex" in classes_of("\u091F"), "tta not retroflex"
assert classes_of("\u0920") >= {"aspirated","retroflex"}, "ttha should be both"
assert classes_of("\u0915") == set(), "ka (plain) should be unclassified"
print("  ok: matra/aspirated/retroflex/both/plain all correct")
print("  classes of ठ:", classes_of("\u0920"))

print("="*60); print("TEST 2: char + token weights (max over classes)")
cw = {"matra": 3.0, "aspirated": 2.0, "retroflex": 2.5}
assert char_weight("\u093E", cw) == 3.0          # matra
assert char_weight("\u0920", cw) == 2.5          # ttha: max(2.0 asp, 2.5 retro)=2.5
assert char_weight("\u0915", cw) == 1.0          # plain -> default
vocab = {"\u093E":0,"\u0916":1,"\u091F":2,"\u0915":3,"|":4,"[UNK]":5,"[PAD]":6}
w = build_token_weights(vocab, cw)
assert w == [3.0,2.0,2.5,1.0,1.0,1.0,1.0], w
print("  ok: token weights =", w)

print("="*60); print("TEST 3: focal p VARIES and gamma=0 -> standard CTC")
torch.manual_seed(0)
B,T,V = 4, 30, 8
logits = torch.randn(B,T,V, requires_grad=True)
labels = torch.tensor([[1,2,3,-100,-100],[1,2,3,4,5],[2,2,-100,-100,-100],[1,3,5,2,4]])
tl = (labels>=0).sum(-1); il = torch.full((B,),T)
lp = F.log_softmax(logits,-1).transpose(0,1)
per = per_sample_ctc(lp, labels.masked_select(labels>=0), il, tl, blank=0)
loss_f, p = focal_ctc_loss(per, tl, gamma=0.5, temperature=10.0)
print("  per-sample loss:", [round(x,2) for x in per.tolist()])
print("  focal p        :", [round(x,3) for x in p.tolist()])
assert p.std() > 1e-4, "focal p does not vary -> modulator is a no-op!"
loss_std,_ = focal_ctc_loss(per, tl, gamma=0.0)
assert torch.allclose(loss_std, per.mean()), "gamma=0 must equal standard CTC mean"
print("  ok: p varies (std=%.4f); gamma=0 == standard CTC" % p.std())

print("="*60); print("TEST 4: matra aux loss is differentiable & responds to weights")
tok_w_flat = [1.0]*V
tok_w_matra = [1.0]*V; tok_w_matra[2] = 4.0   # pretend token id 2 is a matra
aux_flat  = matra_weighted_aux(logits, labels, tok_w_flat, blank=0, only_salient=False)
aux_weight= matra_weighted_aux(logits, labels, tok_w_matra, blank=0, only_salient=True)
print("  aux (flat weights, all frames):", round(float(aux_flat),4))
print("  aux (matra up-weighted, salient-only):", round(float(aux_weight),4))
# differentiability
g = torch.autograd.grad(aux_weight, logits, retain_graph=True, allow_unused=True)[0]
assert g is not None and torch.isfinite(g).all(), "aux not differentiable / non-finite grad"
print("  ok: aux differentiable, finite gradient")

print("="*60); print("TEST 5: combined_loss dispatch (standard/focal/matra)")
for obj in ["standard","focal","matra"]:
    L,info = combined_loss(logits, labels, il, tl, tok_w_matra, blank=0,
                           objective=obj, gamma=0.5, temperature=10.0, lam=0.3)
    assert torch.isfinite(L), f"{obj} loss non-finite"
    print(f"  {obj:9s}: loss={float(L):.4f}  info={ {k:round(v,3) for k,v in info.items()} }")
print("  ok: all three objectives return finite loss")

print("="*60); print("TEST 6: per-class error metric on a known example")
# ref has matra ी (U+0940); hyp swaps it for ि (U+093F) -> a matra error
ref = ["\u0915\u0940\u091F"]   # k + II-matra + tta(retroflex)
hyp = ["\u0915\u093F\u091F"]   # k + I-matra  + tta  (matra substituted)
cer = class_error_rates(ref, hyp)
print("  matra:", cer["matra"], " retroflex:", cer["retroflex"])
assert cer["matra"]["ref_count"] == 1 and cer["matra"]["errors"] == 1, cer["matra"]
assert cer["retroflex"]["ref_count"] == 1 and cer["retroflex"]["errors"] == 0, cer["retroflex"]
print("  ok: matra error caught (1/1), retroflex correct (0/1)")

print("\nALL TESTS PASSED")
