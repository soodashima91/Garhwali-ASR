# Seeds Before Objectives: Multi-Seed Garhwali ASR

Code and per-seed numerical results for the paper *"Seeds Before Objectives:
Rethinking Evaluation for Low-Resource Garhwali ASR"* (ICNLSP 2026).

The central claim is methodological: at low-resource-dialect corpus sizes,
single-run comparisons are unreliable, so we evaluate every intervention over
five seeds on the **official VAANI splits** with paired significance testing.

## TL;DR

At the corpus sizes typical of low-resource dialects, single-run ASR comparisons
can show "gains" that are really just seed noise. We build the first reproducible
multi-seed benchmark for **Garhwali** on the official VAANI splits, evaluating
every intervention over five seeds with significance testing. The result: neither
Focal CTC nor a matra-weighted objective beats standard CTC, the matra objective
fails to cut even its targeted errors, and Hindi→Garhwali transfer adds nothing
over direct fine-tuning. What holds up is mundane — **w2v-BERT 2.0 with standard
CTC reaches 47.0% WER over five seeds**, beating the larger MMS-1B, and speed
augmentation gives a small, consistent gain.

## Layout

```
garhwali_asr/            importable package (the paper's methods)
  config.py              single source of truth: seeds, hyperparameters, paths
  phonology.py           Devanagari phonological character classes (matra, etc.)
  losses.py              standard / focal / matra-weighted CTC objectives (App. D)
  data/prepare.py        text normaliser, official-split loader, vocab builder
  data/loaders.py        processor + speed-augmented feature extraction
  models/w2vbert.py      w2v-BERT 2.0 setup + objective-routing Trainer
  analyze/class_errors.py per-phonological-class error rates (char alignment)
  analyze/stats.py       bootstrap CIs, Holm-corrected paired tests, power inputs
scripts/
  prepare_features.py    build + cache speed-augmented features once
  train_one.py           train one (objective, seed) w2v-BERT 2.0 run
  train_transfer.py      two-stage Hindi(FLEURS)->Garhwali transfer
  probe.py               layer-wise linear CTC probing (exploratory, App. C)
  aggregate.py           per-seed result.json -> tables + significance
  aggregate_transfer.py  transfer runs vs direct standard CTC
  aggregate_class_errors.py per-class error aggregation across seeds
  power_analysis.py      post-hoc power / seeds-for-80%-power (Sec 4.5)
  reproduce_stats.py     regenerate headline stats from committed numbers (no GPU)
  test_logic.py          unit checks for losses / phonology / alignment
baselines/
  baseline_data.py       raw-waveform data path for wav2vec2-family + MMS
  train_baseline.py      train XLS-R / HuBERT / MMS baseline on the same vocab
results/
  data/vocab.json        the 66-token Devanagari vocabulary
  data/split_manifest.json  official split sizes (4778/666/450)
  aggregate/*.csv,*.json    per-seed WER/CER, per-class errors, significance,
                            power, transfer, bootstrap  (numbers only)
  probing/*.json            layer-wise probe WER/CER (seed 42)
```

## Quick start: verify the statistics (no GPU, no data)

```bash
pip install -r requirements.txt
python scripts/reproduce_stats.py
```

This reads the committed per-seed numbers and reprints Table 2, the Section 4.1
Holm-corrected paired tests (no pair significant), the Section 4.5 power
analysis, and the Section 4.4 transfer null — the paper's core methodological
result, checkable in seconds.

```bash
python scripts/test_logic.py     # unit checks: gamma=0 == standard CTC, etc.
```

## Reproduce from scratch (GPU)

Set an output root (defaults to `./results`) and authenticate to the Hub for
VAANI access:

```bash
export GARHWALI_RESULTS_ROOT=/path/to/results
huggingface-cli login            # VAANI is access-gated by its authors

# 1. build vocab + manifest, then cache speed-augmented features
python -m garhwali_asr.data.prepare
python scripts/prepare_features.py

# 2. main systems: 3 objectives x 5 seeds
for obj in standard focal matra; do
  for s in 42 123 777 2025 1234; do
    python scripts/train_one.py --objective $obj --seed $s
  done
done

# 3. aggregate + significance + power
python scripts/aggregate.py --exp objective
python scripts/power_analysis.py --exp objective
```

The seeds, hyperparameters, and split protocol are fixed in
`garhwali_asr/config.py`; only the objective and the augmentation flag differ
across the compared runs, so any difference in results is attributable to those
two factors. All decoding is greedy CTC (no external LM), held fixed across
systems by design.

Hardware in the paper: a single NVIDIA A100 (80 GB). ~1.4 h/run without
augmentation, ~2.9 h/run with 3x speed augmentation.

## Notes on faithful reproduction

- **Normalisation is load-bearing.** WER/CER are only comparable under the
  exact `clean_text` pipeline in `data/prepare.py` (markup/zero-width-joiner
  removal, Latin-script stripping, whitespace collapsing) and the committed
  `results/data/vocab.json`. Both are used identically by every system.
- **Official splits only.** We use the VAANI train/validation/test splits as
  shipped (4778/666/450); no internal re-partitioning.
- Pin versions from `requirements.txt`: CTC and decoding behaviour can shift
  across `torch`/`transformers` releases.

## Citation

See the paper. Code, splits, and per-seed outputs are released to support
reproducible, variance-aware evaluation for low-resource dialectal ASR.

## Interactive companion site

An interactive summary of this paper's results — the per-seed scatter, model
comparison, objective/power analysis, and layer-wise probing — is published
from the [`docs/`](docs/) folder via GitHub Pages:

**https://soodashima91.github.io/Garhwali-ASR/**

Every figure on the site is generated from the committed numbers in
`results/aggregate/` and `results/probing/`.

## Note on the HuBERT baseline

`results/aggregate/master_results.csv` reports HuBERT Large at **62.14 WER**,
the raw mean over all five seeds. Seed 777 failed to converge (validation WER
0.73 vs. ~0.66 for the other seeds). The paper therefore excludes that seed and
reports the four-good-seed mean of **60.90 ± 0.49** (see the Appendix table and
footnote). Both numbers are correct for what they measure; the paper uses the
outlier-excluded value.
