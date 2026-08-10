# Data

This repository ships **no audio and no transcripts**. Everything below is
obtained from its original source, under that source's own terms.

## VAANI Garhwali (main corpus)

- Source: `ARTPARK-IISc/Vaani-transcription-part` on the Hugging Face Hub,
  config `Garhwali`.
- Distributed by its authors (ARTPARK-IISc) for research use, under the terms
  that accompany the corpus. We add no recordings and redistribute no reference
  text; consent and collection protocols are those of the original corpus.
- Access is gated on the Hub. Request access there, then authenticate:
  ```bash
  huggingface-cli login
  ```
- The scripts use the **official** train/validation/test splits exactly as
  shipped (4,778 / 666 / 450 utterances); we do not re-partition. See
  `results/data/split_manifest.json`.

## FLEURS Hindi (transfer source, Section 4.4 only)

- Source: `google/fleurs`, config `hi_in`, splits `train+validation`.
- Loaded on demand by `scripts/train_transfer.py`. Only needed to reproduce the
  Hindi->Garhwali transfer experiment.

## indic_nlp_resources (text normalisation)

The `clean_text` normaliser depends on the Indic NLP resources, which are a
large separate download and are **not** bundled here:

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources
export INDIC_RESOURCES=/path/to/indic_nlp_resources
```

`results/data/vocab.json` (the 66-token vocabulary) is committed so the token
set is fixed and inspectable; it is derived from the official train+validation
transcripts under this exact normaliser.

## What we do release

Only transcript-free numerical results, under `results/`:

- `data/vocab.json`, `data/split_manifest.json`
- `aggregate/*.csv`, `aggregate/*.json` — per-seed WER/CER, per-phonological-
  class error rates, Holm-corrected significance, power analysis, transfer
  comparison, bootstrap intervals
- `probing/*.json` — layer-wise probe WER/CER (seed 42)

Per-utterance predictions are intentionally omitted, because a reference/
hypothesis pair reconstructs corpus text. If you need them for your own
analysis, the trainers regenerate them locally with `--dump-predictions`; do
not redistribute those files.
