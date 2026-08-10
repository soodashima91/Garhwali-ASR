// Garhwali-ASR — data extracted from the committed results in this repo
// Paper: Batra, Singh, Sood, Singh & Sharma,
// "Seeds Before Objectives: Rethinking Evaluation for Low-Resource Garhwali ASR" (ICNLSP 2026)
// Every number below is transcribed from results/aggregate/*.csv and results/probing/*.json.

const DATA = {
  // ---- Headline ----
  headline: {
    wer: "47.0",
    model: "w2v-BERT 2.0 (580M)",
    seeds: 5,
    language: "Garhwali",
    speakers: "~2.5M",
    corpus: "VAANI (official 4,778 / 666 / 450 split)"
  },

  // ---- The core thesis ----
  thesis: "At corpus sizes typical of low-resource dialects, single-run comparisons can yield gains that do not replicate. We build the first reproducible multi-seed ASR benchmark for Garhwali on the official VAANI splits, with per-seed outputs and significance testing — and find that plausible objective gains dissolve into seed noise.",

  // ---- Three findings ----
  findings: [
    { n: 1, title: "Objectives don't beat standard CTC",
      detail: "Neither Focal CTC nor a phonologically-motivated matra-weighted CTC beats standard CTC under seed-level significance testing. Holm-corrected paired Wilcoxon tests find no pair significant." },
    { n: 2, title: "The targeted objective misses its target",
      detail: "The matra-weighted loss was built specifically to cut vowel-diacritic (matra) errors — yet the matra error rate is essentially unchanged (22.26% → 22.31%, p=0.42). A single-run improvement here would have been noise." },
    { n: 3, title: "Transfer adds nothing over direct fine-tuning",
      detail: "Two-stage Hindi (FLEURS) → Garhwali transfer gives 47.22% WER versus 47.02% for direct fine-tuning — no gain (p=0.81). The related standard language adds little beyond the dialect data itself." }
  ],

  // ---- Model comparison (Figure 2 / master_results.csv) ----
  // WER means over 5 seeds (HuBERT reported over 4 good seeds; seed 777 failed to converge)
  models: [
    { name: "w2v-BERT 2.0 (ours)", params: "580M", wer: 47.02, std: 0.61, ours: true },
    { name: "MMS-1B", params: "1B", wer: 48.98, std: 0.49, ours: false },
    { name: "XLS-R 300M", params: "315M", wer: 50.23, std: 0.26, ours: false },
    { name: "HuBERT Large", params: "315M", wer: 60.90, std: 0.49, ours: false, note: "4 seeds; seed 777 failed to converge" },
    { name: "Whisper Large-v3", params: "1.66B", wer: 66.35, std: 3.64, ours: false, note: "reference only, 3 valid seeds" }
  ],

  // ---- Objective comparison (Table 2 / main_results.csv) ----
  objectives: [
    { name: "Standard CTC", noaug: 48.10, noaugStd: 0.74, aug: 47.02, augStd: 0.61, cer: 16.99, matraErr: 22.26 },
    { name: "Focal CTC", noaug: 49.18, noaugStd: 1.17, aug: 47.83, augStd: 0.68, cer: 17.35, matraErr: 22.78 },
    { name: "Matra-weighted (ours)", noaug: 49.00, noaugStd: 0.22, aug: 47.42, augStd: 0.78, cer: 17.22, matraErr: 22.31 }
  ],

  // ---- Per-seed WER (significance.json) shows the spread that motivates multi-seed ----
  perSeed: {
    seeds: [42, 123, 777, 2025, 1234],
    standard: [46.73, 46.32, 47.93, 47.25, 46.88],
    focal:    [46.77, 48.40, 48.36, 47.58, 48.06],
    matra:    [46.52, 47.48, 48.65, 47.05, 47.40]
  },

  // ---- Significance (significance.json, primary seed-level, Holm-corrected) ----
  significance: [
    { pair: "Standard vs Focal", meanDiff: "+0.81", p: "0.19", sig: false },
    { pair: "Standard vs Matra", meanDiff: "+0.40", p: "0.38", sig: false },
    { pair: "Focal vs Matra", meanDiff: "−0.42", p: "0.38", sig: false }
  ],

  // ---- Power analysis (power_analysis.json) ----
  power: {
    largestGap: "Standard vs Focal (+0.81 WER)",
    achievedPower: "0.39",
    seedsNeeded: 11,
    note: "The largest objective gap has only 39% power at 5 seeds; ~11 seeds would be needed for 80% power. So the differences are reported as not reliably established, rather than absent."
  },

  // ---- Speed augmentation (the one intervention that holds up) ----
  augmentation: {
    factors: "0.9× / 1.0× / 1.1×",
    trainBefore: "4,778",
    trainAfter: "14,334",
    effect: "Standard CTC improves 48.10% → 47.02% WER — a small but consistent gain across seeds."
  },

  // ---- Layer-wise probing (Appendix C, probe_*_seed42.json) ----
  probing: {
    layers: [4, 8, 12, 16, 20, 24],
    finetuned: [88.4, 72.6, 66.7, 51.5, 48.1, 46.6],
    base:      [97.4, 93.0, 88.7, 85.8, 95.7, 99.5],
    note: "In the base model, decodable linguistic content peaks mid-network (layer 16, 85.8% WER) and degrades in the top layers — the 'autoencoder' profile of self-supervised speech encoders. Fine-tuning repurposes exactly those upper layers: probe error falls monotonically with depth to the top layer (24), reaching near full-model performance. Exploratory, single-seed."
  },

  // ---- Setup ----
  setup: {
    encoder: "w2v-BERT 2.0, 24-layer, 580M params, all layers fine-tuned",
    head: "trainable 66-token Devanagari CTC head; feature projection frozen",
    features: "80-dim log-Mel (SeamlessM4T), 16 kHz",
    decode: "greedy CTC decoding",
    compute: "single NVIDIA A100-SXM4 (80GB); ~1.4 h/run without augmentation, ~2.9 h with 3× speed augmentation",
    seeds: "42, 123, 777, 1234, 2025",
    vocab: "66-token Devanagari (incl. word-delimiter, [UNK], [PAD])"
  },

  // ---- Error profile (Table 10 — LLM-assisted, ordering only) ----
  errorProfile: [
    { cat: "Orthographic / phonetic drift", pct: "54.0" },
    { cat: "Semantic / lexical substitution", pct: "39.1" },
    { cat: "Word-boundary / spacing", pct: "2.3" },
    { cat: "Deletions", pct: "1.5" },
    { cat: "Dialect → Hindi (true)", pct: "1.4" },
    { cat: "Insertions", pct: "1.1" }
  ],
  errorNote: "Functional error categorisation reconciled across four independent LLMs (GPT-5.5, Claude Sonnet 4.6, Gemini 3.5 Flash, DeepSeek-V3). Percentages convey ordering, not seed-stable quantities. Errors are dominated by fine-grained orthographic/phonetic drift, not gross mistakes.",

  // ---- Limitations ----
  limitations: [
    "Single low-resource language (Garhwali) and one corpus (VAANI); the methodological argument is general, but the specific numbers are corpus-specific.",
    "Probing analysis is single-seed and exploratory — read qualitatively, not as a precise localisation.",
    "Objective and transfer nulls are scoped to this data regime; larger corpora could change the picture."
  ]
};
