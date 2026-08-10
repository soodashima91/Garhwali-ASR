"""garhwali_asr.config — single source of truth (flat layout)."""
import os

# Flat layout: this file is .../ICNLSP_2026/garhwali_asr/config.py
_HERE = os.path.dirname(os.path.abspath(__file__))      # .../ICNLSP_2026/garhwali_asr
PROJECT_ROOT = os.path.dirname(_HERE)                   # .../ICNLSP_2026
RESULTS_ROOT = os.environ.get("GARHWALI_RESULTS_ROOT",
                              os.path.join(PROJECT_ROOT, "results"))
DATA_DIR = os.path.join(RESULTS_ROOT, "data")
RUNS_DIR = os.path.join(RESULTS_ROOT, "runs")
AGG_DIR  = os.path.join(RESULTS_ROOT, "aggregate")
# Path to indic_nlp_resources (needed by the text normaliser). NOT bundled in
# this release (it is a large separate download). Clone it and point here:
#   git clone https://github.com/anoopkunchukuttan/indic_nlp_resources
#   export INDIC_RESOURCES=/path/to/indic_nlp_resources
# Defaults to a folder next to the package if you place it there.
INDIC_RESOURCES = os.environ.get(
    "INDIC_RESOURCES", os.path.join(_HERE, "indic_nlp_resources"))

HF_DATASET = "ARTPARK-IISc/Vaani-transcription-part"
HF_CONFIG  = "Garhwali"
SPLIT_PROTOCOL = "official"
SPLIT_SEED = 42
SAMPLE_RATE = 16000
TEXT_FIELD_CANDIDATES = ["transcript", "transcription", "text", "sentence", "normalized_text"]

SEEDS = [42, 123, 777, 2025, 1234]

BASE_MODEL = "facebook/w2v-bert-2.0"
HIDDEN_DIM = 1024
N_ENCODER_LAYERS = 24

ENCODER_LR = 3e-5
HEAD_LR = 1e-3
NUM_EPOCHS = 20
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 5
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
SPEED_FACTORS = [0.9, 1.0, 1.1]

FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 0.5
FOCAL_TEMPERATURE = 10.0
CLASS_WEIGHTS = {"matra": 3.0, "aspirated": 2.0, "retroflex": 2.5}
MATRA_LAMBDA = 0.3
MATRA_FOCAL_ON_BASE = True

GAMMA_GRID = [0.25, 0.5, 0.75, 1.0]
LAMBDA_GRID = [0.1, 0.3, 0.5]

ZERO_WIDTH = "\u200c\u200d\ufeff"
