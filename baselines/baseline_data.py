"""Raw-audio data path for wav2vec2-family + MMS baselines.
Reuses garhwali_asr's normalizer, official splits, and soundfile decode,
but emits RAW WAVEFORM (input_values) instead of log-Mel features, since
these models have their own conv feature extractor."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from garhwali_asr import config as C
from garhwali_asr.data.prepare import clean_text, decode_audio, load_splits

# model checkpoints
MODELS = {
    "xlsr53":   "facebook/wav2vec2-large-xlsr-53",
    "xlsr300m": "facebook/wav2vec2-xls-r-300m",
    "hubert":   "facebook/hubert-large-ls960-ft",
    "mms":      "facebook/mms-1b-all",
}

def build_tokenizer():
    """Reuse the SAME 66-token Garhwali vocab as the main pipeline."""
    from transformers import Wav2Vec2CTCTokenizer
    vocab_path = os.path.join(C.DATA_DIR, "vocab.json")
    return Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]",
                                word_delimiter_token="|")

def build_feature_extractor():
    from transformers import Wav2Vec2FeatureExtractor
    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=C.SAMPLE_RATE,
                                    padding_value=0.0, do_normalize=True,
                                    return_attention_mask=True)

def _prep_split(ds, fe, tok, text_field, audio_field, factors=None):
    out = {"input_values": [], "labels": []}
    n_in = 0
    import librosa
    for ex in ds:
        text = ex[text_field]
        if not text or not clean_text(text).strip():
            continue
        speech, sr = decode_audio(ex[audio_field])
        speech = np.asarray(speech, dtype=np.float32)
        labels = tok(clean_text(text)).input_ids
        if len(labels) == 0:
            continue
        n_in += 1
        facs = factors if factors else [1.0]
        for fct in facs:
            aug = speech if fct == 1.0 else librosa.effects.time_stretch(speech, rate=fct)
            iv = fe(aug, sampling_rate=C.SAMPLE_RATE).input_values[0]
            out["input_values"].append(np.asarray(iv, dtype=np.float32))
            out["labels"].append(labels)
    return out, n_in

def make_baseline_datasets(factors=None, smoke=False):
    from datasets import Dataset
    train, val, test, text_field, audio_field = load_splits()
    if smoke:
        train = train.select(range(min(40, len(train))))
        val   = val.select(range(min(10, len(val))))
        test  = test.select(range(min(10, len(test))))
    tok = build_tokenizer()
    fe  = build_feature_extractor()
    tr, n_in = _prep_split(train, fe, tok, text_field, audio_field, factors)
    va, _    = _prep_split(val,   fe, tok, text_field, audio_field, None)
    te, _    = _prep_split(test,  fe, tok, text_field, audio_field, None)
    info = {"n_train_in": n_in, "n_train": len(tr["labels"]),
            "n_val": len(va["labels"]), "n_test": len(te["labels"]),
            "vocab_size": len(tok)}
    return tok, fe, Dataset.from_dict(tr), Dataset.from_dict(va), Dataset.from_dict(te), info
