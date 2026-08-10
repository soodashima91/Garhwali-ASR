"""garhwali_asr.data.loaders — processor + speed-aug train features + eval features."""
import os, json, numpy as np
from .. import config as C
from .prepare import clean_text, decode_audio, load_splits

def build_processor():
    from transformers import (Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor,
                              Wav2Vec2BertProcessor)
    vocab_path = os.path.join(C.DATA_DIR, "vocab.json")
    tok = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]",
                               word_delimiter_token="|")
    fe = SeamlessM4TFeatureExtractor.from_pretrained(C.BASE_MODEL)
    proc = Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tok)
    return proc

def _feat_label(proc, speech, sr, text):
    feats = proc(audio=speech, sampling_rate=sr).input_features[0]
    labels = proc.tokenizer(clean_text(text)).input_ids
    return feats, labels

def build_train_aug(proc, ds, text_field, audio_field, factors=None):
    import librosa
    factors = factors or C.SPEED_FACTORS
    out = {"input_features": [], "attention_mask": [], "labels": []}
    n_in = 0
    for ex in ds:
        text = ex[text_field]
        if not text or not clean_text(text).strip():
            continue
        n_in += 1
        speech, sr = decode_audio(ex[audio_field])
        for fct in factors:
            aug = speech if fct == 1.0 else librosa.effects.time_stretch(speech, rate=fct)
            fe, la = _feat_label(proc, aug, sr, text)
            if len(la) == 0:
                continue
            out["input_features"].append(fe)
            out["attention_mask"].append(np.ones(len(fe), dtype=np.int64))
            out["labels"].append(la)
    return out, n_in

def build_eval(proc, ds, text_field, audio_field):
    out = {"input_features": [], "attention_mask": [], "labels": []}
    for ex in ds:
        text = ex[text_field]
        if not text or not clean_text(text).strip():
            continue
        speech, sr = decode_audio(ex[audio_field])
        fe, la = _feat_label(proc, speech, sr, text)
        if len(la) == 0:
            continue
        out["input_features"].append(fe)
        out["attention_mask"].append(np.ones(len(fe), dtype=np.int64))
        out["labels"].append(la)
    return out

def make_datasets(smoke=False, factors=None):
    from datasets import Dataset
    train, val, test, text_field, audio_field = load_splits()
    if smoke:
        train = train.select(range(min(40, len(train))))
        val   = val.select(range(min(10, len(val))))
        test  = test.select(range(min(10, len(test))))
    proc = build_processor()
    train_d, n_in = build_train_aug(proc, train, text_field, audio_field, factors)
    val_d  = build_eval(proc, val, text_field, audio_field)
    test_d = build_eval(proc, test, text_field, audio_field)
    train_ds = Dataset.from_dict(train_d)
    val_ds   = Dataset.from_dict(val_d)
    test_ds  = Dataset.from_dict(test_d)
    fac = factors or C.SPEED_FACTORS
    exp = n_in * len(fac)
    info = {"n_train_in": n_in, "n_train_aug": len(train_ds),
            "expected_aug": exp, "n_val": len(val_ds), "n_test": len(test_ds),
            "vocab_size": len(proc.tokenizer), "aug_ok": (len(train_ds) == exp)}
    return proc, train_ds, val_ds, test_ds, info
