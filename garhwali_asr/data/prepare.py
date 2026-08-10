"""garhwali_asr.data.prepare — normaliser, official-split loader, vocab, soundfile audio decode (no torchcodec)."""
import os, re, json, argparse, io, numpy as np
from .. import config as C

_normalizer = None
def _get_normalizer():
    global _normalizer
    if _normalizer is None:
        import os
        if not os.path.isdir(C.INDIC_RESOURCES):
            raise FileNotFoundError(
                f"indic_nlp_resources not found at {C.INDIC_RESOURCES!r}. "
                "It is not bundled in this release. Clone it and set the path:\n"
                "  git clone https://github.com/anoopkunchukuttan/indic_nlp_resources\n"
                "  export INDIC_RESOURCES=/path/to/indic_nlp_resources")
        from indicnlp import common
        common.set_resources_path(C.INDIC_RESOURCES)
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        _normalizer = IndicNormalizerFactory().get_normalizer("hi", remove_nuktas=False)
    return _normalizer

def clean_text(t):
    if not t:
        return ""
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'\{[^}]*\}', '', t)
    t = re.sub(r'--+', '', t)
    t = t.translate({ord(c): None for c in C.ZERO_WIDTH})
    t = re.sub(r'[।॥,.\-!?;:\'\"|()_]+', ' ', t)
    t = re.sub(r'[A-Za-z]+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return _get_normalizer().normalize(t)

def decode_audio(a):
    """Decode an Audio cell loaded with decode=False (bytes/path) via soundfile.
    Also handles already-decoded dict {'array','sampling_rate'} for safety."""
    import soundfile as sf
    # already-decoded dict
    if isinstance(a, dict) and "array" in a and a["array"] is not None:
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
    else:
        # decode=False gives {'bytes':..., 'path':...}
        if isinstance(a, dict) and a.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        elif isinstance(a, dict) and a.get("path") is not None:
            data, sr = sf.read(a["path"], dtype="float32")
        else:
            raise ValueError(f"Unrecognized audio cell: {type(a)} keys="
                             f"{list(a.keys()) if isinstance(a, dict) else a}")
        arr = np.asarray(data, dtype=np.float32)
        sr = int(sr)
    # mono
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    # resample to target if needed
    if sr != C.SAMPLE_RATE:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=C.SAMPLE_RATE).astype(np.float32)
        sr = C.SAMPLE_RATE
    return arr, sr

def load_splits():
    from datasets import load_dataset, concatenate_datasets, Audio
    dd = load_dataset(C.HF_DATASET, C.HF_CONFIG, token=True,
                      download_mode="reuse_dataset_if_exists")
    any_split = dd["train"] if hasattr(dd, "keys") else dd
    cols = any_split.column_names
    text_field = next((c for c in C.TEXT_FIELD_CANDIDATES if c in cols), None)
    if text_field is None:
        raise KeyError(f"No transcript field among {cols}")
    audio_field = "audio" if "audio" in cols else next(
        (c for c in cols if "audio" in c.lower()), None)
    if C.SPLIT_PROTOCOL == "official":
        train, val, test = dd["train"], dd["validation"], dd["test"]
    elif C.SPLIT_PROTOCOL == "paper_pooled":
        pooled = concatenate_datasets([dd["train"], dd["validation"], dd["test"]]).shuffle(seed=C.SPLIT_SEED)
        n = len(pooled); tr = int(0.8*n); va = int(0.1*n)
        train = pooled.select(range(tr)); val = pooled.select(range(tr, tr+va)); test = pooled.select(range(tr+va, n))
    else:
        raise ValueError(C.SPLIT_PROTOCOL)
    # decode=False -> we decode bytes ourselves with soundfile (no torchcodec)
    train = train.cast_column(audio_field, Audio(decode=False))
    val   = val.cast_column(audio_field,   Audio(decode=False))
    test  = test.cast_column(audio_field,  Audio(decode=False))
    return train, val, test, text_field, audio_field

def build_vocab(train, val, text_field):
    chars = set()
    for s in list(train) + list(val):
        chars.update(clean_text(s[text_field]))
    deva = sorted(c for c in chars if '\u0900' <= c <= '\u097F')
    non  = sorted(c for c in chars if not ('\u0900' <= c <= '\u097F') and not c.isspace())
    vocab = {c: i for i, c in enumerate(deva)}
    vocab["|"] = len(vocab); vocab["[UNK]"] = len(vocab); vocab["[PAD]"] = len(vocab)
    return vocab, deva, non

def main(smoke=False):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    train, val, test, text_field, audio_field = load_splits()
    print(f"text_field={text_field!r} audio_field={audio_field!r}")
    print(f"splits: train {len(train)} | val {len(val)} | test {len(test)}")
    if smoke:
        train = train.select(range(min(40, len(train))))
        val   = val.select(range(min(10, len(val))))
        test  = test.select(range(min(10, len(test))))
        print(f"SMOKE subset: train {len(train)} | val {len(val)} | test {len(test)}")
        arr, sr = decode_audio(train[0][audio_field])
        print(f"decode_audio OK: shape={arr.shape} sr={sr} dtype={arr.dtype}")
        print("clean_text sample:", repr(clean_text(train[1][text_field])))
    vocab, deva, non = build_vocab(train, val, text_field)
    print(f"vocab size: {len(vocab)} (Devanagari {len(deva)} + 3 specials)")
    print(f"non-Devanagari leftovers (want empty): {non}")
    manifest = {"split_protocol": C.SPLIT_PROTOCOL, "text_field": text_field,
                "audio_field": audio_field,
                "sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "vocab_size": len(vocab), "smoke": smoke}
    with open(os.path.join(C.DATA_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(C.DATA_DIR, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved vocab.json and split_manifest.json to", C.DATA_DIR)
    return manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(smoke=a.smoke)
