#!/usr/bin/env python
"""Build speed-augmented train + eval features ONCE and cache to disk.
Every training job then loads the cache instead of rebuilding (no 15x redundancy,
no per-job memory spike). Run once after vocab is built:
    python prepare_features.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from garhwali_asr import config as C
from garhwali_asr.data.loaders import (build_processor, build_train_aug, build_eval)
from garhwali_asr.data.prepare import load_splits
from datasets import Dataset

FEAT_DIR = os.path.join(C.DATA_DIR, "features")

def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    train, val, test, text_field, audio_field = load_splits()
    proc = build_processor()
    print(f"Building features: train {len(train)} (x{len(C.SPEED_FACTORS)} aug) | "
          f"val {len(val)} | test {len(test)}")

    train_d, n_in = build_train_aug(proc, train, text_field, audio_field)
    Dataset.from_dict(train_d).save_to_disk(os.path.join(FEAT_DIR, "train"))
    print(f"  train cached: {len(train_d['labels'])} (from {n_in} inputs)")

    val_d = build_eval(proc, val, text_field, audio_field)
    Dataset.from_dict(val_d).save_to_disk(os.path.join(FEAT_DIR, "val"))
    print(f"  val cached: {len(val_d['labels'])}")

    test_d = build_eval(proc, test, text_field, audio_field)
    Dataset.from_dict(test_d).save_to_disk(os.path.join(FEAT_DIR, "test"))
    print(f"  test cached: {len(test_d['labels'])}")

    info = {"n_train_in": n_in, "n_train_aug": len(train_d["labels"]),
            "expected_aug": n_in * len(C.SPEED_FACTORS),
            "n_val": len(val_d["labels"]), "n_test": len(test_d["labels"]),
            "vocab_size": len(proc.tokenizer),
            "aug_ok": len(train_d["labels"]) == n_in * len(C.SPEED_FACTORS)}
    with open(os.path.join(FEAT_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  info: {info}")
    print(f"Features cached to {FEAT_DIR}")

if __name__ == "__main__":
    main()
