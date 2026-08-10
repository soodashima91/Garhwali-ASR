"""
garhwali_asr.phonology
=======================
Phonological character classes for Devanagari (Unicode block U+0900-U+097F),
used to build class-weighted CTC training and per-class error metrics.

Class membership is defined by Unicode codepoint, following:
  - The Unicode Standard, Devanagari block (U+0900-U+097F).
  - Standard Devanagari phonological descriptions (dependent vowel signs =
    "matra"; the aspirated/retroflex consonant inventory).

IMPORTANT: these classes are defined on the WRITTEN
(orthographic) form, not on a phonetic transcription. "matra" here means the
dependent vowel signs as encoded in Unicode, which is the unit the CTC model
predicts. Independent vowels (अ आ इ ...) are NOT matras; they are full vowel
characters and are placed in their own class. This distinction matters because
the paper's residual-error finding is about the dependent vowel signs that
attach to consonants.
"""

# ----------------------------------------------------------------------
# Dependent vowel signs ("matras") -- U+093A..U+094C plus U+094E..U+094F,
# and the vowel-sign extensions. These are the diacritics that attach to a
# base consonant. This is the class the paper identifies as the dominant
# residual error.
# ----------------------------------------------------------------------
MATRA_SIGNS = {
    "\u093A",  # ꣰  vowel sign OE (rare)
    "\u093B",  # vowel sign OOE (rare)
    "\u093E",  # ा  AA
    "\u093F",  # ि  I
    "\u0940",  # ी  II
    "\u0941",  # ु  U
    "\u0942",  # ू  UU
    "\u0943",  # ृ  vocalic R
    "\u0944",  # ॄ  vocalic RR
    "\u0945",  # ॅ  candra E
    "\u0946",  # ॆ  short E
    "\u0947",  # े  E
    "\u0948",  # ै  AI
    "\u0949",  # ॉ  candra O
    "\u094A",  # ॊ  short O
    "\u094B",  # ो  O
    "\u094C",  # ौ  AU
    "\u094E",  # ॎ  PRISHTHAMATRA E
    "\u094F",  # ॏ  AW
    "\u0956",  # ॖ  vowel sign UE
    "\u0957",  # ॗ  vowel sign UUE
    "\u0962",  # ॢ  vocalic L
    "\u0963",  # ॣ  vocalic LL
}

# Anusvara / candrabindu / visarga / nukta / virama: nasalization, vowel
# modification and the "halant" that suppresses the inherent vowel. The paper
# flags nasalization drops as an error type, so we track these too.
NASAL_MODIFIER_SIGNS = {
    "\u0900",  # ॐ candrabindu variants / inverted
    "\u0901",  # ँ  candrabindu (nasalization)
    "\u0902",  # ं  anusvara
    "\u0903",  # ः  visarga
}
VIRAMA = {"\u094D"}          # ्  halant / virama (vowel suppression)
NUKTA  = {"\u093C"}          # ़  nukta

# ----------------------------------------------------------------------
# Aspirated consonants -- the "h"-bearing members of each stop series.
# Devanagari encodes each aspirated stop as its OWN codepoint (not consonant
# + h), so aspiration is a single-character distinction the model must make.
# ----------------------------------------------------------------------
ASPIRATED_CONSONANTS = {
    "\u0916",  # ख  kha
    "\u0918",  # घ  gha
    "\u091D",  # झ  jha
    "\u0920",  # ठ  ttha (retroflex, also aspirated)
    "\u0922",  # ढ  ddha (retroflex, also aspirated)
    "\u0925",  # थ  tha
    "\u0927",  # ध  dha
    "\u092B",  # फ  pha
    "\u092D",  # भ  bha
}

# ----------------------------------------------------------------------
# Retroflex consonants -- the ट-series (cerebral) stops + retroflex nasal,
# flap, and sibilant. ठ and ढ are BOTH retroflex AND aspirated; member of
# both sets (handled by allowing multi-class membership in the weighter).
# ----------------------------------------------------------------------
RETROFLEX_CONSONANTS = {
    "\u091F",  # ट  tta
    "\u0920",  # ठ  ttha
    "\u0921",  # ड  dda
    "\u0922",  # ढ  ddha
    "\u0923",  # ण  nna (retroflex nasal)
    "\u0931",  # ऱ  rra
    "\u0933",  # ळ  lla (retroflex lateral)
    "\u0934",  # ऴ  llla
    "\u0937",  # ष  ssa (retroflex sibilant)
    "\u095C",  # ड़ dddha (nukta form)
    "\u095D",  # ढ़ rha (nukta form)
}

# Independent (full) vowels -- distinct from matras; kept separate.
INDEPENDENT_VOWELS = {
    "\u0904","\u0905","\u0906","\u0907","\u0908","\u0909","\u090A","\u090B",
    "\u090C","\u090D","\u090E","\u090F","\u0910","\u0911","\u0912","\u0913",
    "\u0914","\u0960","\u0961",
}

# Named classes available for weighting. A character may belong to several
# (e.g. ठ is both aspirated and retroflex); the weighter takes the MAX weight
# over the classes a character belongs to.
CLASS_SETS = {
    "matra":      MATRA_SIGNS,
    "nasal":      NASAL_MODIFIER_SIGNS,
    "virama":     VIRAMA,
    "nukta":      NUKTA,
    "aspirated":  ASPIRATED_CONSONANTS,
    "retroflex":  RETROFLEX_CONSONANTS,
    "indep_vowel": INDEPENDENT_VOWELS,
}


def classes_of(ch: str):
    """Return the set of phonological class names a character belongs to."""
    return {name for name, s in CLASS_SETS.items() if ch in s}


def char_weight(ch: str, class_weights: dict, default: float = 1.0) -> float:
    """
    Weight for a single character = max weight over the classes it belongs to,
    or `default` if it belongs to none. `class_weights` maps class name -> weight.
    Using max (not sum/product) keeps weights interpretable and bounded by the
    largest single salient property of the character.
    """
    cls = classes_of(ch)
    if not cls:
        return default
    return max([class_weights.get(c, default) for c in cls] + [default]) \
        if any(c in class_weights for c in cls) else default


def build_token_weights(vocab: dict, class_weights: dict, default: float = 1.0):
    """
    Map a tokenizer vocab {char: id} to a list `w` where w[id] is the weight of
    that token's character. Special tokens (|, [UNK], [PAD]) get `default`.
    """
    inv = {i: c for c, i in vocab.items()}
    n = max(vocab.values()) + 1
    w = [default] * n
    for i in range(n):
        ch = inv.get(i, "")
        if ch in ("|", "[UNK]", "[PAD]") or len(ch) != 1:
            w[i] = default
        else:
            w[i] = char_weight(ch, class_weights, default)
    return w
