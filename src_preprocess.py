# src_preprocess.py
import re
import unicodedata

# --- Optional: Indic NLP (better normalization/tokenization if available) ---
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  # library name changed in 0.92
    from indicnlp.tokenize import indic_tokenize
    HAS_INDIC = True
except Exception:
    HAS_INDIC = False

# Marathi stopwords (compact but useful set)
MARATHI_STOPWORDS = {
    "आहे","होते","होता","होती","नाही","आणि","की","या","त्या","हे","ते","तो","ती",
    "किंवा","पर्यंत","पण","मात्र","कारण","ज्यामुळे","वरील","खाली","मध्ये","वर",
    "साठी","अधिक","काही","सुद्धा","होणार","असे","अशी","असेल","असतील","आला","आली",
    "आले","गेला","गेली","गेले","झाला","झाली","झाले","होऊन","होऊनही","फक्त","मग",
    "तर","आत","बाहेर","पुढे","मागे","आज","उद्या","काल","इथे","तिथे","कोण","काय",
    "कुठे","कधी","का","कसा","कशी","कसे","कोणता","कोणती","कोणते","सर्व","प्रत्येक",
    "कुठलाही","नंतर","आधी","आतापर्यंत","अजून","अजूनही","तरी","जरी","जसे","तसे"
}

# Small lemma dictionary (extend as you see patterns)
LEMMAS = {
    "आहे":"असणे","आहेत":"असणे","होतो":"असणे","होते":"असणे","होता":"असणे","होती":"असणे","होईल":"असणे","होणार":"असणे",
    "झाला":"होणे","झाली":"होणे","झाले":"होणे",
    "करतो":"करणे","करते":"करणे","केले":"करणे","केली":"करणे","केलं":"करणे","करतील":"करणे","करणार":"करणे",
    "गेला":"जाणे","गेली":"जाणे","गेले":"जाणे","जातो":"जाणे","जाते":"जाणे","जातील":"जाणे","जाणार":"जाणे",
    "आला":"येणे","आली":"येणे","आले":"येणे","येतो":"येणे","येते":"येणे","येतील":"येणे","येणार":"येणे",
    "दिला":"देणे","दिली":"देणे","दिले":"देणे","देतो":"देणे","देते":"देणे",
    "घेतला":"घेणे","घेतली":"घेणे","घेतले":"घेणे","घेतो":"घेणे","घेते":"घेणे",
    "बोलला":"बोलणे","बोलली":"बोलणे","बोलले":"บोलणे","बोलतो":"बोलणे","बोलते":"बोलणे",
}

# Conservative suffix list for stemming (longest first)
STEM_SUFFIXES = sorted([
    "ांमध्ये","ांकडून","ांबरोबर","ांपासून","ांच्या",
    "ींतील","ांतील","ातील",
    "ींनी","ींचे","ींची","ींचा",
    "ांना","ांचे","ांची","ांचा",
    "ांच्या","करून","मध्ये","पासून","कडे","वरून",
    "ांनी","ातून","हून",
    "तील","वरील","खालील",
    "णे","तो","ते","ती",
    "ला","ने","चे","ची","चा","त","वर"
], key=len, reverse=True)

# ---------- Core helpers ----------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if HAS_INDIC:
        try:
            normalizer = IndicNormalizerFactory().get_normalizer("mr")
            text = normalizer.normalize(text)
        except Exception:
            pass
    return text

def split_sentences(text: str):
    if not text:
        return []
    # Split on Marathi danda + common punctuation
    parts = re.split(r"(?<=[।.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def basic_tokenize(text: str):
    if not text:
        return []
    if HAS_INDIC:
        try:
            return [t for t in indic_tokenize.trivial_tokenize(text) if re.search(r"[\u0900-\u097F]", t)]
        except Exception:
            pass
    # Fallback: keep Devanagari word chunks
    return re.findall(r"[\u0900-\u097F]+", text)

def lemmatize_token(tok: str) -> str:
    return LEMMAS.get(tok, tok)

def stem_token(tok: str) -> str:
    for suf in STEM_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[:-len(suf)]
    return tok

def remove_stopwords_tokens(tokens):
    return [t for t in tokens if t not in MARATHI_STOPWORDS]

def lemmatize_tokens(tokens):
    return [lemmatize_token(t) for t in tokens]

def stem_tokens(tokens):
    return [stem_token(t) for t in tokens]

def preprocess_tokens_for_similarity(text_or_sentence: str):
    toks = basic_tokenize(text_or_sentence)
    toks = remove_stopwords_tokens(toks)
    toks = lemmatize_tokens(toks)
    toks = stem_tokens(toks)
    return toks

def make_bigrams(tokens):
    return [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]

# (Optional convenience) full pipeline returning sentences + processed tokens
def preprocess_text(text: str):
    text = normalize_text(text)
    sents = split_sentences(text)
    procd = [preprocess_tokens_for_similarity(s) for s in sents]
    return sents, procd

if __name__ == "__main__":
    sample = "पुण्यात पावसाने हजेरी लावली आहे. वातावरण आनंददायी आहे आणि लोकांनी फिरण्यासाठी बाहेर पडले."
    sents, toks = preprocess_text(sample)
    for i, (s, t) in enumerate(zip(sents, toks), 1):
        print(f"{i}. {s}\n   -> {t}")
