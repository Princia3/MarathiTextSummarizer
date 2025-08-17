import re
import unicodedata

# Try to use Indic NLP library (optional)
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  
    from indicnlp.tokenize import indic_tokenize
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

# Marathi stopwords
MARATHI_STOPWORDS = {
    "आहे","होते","होता","होती","नाही","आणि","की","या","त्या","हे","ते","तो","ती",
    "किंवा","पर्यंत","पण","मात्र","कारण","ज्यामुळे","वरील","खाली","मध्ये","वर",
    "साठी","अधिक","काही","सुद्धा","होणार","असे","अशी","असेल","असतील","आला","आली",
    "आले","गेला","गेली","गेले","झाला","झाली","झाले","फक्त","मग","तर","आत","बाहेर",
    "पुढे","मागे","आज","उद्या","काल","इथे","तिथे","कोण","काय","कुठे","कधी","का",
    "कसा","कशी","कसे","कोणता","कोणती","कोणते","सर्व","प्रत्येक","नंतर","आधी"
}

# Lemma dictionary (inflected -> base form)
LEMMAS = {
    "आहे":"असणे","आहेत":"असणे","होतो":"असणे","होते":"असणे","होता":"असणे","होती":"असणे","होईल":"असणे",
    "झाला":"होणे","झाली":"होणे","झाले":"होणे",
    "करतो":"करणे","करते":"करणे","केले":"करणे","केली":"करणे","करणार":"करणे",
    "गेला":"जाणे","गेली":"जाणे","गेले":"जाणे","जातो":"जाणे","जाते":"जाणे","जाणार":"जाणे",
    "आला":"येणे","आली":"येणे","आले":"येणे","येतो":"येणे","येते":"येणे","येणार":"येणे",
    "दिला":"देणे","दिली":"देणे","दिले":"देणे","देतो":"देणे","देते":"देणे",
    "घेतला":"घेणे","घेतली":"घेणे","घेतले":"घेणे","घेतो":"घेणे","घेते":"घेणे"
}

# Suffixes for stemming
STEM_SUFFIXES = sorted([
    "ांमध्ये","ांकडून","ांबरोबर","ांपासून","ांच्या","ींनी","ांना","ांचे","ांची","ांचा",
    "मध्ये","पासून","कडे","वरून","हून","णे","तो","ते","ती","ला","ने","चे","ची","चा","त","वर"
], key=len, reverse=True)


# ---------- Preprocessing Functions ----------

def normalize_text(text: str) -> str:
    """Normalize Unicode & clean extra spaces"""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if HAS_INDIC:  # Use Indic NLP normalizer if available
        normalizer = IndicNormalizerFactory().get_normalizer("mr")
        text = normalizer.normalize(text)
    return text

def split_sentences(text: str):
    """Split Marathi text into sentences"""
    return [p.strip() for p in re.split(r"(?<=[।.!?])\s+", text) if p.strip()]

def tokenize(text: str):
    """Tokenize text into words"""
    if HAS_INDIC:
        return [t for t in indic_tokenize.trivial_tokenize(text) if re.search(r"[\u0900-\u097F]", t)]
    return re.findall(r"[\u0900-\u097F]+", text)

def lemmatize(tok: str) -> str:
    return LEMMAS.get(tok, tok)

def stem(tok: str) -> str:
    for suf in STEM_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            return tok[:-len(suf)]
    return tok

def preprocess_tokens(text: str):
    """Tokenize -> remove stopwords -> lemmatize -> stem"""
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in MARATHI_STOPWORDS]
    tokens = [lemmatize(t) for t in tokens]
    tokens = [stem(t) for t in tokens]
    return tokens

def preprocess_text(text: str):
    """Full pipeline: normalize, split sentences, preprocess each"""
    text = normalize_text(text)
    sentences = split_sentences(text)
    processed = [preprocess_tokens(s) for s in sentences]
    return sentences, processed
