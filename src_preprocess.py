# src_preprocess.py
import re
import unicodedata

# Marathi stopwords list (~70 common ones)
MARATHI_STOPWORDS = {
    "आहे", "होते", "होता", "होती", "नाही", "आणि", "की", "या", "त्या", "हे", "ते",
    "तो", "ती", "किंवा", "पर्यंत", "पण", "मात्र", "कारण", "ज्यामुळे", "वरील",
    "खाली", "मध्ये", "वर", "साठी", "अधिक", "काही", "सुद्धा", "होणार", "असे",
    "अशी", "असेल", "असतील", "आला", "आली", "आले", "गेला", "गेली", "गेले", "झाला",
    "झाली", "झाले", "होऊन", "होऊनही", "फक्त", "मग", "तर", "आत", "बाहेर", "पुढे",
    "मागे", "आज", "उद्या", "काल", "इथे", "तिथे", "कोण", "काय", "कुठे", "कधी",
    "का", "कसा", "कशी", "कसे", "कोणता", "कोणती", "कोणते", "सर्व", "प्रत्येक",
    "कुठलाही", "नंतर", "आधी", "आतापर्यंत", "अजून", "अजूनही", "तरी", "जरी",
    "जसे", "तसे"
}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text: str):
    parts = re.split(r'(?<=[।.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]

def remove_stopwords_tokens(tokens):
    return [t for t in tokens if t not in MARATHI_STOPWORDS]

def basic_tokenize(text: str):
    # Match full Marathi words (Devanagari range)
    tokens = re.findall(r'[\u0900-\u097F]+', text)
    return tokens

if __name__ == "__main__":
    sample_text = "पुण्यात पावसाने हजेरी लावली आहे. वातावरण आनंददायी आहे आणि लोकांनी फिरण्यासाठी बाहेर पडले."
    text = normalize_text(sample_text)
    sentences = split_sentences(text)
    for i, sent in enumerate(sentences, 1):
        tokens = basic_tokenize(sent)
        tokens_wo_sw = remove_stopwords_tokens(tokens)
        print(f"Sentence {i} tokens (no stopwords): {tokens_wo_sw}")
