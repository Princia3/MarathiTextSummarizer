# src_extractive.py
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src_preprocess import (
    normalize_text,
    split_sentences,
    basic_tokenize,
    remove_stopwords_tokens,
)

def summarize_extractive(text: str, num_sentences: int = 3) -> str:
    text = normalize_text(text)
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return text

    # Custom tokenizer: tokenize → remove stopwords
    def marathi_tokenizer(sentence):
        tokens = basic_tokenize(sentence)
        return remove_stopwords_tokens(tokens)

    vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=marathi_tokenizer,
    lowercase=False,
    token_pattern=None  
)

    tfidf = vectorizer.fit_transform(sentences)

    scores = tfidf.sum(axis=1).A1
    top_idx = np.argsort(scores)[::-1][:num_sentences]
    top_idx_sorted = sorted(top_idx)

    return " ".join([sentences[i] for i in top_idx_sorted])

if __name__ == "__main__":
    sample_text = (
        "पुण्यातील हवामानात अचानक बदल झाला आहे। "
        "हलका पाऊस आणि थंड वारा जाणवत आहे। "
        "शहरातील वाहतुकीवर त्याचा परिणाम दिसून येतो। "
        "तज्ज्ञांच्या मते पुढील दोन दिवस ढगाळ वातावरण राहू शकते। "
        "नागरिकांना आवश्यक ती काळजी घेण्याचा सल्ला देण्यात आला आहे!"
    )
    print(summarize_extractive(sample_text, num_sentences=2))
