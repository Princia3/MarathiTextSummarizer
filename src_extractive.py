# src_extractive.py
import math
import numpy as np
import networkx as nx
from collections import Counter

from src_preprocess import (
    normalize_text, split_sentences,
    preprocess_tokens_for_similarity, make_bigrams
)

# ---------- Similarities ----------
def jaccard_set(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def jaccard_bigrams(tokens1, tokens2):
    big1 = set(make_bigrams(tokens1))
    big2 = set(make_bigrams(tokens2))
    if not big1 or not big2:
        return 0.0
    return len(big1 & big2) / len(big1 | big2)

def compute_idf(list_of_token_lists):
    n = len(list_of_token_lists)
    df = Counter()
    for toks in list_of_token_lists:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((1 + n) / (1 + df_t)) + 1.0 for t, df_t in df.items()}  # smoothed

def weighted_jaccard_tf_idf(tokens1, tokens2, idf):
    tf1, tf2 = Counter(tokens1), Counter(tokens2)
    w1 = {t: tf1[t] * idf.get(t, 1.0) for t in tf1}
    w2 = {t: tf2[t] * idf.get(t, 1.0) for t in tf2}
    keys = set(w1) | set(w2)
    if not keys:
        return 0.0
    inter = sum(min(w1.get(k, 0.0), w2.get(k, 0.0)) for k in keys)
    union = sum(max(w1.get(k, 0.0), w2.get(k, 0.0)) for k in keys)
    return 0.0 if union == 0.0 else inter / union

def sentence_similarity(tokens_i, tokens_j, *, mode="weighted_jaccard", idf=None):
    if mode == "jaccard":
        return jaccard_set(tokens_i, tokens_j)
    if mode == "jaccard_bigrams":
        return jaccard_bigrams(tokens_i, tokens_j)
    # default & best: weighted_jaccard
    if idf is None:
        idf = {}
    return weighted_jaccard_tf_idf(tokens_i, tokens_j, idf)

# ---------- Compressed one-line summary (still extractive) ----------
def compress_summary(tokenized, top_indices, idf, max_tokens=18):
    # pool words from top sentences
    bag = [tok for i in top_indices for tok in tokenized[i]]
    tf = Counter(bag)
    weights = {t: tf[t] * idf.get(t, 1.0) for t in tf}
    # keep distinct, high-weight tokens in original order of appearance
    seen = set()
    ordered = []
    for i in top_indices:
        for tok in tokenized[i]:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append((weights.get(tok, 0.0), tok))
    # sort by weight desc but keep stability by small tie-break (index)
    ordered.sort(key=lambda x: -x[0])
    kept = [tok for _, tok in ordered[:max_tokens]]
    # join as a short phrase sentence (Marathi-friendly punctuations)
    if not kept:
        return ""
    # Light heuristics to add commas/‘आणि’
    if len(kept) > 3:
        return " , ".join(kept[:-1]) + " आणि " + kept[-1]
    return " ".join(kept)

# ---------- Main summarizer ----------
def summarize_extractive(
    text: str,
    num_sentences: int = 2,
    similarity_mode: str = "weighted_jaccard",
    compressed: bool = False,
    compressed_max_tokens: int = 18
) -> str:
    text = normalize_text(text)
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return ""
    if n <= num_sentences and not compressed:
        return text

    tokenized = [preprocess_tokens_for_similarity(s) for s in sentences]
    idf = compute_idf(tokenized)

    # similarity graph
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim_matrix[i, j] = sentence_similarity(
                tokenized[i], tokenized[j],
                mode=similarity_mode, idf=idf
            )

    # PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # pick top sentences, keep original order for readability
    ranked = sorted(((scores[i], i) for i in range(n)), reverse=True)
    top_idx = [i for _, i in ranked[:max(num_sentences, 1)]]
    top_idx.sort()

    if compressed:
        one_liner = compress_summary(tokenized, top_idx, idf, max_tokens=compressed_max_tokens)
        # Capitalize + full stop if missing
        return (one_liner[:1].upper() + one_liner[1:] if one_liner else one_liner).rstrip(" .।") + "।"

    return " ".join([sentences[i] for i in top_idx])

if __name__ == "__main__":
    sample = (
        "सोन्याच्या दरात 13 ऑगस्ट 2025 रोजी लक्षणीय वाढ झाली असून, 10 ग्रॅम सोन्याची किंमत वाढल्यामुळे ग्राहकांमध्ये चिंता निर्माण झाली आहे. विशेषतः 24 कॅरेट सोन्याच्या भावात मोठा बदल दिसून आला आहे. तज्ज्ञांच्या मते, पुढील पाच वर्षांत सोन्याचे दर प्रति तोळ्याला ₹2,50,000 पर्यंत पोहोचू शकतात, जागतिक आर्थिक घडामोडी आणि चलनफुगवटा यामुळे ही वाढ होण्याची शक्यता आहे. टायटन कंपनीच्या दागिन्यांच्या विक्रीत 19% वाढ झाली असून, महसूल 21% वाढला आहे. सोन्याच्या दरात वाढ होण्याची मुख्य कारणे जागतिक व्यापार तणाव, महागाई, आणि केंद्रीय बँकांची खरेदी आहेत. ग्राहकांनी दागिने किंवा गुंतवणुकीसाठी खरेदी करण्यापूर्वी नवीन दरांची माहिती घेणे आवश्यक आहे."

    )
    print("=== Top-2 sentences ===")
    print(summarize_extractive(sample, num_sentences=2))

