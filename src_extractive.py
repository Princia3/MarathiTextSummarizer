# src_extractive.py
import numpy as np
import networkx as nx
from src_preprocess import normalize_text, split_sentences, basic_tokenize, remove_stopwords_tokens

def sentence_similarity(sent1_tokens, sent2_tokens):
    # Jaccard similarity
    set1 = set(sent1_tokens)
    set2 = set(sent2_tokens)
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def summarize_extractive(text: str, num_sentences: int = 3) -> str:
    text = normalize_text(text)
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return text

    # Tokenize each sentence
    tokenized = [remove_stopwords_tokens(basic_tokenize(s)) for s in sentences]

    # Build similarity matrix
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(tokenized[i], tokenized[j])

    # Build graph and apply PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences and pick top N
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])
    return summary

if __name__ == "__main__":
    sample_text = input("Enter any Marathi text to summarize:\n")
    n_sent = int(input("Enter number of sentences for summary: "))
    summary = summarize_extractive(sample_text, num_sentences=n_sent)
    print("\n=== SUMMARY ===\n")
    print(summary)
