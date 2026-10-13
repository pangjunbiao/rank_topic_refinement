"""Library-exact window evidence and independent duplicate-group support.

Counts deliberately use the pinned Gensim accumulator, including its repeated
token convention. Independent support uses actual sets in each lexical window.
Out-of-vocabulary positions are retained; documents are never concatenated.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import numpy as np
from scipy import sparse


def virtual_windows(tokens, width):
    if width < 1:
        raise ValueError("Window width must be positive.")
    if len(tokens) <= width:
        yield tokens
    else:
        for start in range(len(tokens) - width + 1):
            yield tokens[start:start + width]


def count_group_evidence(texts, words, width=10):
    from gensim.corpora import Dictionary
    from gensim.topic_coherence.text_analysis import WordOccurrenceAccumulator
    words = list(words)
    if not words or len(set(words)) != len(words):
        raise ValueError("Evidence words must be nonempty and unique.")
    dictionary = Dictionary([words])
    relevant = {dictionary.token2id[w] for w in words}
    mapping = {w: i for i, w in enumerate(words)}
    word_rows, pair_rows, true_pairs, df_rows, windows = [], [], [], [], []
    for text in texts:
        acc = WordOccurrenceAccumulator(relevant, dictionary).accumulate([text], width)
        order = [acc.id2contiguous[dictionary.token2id[w]] for w in words]
        word_rows.append(acc._occurrences[order].astype(np.int64))
        matrix = acc._co_occurrences.tocsr()[order][:, order]
        upper = sparse.triu(matrix, k=1).tocoo()
        pair_rows.append({(int(i), int(j)): int(v) for i, j, v in zip(upper.row, upper.col, upper.data)})
        pairs = set()
        for window in virtual_windows(text, width):
            ids = sorted({mapping[w] for w in window if w in mapping})
            pairs.update(combinations(ids, 2))
        true_pairs.append(pairs)
        df_rows.append(sorted({mapping[w] for w in text if w in mapping}))
        windows.append(int(acc.num_docs))
    # Every pair is retained, including those with zero observed co-occurrence.
    universe = list(combinations(range(len(words)), 2))
    pair_index = {pair: i for i, pair in enumerate(universe)}
    def pair_sparse(rows):
        rr, cc, vv = [], [], []
        for r, entries in enumerate(rows):
            for p in entries:
                rr.append(r); cc.append(pair_index[p]); vv.append(entries[p] if isinstance(entries, dict) else 1)
        return sparse.csr_matrix((np.asarray(vv, dtype=np.int64), (rr, cc)), shape=(len(texts), len(universe)))
    rr, cc = [], []
    for r, ids in enumerate(df_rows):
        rr.extend([r] * len(ids)); cc.extend(ids)
    word_support = sparse.csr_matrix((np.ones(len(rr), dtype=np.int64), (rr, cc)), shape=(len(texts), len(words)))
    return {"words": words, "word_windows": np.asarray(word_rows, dtype=np.int64),
            "pair_windows": pair_sparse(pair_rows), "pair_groups": pair_sparse(true_pairs),
            "word_groups": word_support, "pairs": np.asarray(universe, dtype=np.int64).reshape(-1, 2),
            "windows_per_group": np.asarray(windows, dtype=np.int64)}
