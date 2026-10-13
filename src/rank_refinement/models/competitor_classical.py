"""Faithful native GSDMM and BERTopic interfaces with explicit availability.

GSDMM's pinned unique-word-set Gibbs update is evaluated in log space to avoid
underflow. Empty topics are never padded with prior words for coherence.
BERTopic c-TF-IDF is a descriptor score, not a word likelihood; its NLL is NA.
"""
from __future__ import annotations
import contextlib
import io
import types
import warnings
import numpy as np
from scipy.special import gammaln, logsumexp
from .competitor_neural import normalized_probabilities


def native_topics(weights, vocabulary, topic_ids=None):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(vocabulary) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid native descriptor matrix.")
    ids = list(range(len(weights))) if topic_ids is None else list(topic_ids)
    topics = []
    failures = []
    for k in ids:
        positions = np.lexsort((np.arange(len(vocabulary)), -weights[k]))[:10]
        if len(positions) != 10 or (weights[k, positions] <= 0).any():
            failures.append(int(k))
        else:
            topics.append([vocabulary[int(i)] for i in positions])
    info = {"native_topic_ids": [int(k) for k in ids], "native_topic_count": len(ids),
            "fixed_k10": len(ids) == 10, "all_native_topic_slots_retained": not failures,
            "topics_without_ten_supported_words": failures}
    return (topics if ids and not failures else None), info


def stable_gsdmm_score(model, doc):
    beta, v, k = model.beta, model.vocab_size, model.K
    length = len(doc)
    lp = np.empty(k, dtype=np.float64)
    for cluster in range(k):
        total = model.cluster_word_count[cluster] + v * beta
        lp[cluster] = (np.log(model.cluster_doc_count[cluster] + model.alpha)
            + sum(np.log(model.cluster_word_distribution[cluster].get(word, 0) + beta) for word in doc)
            + gammaln(total) - gammaln(total + length))
    return np.exp(lp - logsumexp(lp)).tolist()


def gsdmm_predict(model, observed, vocabulary):
    word_index = {w: i for i, w in enumerate(vocabulary)}
    counts = np.zeros((model.K, len(vocabulary)), dtype=np.float64)
    for k, row in enumerate(model.cluster_word_distribution):
        for word, n in row.items():
            counts[k, word_index[word]] = n
    probabilities = np.empty_like(observed, dtype=np.float64)
    posterior = np.empty((len(observed), model.K), dtype=np.float64)
    for i, row in enumerate(observed):
        positions = np.flatnonzero(row > 0)
        doc = [vocabulary[int(j)] for j in positions]
        mixture = np.asarray(stable_gsdmm_score(model, doc))
        # Unique observed evidence is consistent with the pinned fitting model.
        posterior_beta = counts + model.beta
        posterior_beta[:, positions] += 1
        posterior_beta /= posterior_beta.sum(1, keepdims=True)
        probabilities[i] = mixture @ posterior_beta
        posterior[i] = mixture
    return counts, normalized_probabilities(probabilities), posterior


def fit_gsdmm(fit, observed, vocabulary, seed, profile, log):
    from ..vendor.gsdmm_mgp import MovieGroupProcess
    docs = [[vocabulary[int(i)] for i in np.flatnonzero(row)] for row in fit]
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        model = MovieGroupProcess(K=10, alpha=profile["alpha"], beta=profile["beta"], n_iters=profile["iterations"])
        model.score = types.MethodType(stable_gsdmm_score, model)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            labels = model.fit(docs, len(vocabulary))
    finally:
        np.random.set_state(state)
    counts, probabilities, theta = gsdmm_predict(model, observed, vocabulary)
    occupied = np.flatnonzero(np.asarray(model.cluster_doc_count) > 0).tolist()
    topics, info = native_topics(counts, vocabulary, occupied)
    # Save every occupied topic. If K<10 these genuine results appear only in
    # the variable-K supplement, while fixed-K panel cells remain NA.
    details = {**info, "prediction_interface": "gsdmm_unique_observed_dirichlet_multinomial_posterior_predictive",
        "training_input": "unique_word_sets_as_pinned_upstream", "formula3_log_domain_underflow_repair": True,
        "cluster_doc_count": model.cluster_doc_count, "cluster_word_count": model.cluster_word_count,
        "iterations_requested": profile["iterations"], "fitting_labels": [int(i) for i in labels],
        "upstream_log": captured.getvalue(), "observed_only_inference": True,
        "development_full_text_used_for_inference": False, "labels_used": False, "device": "cpu"}
    log.info("GSDMM FIT | occupied=%d/10 | fixed-K descriptors=%s | native prediction available", len(occupied), info["fixed_k10"])
    return topics, counts, probabilities, theta, details


def lexical_docs(counts, vocabulary):
    return [" ".join(vocabulary[int(i)] for i in np.flatnonzero(row) for _ in range(int(row[i]))) for row in counts]


def fit_bertopic(fit, observed, vocabulary, seed, profile, vectors, log):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    if len(vectors["fit_full"]) != len(fit) or len(vectors["development_observed"]) != len(observed):
        raise ValueError("BERTopic embeddings do not align with canonical rows.")
    reducer = UMAP(n_neighbors=profile["neighbors"], n_components=profile["components"],
        min_dist=profile["min_dist"], metric=profile["metric"], random_state=int(seed), n_jobs=1,
        transform_seed=int(seed))
    if profile["cluster"] == "kmeans":
        cluster = KMeans(n_clusters=10, n_init=profile["n_init"], random_state=int(seed))
    else:
        from hdbscan import HDBSCAN
        cluster = HDBSCAN(min_cluster_size=profile["min_cluster_size"], metric="euclidean",
                          cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(vocabulary={w: i for i, w in enumerate(vocabulary)}, tokenizer=str.split,
                                preprocessor=None, token_pattern=None, lowercase=False)
    model = BERTopic(language="multilingual", embedding_model=None, umap_model=reducer, hdbscan_model=cluster,
        vectorizer_model=vectorizer, ctfidf_model=ClassTfidfTransformer(bm25_weighting=profile["ctfidf_bm25"],
        reduce_frequent_words=profile["reduce_frequent_words"]), top_n_words=10, nr_topics=None,
        calculate_probabilities=False, verbose=False)
    docs = lexical_docs(fit, vocabulary)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels, _ = model.fit_transform(docs, embeddings=vectors["fit_full"])
        occupied = set(labels) - {-1}
        if profile["cluster"] == "hdbscan" and len(occupied) > 10:
            model.reduce_topics(docs, nr_topics=10 + int(-1 in set(labels)))
            labels = model.topics_
            occupied = set(labels) - {-1}
        offset = int(-1 in set(labels))
        expected = sorted(occupied)
        if expected != list(range(len(expected))) or model.c_tf_idf_.shape != (len(expected) + offset, len(vocabulary)):
            raise ValueError("Unexpected native BERTopic c-TF-IDF topic alignment.")
        weights = model.c_tf_idf_[offset:].toarray()
        topics, info = native_topics(weights, vocabulary)
        # Assignment is a native clustering diagnostic, not a word likelihood.
        assigned, _ = model.transform(lexical_docs(observed, vocabulary), embeddings=vectors["development_observed"])
    details = {**info, "prediction_interface": "unavailable_native_bertopic_word_likelihood",
        "prediction_unavailable_reason": "c_tf_idf_scores_and_cluster_membership_are_not_word_probabilities",
        "fitting_labels": [int(i) for i in labels], "observed_development_labels": [int(i) for i in assigned],
        "outlier_fit_fraction": float(np.mean(np.asarray(labels) == -1)),
        "lexical_preprocessing": "shared_fixed_vocabulary_multilingual_preserves_chinese_and_phrase_underscores",
        "observed_only_inference": True, "development_full_text_used_for_inference": False,
        "labels_used": False, "device": "cpu", "warnings": sorted({str(w.message) for w in caught})}
    log.info("BERTOPIC FIT | native topics=%d | outlier fraction=%.3f | word likelihood=NA", len(expected), details["outlier_fit_fraction"])
    return topics, weights, None, None, details
