"""Native descriptor screening, with no invented likelihood interfaces."""
from __future__ import annotations
import contextlib
import io
import warnings
import numpy as np
from scipy import sparse


class UnusableAnchor(ValueError):
    pass


def topics_from_weights(weights, vocabulary, depth=10):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (10, len(vocabulary)) or not np.isfinite(weights).all() or (weights < 0).any():
        raise UnusableAnchor("Anchor does not expose ten finite nonnegative descriptor rows.")
    topics = []
    for row in weights:
        ordered = np.lexsort((np.arange(len(row)), -row))[:depth]
        if len(ordered) != depth or (row[ordered] <= 0).any():
            raise UnusableAnchor("An anchor topic has fewer than ten positive-weight words; no padding is allowed.")
        topics.append([vocabulary[int(i)] for i in ordered])
    return topics


def fit_anchor(method, counts, vocabulary, seed, profile, embeddings=None):
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    if counts.shape[1] != len(vocabulary) or len(vocabulary) < 100 or (counts.sum(axis=1).A1 == 0).any():
        raise ValueError("Anchor fitting view must use nonempty fitting rows and the common vocabulary.")
    details, caught = {"method": method, "profile": profile, "seed": int(seed), "fit_documents": counts.shape[0],
                       "vocabulary": len(vocabulary), "prediction_interface": "not_evaluated_in_step02"}, []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "lda":
            from gensim.models import LdaModel
            corpus = [[(int(i), float(v)) for i, v in zip(counts[r].indices, counts[r].data)] for r in range(counts.shape[0])]
            model = LdaModel(corpus=corpus, id2word=dict(enumerate(vocabulary)), num_topics=10,
                             random_state=int(seed), alpha=profile["alpha"], eta=profile["eta"],
                             passes=profile["passes"], iterations=profile["iterations"], minimum_probability=0.0)
            weights = model.get_topics()
        elif method == "nmf":
            from sklearn.decomposition import NMF
            from sklearn.feature_extraction.text import TfidfTransformer
            transformed = TfidfTransformer().fit_transform(counts) if profile["input"] == "tfidf" else counts
            model = NMF(n_components=10, init=profile["init"], solver=profile["solver"], beta_loss=profile["beta_loss"],
                        max_iter=profile["max_iter"], tol=profile["tol"], random_state=int(seed), alpha_W=0, alpha_H=0, l1_ratio=0)
            model.fit_transform(transformed)
            weights = model.components_
            details["iterations"] = int(model.n_iter_)
            details["reconstruction_error"] = float(model.reconstruction_err_)
        elif method == "gsdmm":
            # Pinned upstream implementation explicitly takes unique token sets.
            # Passing counts here would silently change its documented algorithm.
            from ...vendor.gsdmm_mgp import MovieGroupProcess
            docs = [[vocabulary[int(i)] for i in counts[r].indices] for r in range(counts.shape[0])]
            random_state = np.random.get_state()
            try:
                np.random.seed(int(seed))
                model = MovieGroupProcess(K=10, alpha=profile["alpha"], beta=profile["beta"], n_iters=profile["iterations"])
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    model.fit(docs, len(vocabulary))
            finally:
                np.random.set_state(random_state)
            weights = np.zeros((10, len(vocabulary)))
            word_to_id = {w: i for i, w in enumerate(vocabulary)}
            for k, row in enumerate(model.cluster_word_distribution):
                for word, count in row.items():
                    weights[k, word_to_id[word]] = count + profile["beta"]
            details["occupied_clusters"] = int(sum(v > 0 for v in model.cluster_doc_count))
            details["upstream_log"] = captured.getvalue()
        elif method == "bertopic":
            from bertopic import BERTopic
            from bertopic.vectorizers import ClassTfidfTransformer
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
            if embeddings is None or embeddings.shape[0] != counts.shape[0] or not np.isfinite(embeddings).all():
                raise ValueError("BERTopic requires pinned, fitting-only document embeddings.")
            # Embeddings see full cleaned fitting text; the vectorizer sees the
            # identical lexical count view used by every other model.
            docs = [" ".join(vocabulary[int(i)] for i, v in zip(counts[r].indices, counts[r].data) for _ in range(int(v)))
                    for r in range(counts.shape[0])]
            reducer = UMAP(n_neighbors=profile["neighbors"], n_components=profile["components"],
                           min_dist=profile["min_dist"], metric=profile["metric"], random_state=int(seed), n_jobs=1)
            if profile["cluster"] == "kmeans":
                cluster = KMeans(n_clusters=10, random_state=int(seed), n_init=profile["n_init"])
            else:
                from hdbscan import HDBSCAN
                cluster = HDBSCAN(min_cluster_size=profile["min_cluster_size"], metric="euclidean",
                                  cluster_selection_method="eom", prediction_data=True)
            vectorizer = CountVectorizer(vocabulary={w: i for i, w in enumerate(vocabulary)},
                                         tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False)
            # BERTopic's English preprocessor removes Chinese and underscores.
            # The supplied common lexical vocabulary must survive unchanged.
            model = BERTopic(language="multilingual", embedding_model=None, umap_model=reducer, hdbscan_model=cluster,
                             vectorizer_model=vectorizer,
                             ctfidf_model=ClassTfidfTransformer(bm25_weighting=profile["ctfidf_bm25"],
                                                               reduce_frequent_words=profile["reduce_frequent_words"]),
                             top_n_words=10, nr_topics=None, calculate_probabilities=False, verbose=False)
            labels, _ = model.fit_transform(docs, embeddings=embeddings)
            details["lexical_preprocessing"] = "multilingual_preserves_chinese_and_phrase_underscores"
            occupied = set(labels) - {-1}
            if profile["cluster"] == "hdbscan" and len(occupied) > 10:
                model.reduce_topics(docs, nr_topics=10 + int(-1 in set(labels)))
                labels = model.topics_
                occupied = set(labels) - {-1}
            outlier = int(-1 in set(labels))
            details["outlier_fit_fraction"] = float(sum(label == -1 for label in labels) / len(labels))
            if sorted(occupied) != list(range(10)) or model.c_tf_idf_.shape != (10 + outlier, len(vocabulary)):
                raise UnusableAnchor("BERTopic did not return the required ten native topics.")
            weights = model.c_tf_idf_[outlier:].toarray()
        else:
            raise ValueError("Unknown screening anchor: " + method)
    details["warnings"] = [{"type": type(w.message).__name__, "message": str(w.message)} for w in caught]
    return topics_from_weights(weights, vocabulary), np.asarray(weights), details
