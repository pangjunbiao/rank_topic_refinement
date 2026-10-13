"""Observed-only classical completion; fitting-only IDF and LDA state.

NMF's normalized nonnegative reconstruction is a predictive cross-entropy
interface, not a claim of a native generative likelihood.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from .adapters.pilot_anchors import topics_from_weights
from .neural_prediction import backoff
from ..evaluation.likelihood import aggregate_nll


def normalized_reconstruction(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Prediction weights must be finite, nonnegative rows.")
    totals = weights.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Zero prediction row; no fabricated latent or prior fallback.")
    return weights / totals


def select_probability_backoff(probabilities, prior, grid, records, masks, sources):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape != (len(records), len(prior)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probability shape or entries.")
    if not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("Completion probabilities do not sum to one.")
    with np.errstate(divide="ignore"):
        logp = np.log(p)
    candidates, curve = [], []
    for rho in grid:
        mixed = backoff(logp, prior, float(rho))
        # Zero model probability remains a real failure at rho=0. A positive
        # shared fitting-prior mixture is explicit, selected and reported.
        if np.isnan(mixed).any() or np.isposinf(mixed).any() or not np.allclose(logsumexp(mixed,axis=1),0.,rtol=0,atol=1e-10):
            raise ValueError("Invalid mixed log probabilities.")
        losses, finite_targets = [], True
        for i, record in enumerate(records):
            mask = masks[record["group_id"]]
            targets = np.asarray(mask["target_ids"])
            if not mask["eligible"] or targets.ndim != 1 or targets.size == 0 or targets.dtype.kind not in "iu" or targets.min() < 0 or targets.max() >= mixed.shape[1]:
                raise ValueError("Invalid accepted completion targets.")
            selected = mixed[i, targets]
            if not np.isfinite(selected).all():
                finite_targets = False
                break
            losses.append({"group_id": record["group_id"], "source": record["source"],
                "negative_log_probability": float(-selected.sum()), "target_tokens": int(targets.size)})
        if not finite_targets:
            curve.append({"rho": float(rho), "status": "zero_probability_not_clipped",
                          "macro_source_nll": None, "micro_nll": None})
            continue
        scores = aggregate_nll(losses, sources)
        candidates.append({"rho": float(rho), "scores": scores, "rows": losses})
        curve.append({"rho": float(rho), "status": "scored", **scores})
    if not candidates:
        raise ValueError("No finite explicitly mixed predictive completion path.")
    return min(candidates, key=lambda r: (r["scores"]["macro_source_nll"], r["rho"])), curve


def fit_and_predict(method, fit_counts, observed_counts, vocabulary, seed, profile):
    fit = sparse.csr_matrix(fit_counts, dtype=np.float64)
    observed = sparse.csr_matrix(observed_counts, dtype=np.float64)
    if method not in ["lda", "nmf"] or fit.shape[1] != len(vocabulary) or observed.shape[1] != len(vocabulary):
        raise ValueError("Invalid classical interface or vocabulary.")
    if len(vocabulary) < 100 or (fit.sum(axis=1).A1 <= 0).any() or (observed.sum(axis=1).A1 <= 0).any():
        raise ValueError("Classical fitting and observed documents must be nonempty.")
    if (fit.data < 0).any() or (observed.data < 0).any() or not np.isfinite(fit.data).all() or not np.isfinite(observed.data).all():
        raise ValueError("Invalid counts.")
    details = {"method": method, "seed": int(seed), "profile": profile,
               "fit_documents": fit.shape[0], "observed_documents": observed.shape[0],
               "vocabulary": len(vocabulary), "development_full_text_used_for_inference": False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "lda":
            from gensim.models import LdaModel
            def corpus(matrix):
                return [[(int(i), float(v)) for i, v in zip(matrix[r].indices, matrix[r].data)]
                        for r in range(matrix.shape[0])]
            model = LdaModel(corpus=corpus(fit), id2word=dict(enumerate(vocabulary)), num_topics=10,
                random_state=int(seed), alpha=profile["alpha"], eta=profile["eta"],
                passes=profile["passes"], iterations=profile["iterations"], minimum_probability=0.)
            weights = model.get_topics().astype(np.float64)
            gamma, _ = model.inference(corpus(observed), collect_sstats=False)
            latent = normalized_reconstruction(gamma)
            probabilities = normalized_reconstruction(latent @ normalized_reconstruction(weights))
            details["prediction_interface"] = "lda_observed_posterior_predictive"
        else:
            from sklearn.decomposition import NMF
            from sklearn.feature_extraction.text import TfidfTransformer
            if profile["input"] != "tfidf":
                raise ValueError("This declared NMF completion uses fitting-only TF-IDF.")
            transformer = TfidfTransformer()
            transformed = transformer.fit_transform(fit)
            observed_transformed = transformer.transform(observed)
            model = NMF(n_components=10, init=profile["init"], solver=profile["solver"],
                beta_loss=profile["beta_loss"], max_iter=profile["max_iter"], tol=profile["tol"],
                random_state=int(seed), alpha_W=0, alpha_H=0, l1_ratio=0.)
            model.fit_transform(transformed)
            weights = model.components_.astype(np.float64)
            latent = model.transform(observed_transformed)
            probabilities = normalized_reconstruction(latent @ weights)
            details.update(prediction_interface="nmf_normalized_tfidf_reconstruction_cross_entropy",
                fitting_only_idf=transformer.idf_.tolist(), iterations=int(model.n_iter_),
                reconstruction_error=float(model.reconstruction_err_))
    details["warnings"] = [{"type": type(w.message).__name__, "message": str(w.message)} for w in caught]
    return topics_from_weights(weights, vocabulary), weights, probabilities, details
