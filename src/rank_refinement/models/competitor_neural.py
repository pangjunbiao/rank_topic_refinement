"""Resumable author-model training and observed-only native decoding.

CombinedTM uses the author's combined inference/decoder networks and summed
KL+reconstruction objective. Its prediction averages 20 decoded posterior
samples, rather than inventing a theta @ normalized-beta likelihood.
FASTopic uses full fitting batches and the author's fitting-reference theta.
GloCOM uses fitting-only cluster banks and the author's affine BN decoder.
"""
from __future__ import annotations
import hashlib
import io
import random
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from ..runtime.artifacts import atomic_bytes, digest, file_hash, read_json, write_json
from .competitor_sources import author_class
from .neural_training import batches, seed_everything


def normalized_probabilities(values, tolerance=2e-5):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Native decoded probabilities are invalid.")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any() or np.max(np.abs(totals - 1)) > tolerance:
        raise ValueError("Native decoded output is not a probability distribution; no arbitrary normalization.")
    return values / totals


def cluster_contexts(fit_counts, fit_embeddings, observed_counts, observed_embeddings, profile, seed):
    if len(fit_counts) < profile["clusters"]:
        raise ValueError("Not enough fitting groups for the declared GloCOM context count.")
    cluster = KMeans(n_clusters=profile["clusters"], n_init=profile["n_init"], random_state=int(seed))
    labels = cluster.fit_predict(fit_embeddings)
    bank = np.zeros((profile["clusters"], fit_counts.shape[1]), dtype=np.float32)
    np.add.at(bank, labels, fit_counts)
    heldout = cluster.predict(observed_embeddings)
    return {"fit": np.concatenate([fit_counts, bank[labels]], axis=1),
            "observed": np.concatenate([observed_counts, bank[heldout]], axis=1),
            "bank": bank, "centers": cluster.cluster_centers_,
            "fit_assignments": labels, "observed_assignments": heldout}


def make_model(method, folder, vocab_size, dimension, profile, topics=10, words=None, seed=809):
    seed_everything(seed)
    cls = author_class(method, folder)
    if method == "combinedtm":
        return cls(vocab_size, dimension, "combined", n_components=topics,
            model_type=profile["model_type"], hidden_sizes=tuple(profile["hidden_sizes"]),
            activation=profile["activation"], dropout=profile["dropout"],
            learn_priors=profile["learn_priors"], label_size=0)
    if method == "fastopic":
        model = cls(topics, theta_temp=profile["theta_temp"], DT_alpha=profile["DT_alpha"], TW_alpha=profile["TW_alpha"])
        model.init(vocab_size, dimension)
        return model
    if method == "glocom":
        names = ["en_units", "dropout", "beta_temp", "aug_coef", "prior_var",
                 "weight_loss_ECR", "sinkhorn_alpha", "sinkhorn_max_iter"]
        if words is None or words.shape != (vocab_size, dimension):
            raise ValueError("GloCOM requires the declared shared Chinese word initialization.")
        return cls(vocab_size=vocab_size, num_topics=topics, embed_size=dimension,
                   pretrained_WE=words, **{n: profile[n] for n in names})
    raise ValueError("Unknown neural competitor.")


def ctm_loss(model, counts, embeddings):
    prior_mu, prior_var, mu, var, logvar, word_dist, _ = model(counts, embeddings)
    kl = .5 * ((var / prior_var).sum(1) + ((prior_mu - mu).square() / prior_var).sum(1)
               - model.n_components + prior_var.log().sum() - logvar.sum(1))
    reconstruction = -(counts * (word_dist + 1e-10).log()).sum(1)
    # Author CTM._train_epoch sums across the batch; Adam betas are retained.
    return {"loss": (kl + reconstruction).sum(), "loss_KL": kl.mean(),
            "loss_reconstruction": reconstruction.mean()}


def save_training(path, model, optimizer, epoch, identity, history):
    import torch
    value = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "identity": identity, "history": history,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())
    write_json(path.with_suffix(".json"), {"identity": identity, "epoch": epoch,
                                         "checkpoint_sha256": file_hash(path)})


def restore_training(path, model, optimizer, identity, device):
    import torch
    index = read_json(path.with_suffix(".json"))
    if index["identity"] != identity or file_hash(path) != index["checkpoint_sha256"]:
        raise ValueError("Competitor checkpoint identity/checksum failed.")
    # This is a locally produced, hash-verified checkpoint, never an external pickle.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["identity"] != identity or value["epoch"] != index["epoch"]:
        raise ValueError("Competitor checkpoint payload differs from its index.")
    # The CTM author forward dynamically registers a second name for beta.
    # A fresh DecoderNetwork has topic_word_matrix=None until its first forward.
    # Restore that same alias, after verifying it is not a different parameter.
    if "topic_word_matrix" in value["model"]:
        if not hasattr(model, "beta") or not torch.equal(value["model"]["topic_word_matrix"], value["model"]["beta"]):
            raise ValueError("CTM saved topic-word alias differs from beta.")
        model.topic_word_matrix = model.beta
    model.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    torch.set_rng_state(value["torch_rng"])
    np.random.set_state(value["numpy_rng"])
    random.setstate(value["python_rng"])
    if device == "cuda" and value["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return value["epoch"], value["history"]


def train_model(root, method, profile, seed, folder, fit_counts, vectors, fit_ids,
                source_specs, device, checkpoint_every, log, contexts=None):
    import torch
    if (fit_counts.ndim != 2 or len(fit_ids) != len(fit_counts) or len(set(fit_ids)) != len(fit_ids)
            or not np.isfinite(fit_counts).all() or (fit_counts < 0).any()
            or (fit_counts.sum(1) <= 0).any()):
        raise ValueError("Invalid unique nonempty fitting rows.")
    dimension = vectors["fit_full"].shape[1]
    identity = {"method": method, "profile": profile, "seed": int(seed), "device": device,
        "fitting_group_ids": fit_ids, "fit_counts_sha256": hashlib.sha256(fit_counts.tobytes()).hexdigest(),
        "fitting_embeddings_sha256": hashlib.sha256(vectors["fit_full"].tobytes()).hexdigest(),
        "word_embeddings_sha256": hashlib.sha256(vectors["words"].tobytes()).hexdigest() if method == "glocom" else None,
        "dimension": dimension, "source_specs": [s for s in source_specs if s["method"] == method],
        "implementation_sha256": file_hash(__file__), "torch": torch.__version__, "topics": 10,
        "context_bank_sha256": hashlib.sha256(contexts["bank"].tobytes()).hexdigest() if contexts else None}
    cache = Path(root) / "data/cache/step04_training" / digest(identity)[:20]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "checkpoint.pt"
    model = make_model(method, folder, fit_counts.shape[1], dimension, profile,
                       words=vectors["words"] if method == "glocom" else None, seed=seed).to(device)
    betas = (profile["momentum"], .99) if method == "combinedtm" else (.9, .999)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile["learning_rate"], betas=betas)
    completed, history = (restore_training(path, model, optimizer, identity, device) if path.is_file() else (0, []))
    if completed > profile["epochs"]:
        raise ValueError("Checkpoint exceeds the frozen training budget.")
    if not completed:
        log.new_neural_fits = getattr(log, "new_neural_fits", 0) + 1
    log.info("COMPETITOR TRAIN | %s %s seed=%d | epoch=%d/%d | %s",
        method, profile["name"], seed, completed, profile["epochs"], device)
    counts = torch.from_numpy(contexts["fit"] if method == "glocom" else fit_counts).to(device)
    embeddings = torch.from_numpy(vectors["fit_full"]).to(device)
    started = time.perf_counter()
    for epoch in range(completed + 1, profile["epochs"] + 1):
        model.train()
        totals, seen = {}, 0
        chunks = [np.arange(len(fit_counts))] if method == "fastopic" else batches(len(fit_counts), profile["batch_size"], seed * 100000 + epoch)
        for ids in chunks:
            index = torch.as_tensor(ids, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if method == "combinedtm":
                losses = ctm_loss(model, counts[index], embeddings[index])
            elif method == "fastopic":
                losses = model(counts[index], embeddings[index])
            else:
                losses = model(counts[index])
            scalars = {k: float(v.detach().cpu() if torch.is_tensor(v) else v) for k, v in losses.items()}
            if not all(np.isfinite(v) for v in scalars.values()):
                raise FloatingPointError(f"Nonfinite author-model loss: {method}/{profile['name']} seed={seed} epoch={epoch}.")
            losses["loss"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite author-model gradient: {method} seed={seed} epoch={epoch}.")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite native model parameter.")
            for key, scalar in scalars.items():
                # CTM's primary scalar is a batch sum; its other losses are means.
                totals[key] = totals.get(key, 0.) + scalar * (1 if method == "combinedtm" and key == "loss" else len(ids))
            seen += len(ids)
        if seen != len(fit_counts):
            raise AssertionError("An epoch omitted or duplicated fitting groups.")
        history.append({"epoch": epoch, "fit_documents_seen": seen, **{k: v / seen for k, v in totals.items()}})
        if epoch == 1 or epoch % checkpoint_every == 0 or epoch == profile["epochs"]:
            save_training(path, model, optimizer, epoch, identity, history)
            log.info("COMPETITOR EPOCH | %s %s seed=%d | %d/%d | loss=%.6f | elapsed=%.1f sec | checkpoint saved",
                method, profile["name"], seed, epoch, profile["epochs"], history[-1]["loss"], time.perf_counter() - started)
    model.eval()
    return model, {"training_identity": identity, "history": history, "epochs": profile["epochs"],
        "checkpoint_relative_path": path.relative_to(root).as_posix(), "checkpoint_sha256": file_hash(path),
        "device": device, "author_objective_preserved": True, "labels_used": False,
        "batch_tail_policy": "merge_singleton_keep_all_fitting_groups" if method != "fastopic" else "full_fitting_batch",
        "development_used_for_training_or_epoch_selection": False}


def decode_ctm(model, observed_counts, observed_embeddings, seed, samples=20):
    import torch
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 470004)
    with torch.inference_mode():
        counts = torch.from_numpy(observed_counts).to(device)
        embeddings = torch.from_numpy(observed_embeddings).to(device)
        mu, logvar = model.get_posterior(counts, embeddings)
        probabilities = np.zeros(observed_counts.shape, dtype=np.float64)
        theta_mean = np.zeros((len(observed_counts), model.n_components), dtype=np.float64)
        for _ in range(samples):
            noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype).to(device)
            theta = torch.softmax(mu + noise * torch.exp(.5 * logvar), dim=1)
            if model.model_type == "prodLDA":
                decoded = torch.softmax(model.beta_batchnorm(theta @ model.beta), dim=1)
            else:
                decoded = theta @ torch.softmax(model.beta_batchnorm(model.beta), dim=1)
            probabilities += decoded.cpu().numpy().astype(np.float64) / samples
            theta_mean += theta.cpu().numpy().astype(np.float64) / samples
        # Author topic descriptors rank raw beta; this monotone transform only
        # provides nonnegative descriptor weights. It is not the native decoder.
        weights = torch.softmax(model.beta, dim=1).cpu().numpy().astype(np.float64)
    return weights, normalized_probabilities(probabilities), theta_mean


def decode_model(method, model, observed_counts, vectors, contexts, seed, samples=20):
    import torch
    device = next(model.parameters()).device
    if method == "combinedtm":
        weights, probabilities, theta = decode_ctm(model, observed_counts, vectors["development_observed"], seed, samples)
        interface = "combinedtm_author_prodlda_posterior_predictive_20_decoded_samples"
    else:
        with torch.inference_mode():
            model.eval()
            weights_t = model.get_beta()
            if method == "fastopic":
                theta_t = model.get_theta(torch.from_numpy(vectors["development_observed"]).to(device),
                                          torch.from_numpy(vectors["fit_full"]).to(device))
                decoded = theta_t @ weights_t
                interface = "fastopic_author_theta_fixed_fitting_embedding_reference_transport_beta"
            elif method == "glocom":
                theta_t = model.get_theta(torch.from_numpy(contexts["observed"]).to(device))
                decoded = torch.softmax(model.decoder_bn(theta_t @ weights_t), dim=1)
                interface = "glocom_author_local_global_encoder_frozen_bn_decoder"
            else:
                raise ValueError("Unknown native decoder.")
            weights = weights_t.cpu().numpy().astype(np.float64)
            theta = theta_t.cpu().numpy().astype(np.float64)
            probabilities = normalized_probabilities(decoded.cpu().numpy())
    if weights.shape != (10, observed_counts.shape[1]) or theta.shape != (len(observed_counts), 10):
        raise ValueError("Native readout/latent dimensions differ from the shared contract.")
    return weights, probabilities, theta, {"prediction_interface": interface,
        "probability_roundoff_only_row_normalization": True, "descriptor_weights_are_prediction_decoder": method != "combinedtm",
        "observed_only_inference": True, "development_full_text_used_for_inference": False}
