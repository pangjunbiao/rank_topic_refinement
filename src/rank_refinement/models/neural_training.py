"""Pinned author objectives, canonical data, resumable native training."""
from __future__ import annotations
import io
import os
import random
import time
import types
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from ..runtime.artifacts import atomic_bytes, digest, file_hash, read_json, write_json
from .neural_sources import author_class


def batches(n, batch_size, seed):
    if n < 2 or batch_size < 2:
        raise ValueError("BatchNorm training requires at least two fitting documents.")
    order = np.random.default_rng(seed).permutation(n)
    chunks = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] = np.r_[chunks[-2], chunks[-1]]
        chunks.pop()
    return chunks


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_model(base, folder, vocab_size, dimension, policy, words=None, seed=809):
    import torch
    seed_everything(seed)
    settings = dict(policy["training"][base])
    for key in ["epochs", "scheduler_step", "scheduler_gamma"]:
        settings.pop(key, None)
    settings.update(vocab_size=vocab_size, num_topics=policy["topics"], embed_size=dimension,
                    pretrained_WE=words, sinkhorn_max_iter=policy["sinkhorn_max_iter"])
    model = author_class(base, folder)(**settings)
    if base == "neuromax":
        # The author's InfoNCE projection is hard-coded to 384. The shared
        # Chinese encoder has a different width; preserve the same objective.
        if model.prj_rep[0].out_features != dimension:
            model.prj_rep[0] = torch.nn.Linear(policy["topics"], dimension)
        # Correct device allocation only: +10 remains exactly the author cost.
        def portable_group_loss(self):
            cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.topic_embeddings)
            return self.GR(cost + torch.full_like(cost, 10.0), self.group_connection_regularizer)
        model.get_loss_GR = types.MethodType(portable_group_loss, model)
    return model


def load_canonical(preparation):
    import json
    def jsonl(name):
        return [json.loads(line) for line in (preparation / name).read_text(encoding="utf-8").splitlines() if line]
    records = {r["group_id"]: r for r in jsonl("feature_records.jsonl")}
    masks = {r["group_id"]: r for r in jsonl("completion_masks.jsonl")}
    fit_ids = read_json(preparation / "model_fit_group_ids.json")
    completion = read_json(preparation / "completion_group_ids_by_split.json")
    fit_observed_ids = [g for g in fit_ids if masks[g]["eligible"]]
    dev_ids = completion["development"]
    if set(fit_ids) & set(dev_ids) or any(records[g]["split"] != "fit" for g in fit_ids):
        raise ValueError("Canonical fitting/development firewall failed.")
    if any(records[g]["split"] != "development" or not masks[g]["eligible"] for g in dev_ids):
        raise ValueError("Canonical development completion population changed.")
    matrix_ids = read_json(preparation / "matrix_rows.json")
    index = {g: i for i, g in enumerate(matrix_ids)}
    full = sparse.load_npz(preparation / "full_counts.npz")
    observed = sparse.load_npz(preparation / "observed_counts.npz")
    metadata = read_json(preparation / "vocabulary.json")
    fit = full[[index[g] for g in fit_ids]].toarray().astype(np.float32)
    fit_observed = observed[[index[g] for g in fit_observed_ids]].toarray().astype(np.float32)
    dev_observed = observed[[index[g] for g in dev_ids]].toarray().astype(np.float32)
    for groups, counts in [(fit_ids, fit), (fit_observed_ids, fit_observed), (dev_ids, dev_observed)]:
        if len(groups) != len(set(groups)) or len(groups) != counts.shape[0] or (counts.sum(axis=1) <= 0).any():
            raise ValueError("Empty, duplicated or misaligned canonical model rows.")
    return {"fit_ids": fit_ids, "fit_observed_ids": fit_observed_ids, "dev_ids": dev_ids,
            "fit": fit, "fit_observed": fit_observed, "dev_observed": dev_observed,
            "fit_records": [records[g] for g in fit_ids], "dev_records": [records[g] for g in dev_ids],
            "dev_coherence_records": [r for _, r in sorted(records.items()) if r["split"] == "development"],
            "fit_observed_records": [records[g] for g in fit_observed_ids], "masks": masks,
            "metadata": metadata, "vocabulary": metadata["tokens"]}


def encode_views(root, data, encoder_policy, device, log):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .encoder_assets import resolve_encoder
    folder, asset = resolve_encoder(root, encoder_policy, log)
    views = {"fit_full": [r["clean_text"] for r in data["fit_records"]],
        "fit_observed": [data["masks"][g]["observed_text"] for g in data["fit_observed_ids"]],
        "development_observed": [data["masks"][g]["observed_text"] for g in data["dev_ids"]],
        "words": [w.replace("_", "") for w in data["vocabulary"]]}
    identity = {"assets": asset, "texts": views, "fit_ids": data["fit_ids"],
        "fit_observed_ids": data["fit_observed_ids"], "dev_ids": data["dev_ids"],
        "pooling": "cls_l2", "max_length": encoder_policy["max_length"],
        "torch": torch.__version__, "device": device}
    key = digest(identity)
    cache = Path(root) / "data/cache/step03_embeddings" / key[:20]
    lock, arrays = cache / "manifest.json", cache / "vectors.npz"
    if lock.is_file():
        manifest = read_json(lock)
        if manifest["key"] != key or file_hash(arrays) != manifest["vectors_sha256"]:
            raise ValueError("Step 3 encoder cache checksum/identity failed.")
        with np.load(arrays, allow_pickle=False) as archive:
            vectors = {name: archive[name] for name in views}
        log.info("EMBEDDINGS | verified cache reused | fitting and observed-only development views")
    else:
        model = AutoModel.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False,
                                          use_safetensors=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        vectors, truncations = {}, {}
        with torch.inference_mode():
            for name, texts in views.items():
                chunks, truncated = [], 0
                size = encoder_policy["batch_size"]
                for start in range(0, len(texts), size):
                    selected = texts[start:start + size]
                    lengths = tokenizer(selected, add_special_tokens=True, truncation=False)["input_ids"]
                    truncated += sum(len(r) > encoder_policy["max_length"] for r in lengths)
                    inputs = tokenizer(selected, return_tensors="pt", padding=True, truncation=True,
                                       max_length=encoder_policy["max_length"])
                    output = model(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[:, 0]
                    chunks.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy().astype(np.float32))
                    log.info("EMBEDDINGS | %s | %d/%d | %s", name, min(start + size, len(texts)), len(texts), device)
                vectors[name] = np.concatenate(chunks)
                truncations[name] = truncated
        stream = io.BytesIO()
        np.savez_compressed(stream, **vectors)
        atomic_bytes(arrays, stream.getvalue())
        manifest = {"key": key, "assets": asset, "vectors_sha256": file_hash(arrays), "device": device,
            "truncated_by_view": truncations, "dimension": vectors["fit_full"].shape[1],
            "view_sizes": {k: len(v) for k, v in views.items()},
            "fitting_group_ids": data["fit_ids"], "fitting_observed_group_ids": data["fit_observed_ids"],
            "development_observed_group_ids": data["dev_ids"], "development_full_text_encoded": False,
            "test_or_external_encoded": False, "pooling": "cls_l2"}
        write_json(lock, manifest)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    for name in views:
        if vectors[name].shape != (len(views[name]), manifest["dimension"]) or not np.isfinite(vectors[name]).all():
            raise ValueError("Invalid encoder view dimensions or values.")
        if not np.allclose(np.linalg.norm(vectors[name], axis=1), 1, atol=1e-5):
            raise ValueError("Encoder vectors failed unit-norm verification.")
    return vectors, manifest


def global_context(data, vectors, policy):
    if policy["aggregation"] != "sum_fitting_counts":
        raise ValueError("Unsupported global-context aggregation.")
    cluster = KMeans(n_clusters=policy["clusters"], n_init=policy["n_init"],
                     random_state=policy["random_state"]).fit(vectors["fit_full"])
    bank = np.zeros((policy["clusters"], data["fit"].shape[1]), dtype=np.float32)
    np.add.at(bank, cluster.labels_, data["fit"])
    return {"fit": np.concatenate([data["fit"], bank[cluster.labels_]], axis=1),
        "fit_observed": np.concatenate([data["fit_observed"], bank[cluster.predict(vectors["fit_observed"])]], axis=1),
        "development_observed": np.concatenate([data["dev_observed"], bank[cluster.predict(vectors["development_observed"])]], axis=1),
        "bank": bank, "centers": cluster.cluster_centers_, "fit_assignments": cluster.labels_}


def save_checkpoint(path, model, optimizer, scheduler, epoch, identity, history):
    import torch
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None, "epoch": epoch,
        "identity": identity, "history": history, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "group_regularizer": getattr(model, "group_connection_regularizer", None)}
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())
    write_json(path.with_suffix(".json"), {"identity": identity, "epoch": epoch, "checkpoint_sha256": file_hash(path)})


def train_native(root, base, profile, seed, policy, folders, source_specs, data, vectors, contexts, device, log):
    import torch
    settings = policy["training"][base]
    identity = {"base": base, "profile": profile, "seed": seed, "training": settings,
        "sinkhorn_max_iter": policy["sinkhorn_max_iter"], "topics": policy["topics"],
        "batch_size": policy["batch_size"], "lr": policy["learning_rate"], "device": device,
        "fitting_ids_sha256": digest(data["fit_ids"]), "fitting_counts_sha256": __import__("hashlib").sha256(data["fit"].tobytes()).hexdigest(),
        "embedding_sha256": __import__("hashlib").sha256(b"".join(vectors[n].tobytes() for n in ["fit_full", "words"])).hexdigest(),
        "global_context_policy": policy["global_context"] if base == "encot" else None,
        "dimension": vectors["fit_full"].shape[1], "source_specs": [s for s in source_specs if s["base"] == base],
        "training_implementation_sha256": file_hash(__file__), "torch": torch.__version__}
    key = digest(identity)
    cache = Path(root) / "data/cache/step03_native" / key[:20]
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / "checkpoint.pt"
    words = vectors["words"] if profile == "chinese_words" else None
    model = make_model(base, folders[base], data["fit"].shape[1], vectors["fit_full"].shape[1], policy, words, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=policy["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings["scheduler_step"], gamma=settings["scheduler_gamma"]) if base == "neuromax" else None
    history, completed = [], 0
    if checkpoint.is_file():
        index = read_json(checkpoint.with_suffix(".json"))
        if index["identity"] != identity or file_hash(checkpoint) != index["checkpoint_sha256"]:
            raise ValueError("Native checkpoint checksum/identity failed; the cache was not silently replaced.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["epoch"] != index["epoch"]:
            raise ValueError("Native checkpoint payload differs from its index.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        if base == "neuromax" and saved["group_regularizer"] is not None:
            model.group_connection_regularizer = saved["group_regularizer"].to(device)
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        completed, history = saved["epoch"], saved["history"]
        log.info("NATIVE CACHE | %s %s seed=%d | resume epoch=%d/%d", base, profile, seed, completed, settings["epochs"])
    else:
        log.info("NATIVE FIT | %s %s seed=%d | epochs=%d | %s", base, profile, seed, settings["epochs"], device)
    train = contexts["fit"] if base == "encot" else data["fit"]
    input_data = torch.from_numpy(train).to(device)
    contextual = torch.from_numpy(vectors["fit_full"]).to(device)
    started = time.perf_counter()
    for epoch in range(completed + 1, settings["epochs"] + 1):
        model.train()
        totals, size = {}, 0
        for ids in batches(len(train), policy["batch_size"], seed * 100000 + epoch):
            ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            inputs = input_data[ids_t] if base == "encot" else {"data": input_data[ids_t], "contextual_embed": contextual[ids_t]}
            result = model(inputs) if base == "encot" else model(inputs, epoch_id=epoch)
            if any(not np.isfinite(float(v.detach().cpu() if torch.is_tensor(v) else v)) for v in result.values()):
                raise FloatingPointError(f"Nonfinite native loss: {base} seed={seed} epoch={epoch}. No model was dropped.")
            result["loss"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite native gradient: {base} seed={seed} epoch={epoch}.")
            optimizer.step()
            for name, value in result.items():
                totals[name] = totals.get(name, 0.) + float(value.detach().cpu() if torch.is_tensor(value) else value) * len(ids)
            size += len(ids)
        if size != len(train):
            raise AssertionError("A native epoch omitted or repeated documents.")
        if scheduler:
            scheduler.step()
        history.append({"epoch": epoch, "groups": size, "learning_rate": optimizer.param_groups[0]["lr"],
                        **{name: value / size for name, value in totals.items()}})
        if epoch == 1 or epoch % policy["checkpoint_every_epochs"] == 0 or epoch == settings["epochs"]:
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, identity, history)
            write_json(cache / "training_history.json", history)
            log.info("TRAIN | %s %s seed=%d | epoch=%d/%d | fitting loss=%.6f | elapsed=%.1fs", base, profile, seed,
                     epoch, settings["epochs"], history[-1]["loss"], time.perf_counter() - started)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite native model parameters.")
    model.eval()
    return model, {"key": key, "identity": identity, "epochs": settings["epochs"],
        "completed_before_run": completed, "new_training_epochs": settings["epochs"] - completed,
        "checkpoint_relative_path": checkpoint.relative_to(Path(root)).as_posix(),
        "checkpoint_sha256": file_hash(checkpoint), "training_history": history,
        "training_seconds_this_run": time.perf_counter() - started,
        "author_objective_preserved": True, "development_used_for_training": False,
        "test_or_external_used": False,
        "adaptations": ["Chinese encoder and word initialization profile", "K=10"] +
            (["Chinese-width InfoNCE projection", "group count 4 for ten topics", "device-correct +10 group cost"] if base == "neuromax"
             else ["fitting-only global count bank; heldout assignment from observed text"])}


def export_latents(model, inputs, device, batch_size=200):
    import torch
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            value = model.get_theta(torch.from_numpy(inputs[start:start + batch_size]).to(device))
            rows.append(value.cpu().double().numpy())
    return np.concatenate(rows)
