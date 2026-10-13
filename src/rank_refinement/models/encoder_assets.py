"""Pinned Chinese encoder; downloads and embeddings precede comparator scores."""
from __future__ import annotations
import json
import os
import urllib.request
from pathlib import Path
import numpy as np
from ..runtime.artifacts import digest, file_hash, read_json, write_json, atomic_bytes

FILES = ["config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]


def resolve_encoder(root, policy, log):
    folder = Path(root) / policy["local_directory"] / policy["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "asset_manifest.json"
    if lock.is_file():
        manifest = read_json(lock)
        if manifest["repository"] != policy["repository"] or manifest["revision"] != policy["revision"]:
            raise RuntimeError("Encoder asset identity mismatch.")
        for info in manifest["files"]:
            if not (folder / info["path"]).is_file() or file_hash(folder / info["path"]) != info["sha256"]:
                raise RuntimeError("A pinned encoder asset changed: " + info["path"])
            if info["sha256"] != policy["file_sha256"][info["path"]]:
                raise RuntimeError("An encoder asset differs from the authoritative pinned file.")
        return folder, manifest
    errors = []
    for name in FILES:
        destination = folder / name
        if not destination.is_file():
            for endpoint in policy["endpoints"]:
                url = endpoint.rstrip("/") + "/" + policy["repository"] + "/resolve/" + policy["revision"] + "/" + name
                try:
                    log.info("ASSET | downloading pinned %s", name)
                    temporary = destination.with_suffix(destination.suffix + ".download")
                    with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as f:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    temporary.replace(destination)
                    break
                except Exception as exc:
                    errors.append({"file": name, "endpoint": endpoint, "error": str(exc)})
            else:
                raise RuntimeError("Pinned Chinese encoder download blocked. No comparator is dropped or substituted. " + str(errors))
    weight_sha = policy.get("weights_sha256")
    if weight_sha and file_hash(folder / policy["weights"]) != weight_sha:
        raise RuntimeError("Encoder weights do not match their authoritative pinned SHA-256.")
    for name in FILES:
        if file_hash(folder / name) != policy["file_sha256"][name]:
            raise RuntimeError("Encoder asset checksum mismatch: " + name)
    manifest = {"repository": policy["repository"], "revision": policy["revision"], "policy": policy,
                "files": [{"path": name, "bytes": (folder / name).stat().st_size, "sha256": file_hash(folder / name)} for name in FILES]}
    write_json(lock, manifest)
    return folder, manifest


def fitting_embeddings(root, records, policy, device, log, destination):
    import torch
    from transformers import AutoTokenizer, AutoModel
    records = sorted(records, key=lambda r: r["group_id"])
    if any(r["split"] != "fit" for r in records) or len({r["group_id"] for r in records}) != len(records):
        raise ValueError("Encoder fitting cache may contain only unique outer-fitting groups.")
    folder, assets = resolve_encoder(root, policy, log)
    identity = {"assets": assets, "group_texts": [{"group_id": r["group_id"], "text": r["clean_text"]} for r in records],
                "torch": torch.__version__, "device": device, "pooling": "cls_l2", "max_length": policy["max_length"]}
    key = digest(identity)
    cache = Path(root) / "data/cache/embeddings" / key
    cache.mkdir(parents=True, exist_ok=True)
    values, metadata = cache / "vectors.npy", cache / "manifest.json"
    reused = values.is_file() and metadata.is_file()
    if reused:
        manifest = read_json(metadata)
        if manifest["key"] != key or file_hash(values) != manifest["vectors_sha256"]:
            raise RuntimeError("Embedding cache failed identity/hash validation.")
        embeddings = np.load(values, allow_pickle=False)
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for encoder but unavailable; no silent CPU device substitution.")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        model = AutoModel.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        model.eval()
        batch = policy["batch_size"]
        vectors, truncations = [], 0
        with torch.inference_mode():
            for start in range(0, len(records), batch):
                texts = [r["clean_text"] for r in records[start:start + batch]]
                lengths = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
                truncations += sum(len(ids) > policy["max_length"] for ids in lengths)
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=policy["max_length"], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model(**inputs).last_hidden_state[:, 0, :]
                output = torch.nn.functional.normalize(output, p=2, dim=1)
                vectors.append(output.cpu().numpy().astype(np.float32))
                log.info("EMBEDDINGS | %d/%d fitting groups | device=%s", min(start + batch, len(records)), len(records), device)
        embeddings = np.concatenate(vectors, axis=0)
        temporary = values.with_suffix(".tmp.npy")
        np.save(temporary, embeddings, allow_pickle=False)
        temporary.replace(values)
        manifest = {"key": key, "assets": assets, "group_ids": [r["group_id"] for r in records],
                    "device": device, "vectors_sha256": file_hash(values), "truncated_documents": truncations,
                    "max_length": policy["max_length"], "shape": list(embeddings.shape),
                    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0}
        write_json(metadata, manifest)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if embeddings.shape[0] != len(records) or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise AssertionError("Pinned embedding shape/normalization check failed.")
    write_json(Path(destination) / "encoder_assets.json", assets)
    current_manifest = {**manifest, "reused_embeddings": reused,
                        "current_run_peak_cuda_allocated_gib": 0.0 if reused else manifest["peak_cuda_allocated_gib"]}
    write_json(Path(destination) / "encoder_embeddings_manifest.json", current_manifest)
    return embeddings, {r["group_id"]: i for i, r in enumerate(records)}, current_manifest
