"""Exact frozen affine readout, fitting-only moments, log-space completion."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from ..evaluation.likelihood import aggregate_nll


def affine_parameters(model):
    bn = model.decoder_bn
    if bn.training or not bn.track_running_stats:
        raise ValueError("The decoder must be frozen in evaluation mode with running statistics.")
    mean = bn.running_mean.detach().cpu().double().numpy()
    variance = bn.running_var.detach().cpu().double().numpy()
    weight = bn.weight.detach().cpu().double().numpy()
    bias = bn.bias.detach().cpu().double().numpy()
    gain = weight / np.sqrt(variance + bn.eps)
    offset = bias - gain * mean
    if not np.isfinite(gain).all() or not np.isfinite(offset).all():
        raise ValueError("Nonfinite frozen decoder coefficients.")
    return gain, offset


def moment_parameters(theta_fit, native, refined, floor=1e-6):
    if floor <= 0:
        raise ValueError("Moment floor must be positive.")
    before, after = theta_fit @ native, theta_fit @ refined
    if np.array_equal(native, refined):
        return np.ones(native.shape[1]), np.zeros(native.shape[1])
    gain = np.maximum(before.std(axis=0), floor) / np.maximum(after.std(axis=0), floor)
    offset = before.mean(axis=0) - gain * after.mean(axis=0)
    return gain, offset


def log_predictions(theta, affinity, decoder_gain, decoder_offset, moment=None):
    theta, affinity = np.asarray(theta, dtype=np.float64), np.asarray(affinity, dtype=np.float64)
    if theta.ndim != 2 or affinity.ndim != 2 or theta.shape[1] != affinity.shape[0]:
        raise ValueError("Incompatible latent/readout dimensions.")
    if not np.isfinite(theta).all() or not np.isfinite(affinity).all() or (theta < 0).any() or (affinity < 0).any():
        raise ValueError("Latents and readout must be finite and nonnegative.")
    readout = theta @ affinity
    if moment is not None:
        readout = readout * moment[0] + moment[1]
    logits = readout * decoder_gain + decoder_offset
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite decoder logits.")
    return logits - logsumexp(logits, axis=1, keepdims=True)


def backoff(logp, prior, rho):
    prior = np.asarray(prior, dtype=np.float64)
    if not 0 <= rho < 1 or (prior <= 0).any() or abs(prior.sum() - 1) > 1e-10:
        raise ValueError("Invalid backoff/prior.")
    if rho == 0:
        return logp.copy()
    return np.logaddexp(np.log1p(-rho) + logp, np.log(rho) + np.log(prior)[None, :])


def score_log_predictions(logp, records, masks, sources):
    if logp.shape[0] != len(records) or not np.isfinite(logp).all() or not np.allclose(logsumexp(logp, axis=1), 0, atol=1e-10):
        raise ValueError("Invalid normalized completion log-probabilities.")
    rows = []
    for i, record in enumerate(records):
        mask = masks[record["group_id"]]
        targets = np.asarray(mask["target_ids"], dtype=np.int64)
        if not mask["eligible"] or targets.size == 0 or targets.min() < 0 or targets.max() >= logp.shape[1]:
            raise ValueError("Invalid accepted completion mask.")
        rows.append({"group_id": record["group_id"], "source": record["source"],
            "negative_log_probability": float(-logp[i, targets].sum()), "target_tokens": int(targets.size)})
    return aggregate_nll(rows, sources), rows


def select_backoff(logp, prior, grid, records, masks, sources):
    candidates = []
    for rho in grid:
        scores, rows = score_log_predictions(backoff(logp, prior, rho), records, masks, sources)
        candidates.append({"rho": float(rho), "scores": scores, "rows": rows})
    chosen = min(candidates, key=lambda r: (r["scores"]["macro_source_nll"], r["rho"]))
    return chosen, [{"rho": r["rho"], **r["scores"]} for r in candidates]


def verify_native_interface(model, input_tensor, tolerance=1e-4):
    import torch
    model.eval()
    with torch.no_grad():
        theta = model.get_theta(input_tensor)
        affinity = model.get_beta()
        native_logits = model.decoder_bn(theta @ affinity)
    gain, offset = affine_parameters(model)
    latent = theta.detach().cpu().double().numpy()
    beta = affinity.detach().cpu().double().numpy()
    replay = (latent @ beta) * gain + offset
    float32_error = float(np.max(np.abs(replay - native_logits.detach().cpu().double().numpy())))
    # Independent float64 replay verifies the affine extraction itself.
    double_bn = torch.nn.BatchNorm1d(beta.shape[1], eps=model.decoder_bn.eps).double().eval()
    double_bn.load_state_dict({k: v.detach().cpu().double() if v.is_floating_point() else v.detach().cpu()
                               for k, v in model.decoder_bn.state_dict().items()})
    with torch.no_grad():
        exact = double_bn(torch.from_numpy(latent) @ torch.from_numpy(beta)).numpy()
    double_error = float(np.max(np.abs(replay - exact)))
    passed = float32_error <= tolerance and double_error <= 1e-10
    if not passed:
        raise AssertionError("Native frozen readout replay failed.")
    return {"verified": True, "float32_logit_max_error": float32_error,
            "float64_affine_max_error": double_error, "theta_row_normalized_after_export": False,
            "affinity_row_normalized_after_export": False, "decoder": "frozen_affine_batchnorm"}
