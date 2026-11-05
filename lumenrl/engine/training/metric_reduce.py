"""Reductions shared by the two actor-worker implementations.

``mismatch_abs_max`` is a max across micro-batches. ``mismatch_count`` is a
sum. Every other listed metric stays a mean, so a later ratio of two means is
still the global token-weighted value.
"""

from __future__ import annotations

import torch


def reduce_listed_metrics(values: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, list) and value:
            if key == "mismatch_abs_max":
                metrics[key] = max(value)
            elif key == "mismatch_count":
                metrics[key] = float(sum(value))
            else:
                metrics[key] = sum(value) / len(value)
        else:
            metrics[key] = float(value)
    return metrics


def mismatch_sums(delta, mask, tok: float) -> dict[str, float]:
    """Sums the controller turns into token-weighted mismatch means.

    ``delta`` is already multiplied by the response mask. ``mismatch_count``
    counts nonzero masked tokens and is reduced by sum. ``mismatch_abs_max``
    is reduced by max.
    """
    # exp() of a log-ratio: clamp so a pathological sequence cannot turn the
    # whole metric into inf and hide the rest.
    weight = torch.exp(delta.clamp(-10.0, 10.0))
    seq_delta = delta.sum(dim=-1).clamp(-10.0, 10.0)
    seq_weight = torch.exp(seq_delta)
    n_seq = float(delta.shape[0])
    return {
        "mismatch_abs_sum": float(delta.abs().sum()),
        "mismatch_abs_max": float(delta.abs().max()) if delta.numel() else 0.0,
        "mismatch_k3_sum": float((((weight - 1.0) - delta) * mask).sum()),
        "mismatch_chi2_tok_sum": float((((weight - 1.0) ** 2) * mask).sum()),
        "mismatch_chi2_seq_sum": float(((seq_weight - 1.0) ** 2).sum()),
        "mismatch_seq_n": n_seq,
        "mismatch_tail_sum": float(((delta.abs() > 0.1).float() * mask).sum()),
        "mismatch_count": float((delta != 0).sum()),
        "mismatch_tok": tok,
    }
