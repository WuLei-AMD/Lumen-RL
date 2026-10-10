"""Frozen-token cases for the Qwen3-8B ATOM train-align contract.

The numbers are the measured eager, TP=1 runs. They are the acceptance record
for this case, not a fresh benchmark.
"""

from __future__ import annotations

# 93 frozen tokens, one GPU, eager, TP=1. Qwen3-8B BF16.
PREFILL_MS = {"original": 14.730, "train_align": 14.782}
DECODE_MS_PER_TOKEN = {"original": 13.835, "train_align": 23.931}
DECODE_BODY_MS_PER_TOKEN = {"original": 13.575, "train_align": 23.595}
DECODE_LM_HEAD_MS_PER_TOKEN = {"original": 0.246, "train_align": 0.323}

# Where the extra decode time went, 93 steps, self device time.
KERNEL_DELTA_MS = {
    "padded_gate_down_oproj": 78.0,
    "pad_copy_cat_fill": 99.0,
    "qkv_split_k1_instead_of_split_k4": 16.0,
    "varlen_instead_of_pa": 5.0,
}

# Decode QKV split_k4 is not a stable target. Two launches of one input.
QKV_SPLITK4_SELF_MISMATCH = (1222, 4096)
QKV_SPLITK4_MAX_ABS = 1.953e-3


def contract_holds(place_ok: bool, qkv_name_ok: bool, pad_shapes_ok: bool) -> bool:
    return place_ok and qkv_name_ok and pad_shapes_ok
