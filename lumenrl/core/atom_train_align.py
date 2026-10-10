"""Operators that make ATOM decode match a Lumen ``rollout: ATOM`` prefill.

The training forward and ATOM prefill already share one BF16 contract on
Qwen3-8B: RMSNorm, cached RoPE, varlen attention, and one full-M GEMM.
Decode does not. Its QKV kernel splits K into 4 parts and atomic-adds the
BF16 partials, so two launches of the same input disagree. Gate, down,
``o_proj`` and the LM head also change bits when M is 1 instead of the
prefill length.

This module is the contract those paths have to share. ``activate`` exports
it into the process. ``install`` points ATOM's existing decode hooks at the
operators below. Lumen's ``rollout: ATOM`` patch stays the training side of
the same contract.
"""

from __future__ import annotations

import os
from typing import Optional

# Qwen3-8B fused QKV. Prefill selects this flydsl kernel (solidx 4696).
# Replaying row 0 of an M=93 GEMM on the same kernel is bitwise identical.
QKV_WEIGHT = (6144, 4096)
QKV_SOLIDX = 4696
QKV_SPLITK1 = (
    "flydsl_gemm5_abf16_wbf16_bf16_t64x64x128_split_k1_"
    "block_m_warp2_block_n_warp2_block_k_warp2_"
    "async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950"
)

# Shapes whose M=1 GEMM does not match the prefill row. Gate/up is the fused
# N=24576 projection. Down is K=12288. o_proj and the LM head are padded by
# their own callers; they are not in this set.
PAD_WEIGHTS = {(24576, 4096), (4096, 12288)}

ENV_M = "ATOM_TRAIN_ALIGN_M"
ENV_BITWISE_M = "ATOM_BITWISE_M"
ENV_ENABLE = "ATOM_TRAIN_ALIGN"
ENV_LUMEN_ROLLOUT = "LUMEN_ROLLOUT"


def configured_m() -> int:
    """Prefill length the decode row must be placed into. 0 disables the contract."""
    raw = os.environ.get(ENV_M) or os.environ.get(ENV_BITWISE_M) or ""
    return int(raw) if raw else 0


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") == "1" and configured_m() > 1


def activate(seqlen: int) -> int:
    """Turn the contract on for this process and any child that inherits the env.

    ``seqlen`` is the prefill M. Decode still runs one token at a time; the
    operators below put that token back at its row inside this M.
    """
    length = int(seqlen)
    if length <= 1:
        raise ValueError(f"ATOM train-align needs a prefill length > 1, got {length}")
    os.environ[ENV_ENABLE] = "1"
    os.environ[ENV_M] = str(length)
    os.environ[ENV_BITWISE_M] = str(length)
    os.environ[ENV_LUMEN_ROLLOUT] = "ATOM"
    install()
    return length


def lumen_rollout_name() -> str:
    """Value Lumen's ``rollout`` field should take when this contract is on."""
    if os.environ.get(ENV_ENABLE, "0") == "1":
        return "ATOM"
    if os.environ.get(ENV_LUMEN_ROLLOUT, "").upper() == "ATOM":
        return "ATOM"
    return ""


def bind_lumen_rollout() -> None:
    """Called from Lumen when ``rollout`` is already ATOM.

    Training uses the full sequence, so it does not pad. Exporting the flag
    keeps the rollout process on the same operator set.
    """
    os.environ[ENV_LUMEN_ROLLOUT] = "ATOM"


def place_decode_row(activation, length: int, position: int):
    """Write one decode row into a zero prefill tensor and return ``(padded, position)``."""
    import torch

    if length <= 1 or activation.dim() != 2 or activation.shape[0] != 1:
        return None
    if position < 0 or position >= length:
        return None
    padded = torch.zeros(
        length, activation.shape[-1], dtype=activation.dtype, device=activation.device
    )
    padded[position : position + 1] = activation
    return padded, position


def bitwise_place(activation):
    """ATOM decode hook: place the current token at its prefill row.

    Prefill returns ``None`` and keeps the full-M GEMM. Decode returns the
    padded tensor and the row to slice back out.
    """
    length = configured_m()
    if length <= 1 or activation.dim() != 2 or activation.shape[0] != 1:
        return None
    from atom.utils.forward_context import get_forward_context

    ctx = get_forward_context().context
    if ctx is None or ctx.is_prefill:
        return None
    position = int(ctx.positions.reshape(-1)[0].item())
    return place_decode_row(activation, length, position)


def pin_qkv(activation, weight, bias=None, otype=None):
    """Run fused QKV on the prefill ``split_k1`` kernel, including M=1 decode."""
    from aiter.tuned_gemm import flydsl_gemm

    return flydsl_gemm(
        activation,
        weight,
        QKV_SOLIDX,
        bias=bias,
        otype=otype,
        config={"kernelName": QKV_SPLITK1},
    )


def should_pin_qkv(weight_shape, rows: int) -> bool:
    return tuple(weight_shape) == QKV_WEIGHT and int(rows) == 1 and configured_m() > 1


def should_pad_weight(weight_shape) -> bool:
    return tuple(weight_shape) in PAD_WEIGHTS and configured_m() > 1


def run_padded_gemm(activation, weight, bias, otype, gemm):
    """Pad a decode row, run ``gemm``, and slice that row back out."""
    placed = bitwise_place(activation)
    if placed is None:
        return gemm(activation, weight, bias, otype)
    padded, position = placed
    return gemm(padded, weight, bias, otype)[position : position + 1]


def install() -> Optional[int]:
    """Point ATOM's decode hooks at this module. No-op when the contract is off."""
    length = configured_m()
    if length <= 1:
        return None
    try:
        import atom.model_ops.linear as linear
    except Exception:
        # CPU config load and processes without a GPU cannot import ATOM.
        # The env flags above are what the GPU worker reads.
        return length
    linear.bitwise_m = configured_m
    linear.bitwise_place = bitwise_place
    linear._QKV_SPLITK1 = QKV_SPLITK1
    linear._PAD_WEIGHTS = PAD_WEIGHTS
    return length
