"""Deterministic, batch-invariant blockwise FP8 MoE experts shared by the rollout and the trainer.

Used with ``LUMENRL_BATCH_INVARIANT_MOE=1`` when the rollout runs ATOM per_1x128 FP8 experts
(``per_block_fp8``) and the trainer has ``LUMENRL_FP8_MOE=1``.

Weights (``quant_weight_atom``): ATOM's online 128x128 quantizer
(``quantize_weight_to_fp8_128x128_blockscale``: fp32 block amax, ``amax / 448`` clamped at the
fp32 tiny, ``w / scale`` clamped to +-448, round-to-nearest-even cast) as one Triton kernel, so
the trainer gets the bytes and scales ATOM loads and syncs, at 1/15 of the cost.

Experts (``batch_invariant_fp8_experts_core``): aiter per_1x128 ``fused_moe`` on expanded rows
(each (token, slot) is its own row, topk=1, weight 1), then the fixed-order fp32 reduce of
``batch_invariant_experts``. aiter picks its MoE dispatch from the row count: 1-stage asm tiles
and split-K both change a row's bits with the batch around it. ``pin_fp8_moe_dispatch`` fixes
every topk=1 per_1x128 call to 2-stage CK without split-K (block_m 16 or 64, which agree), which
gives the same bits at any row count.

Trainer (``BatchInvariantFp8HFExperts``): forward is the core on ATOM-quantized, shuffled copies
of the HF parameters, cached until the next optimizer step; backward runs the Sonic blockwise
FP8 grouped GEMMs of ``lumenrl.moe.fp8_experts``. Parameters stay in HF layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from lumen.ops.gemm.fp8_dense import _store_tile

from lumenrl.moe.batch_invariant_experts import _dense, _expanded_reduce

_BLOCK = 128
_FP8_MAX = 448.0
_INV_FP8_MAX = (torch.ones((), dtype=torch.float32) / _FP8_MAX).item()
# aiter's per_1x128 heuristic picks block_m 16 at token 16 and block_m 64 at token 4096.
_SMALL_TOKEN = 16
_PIN_TOKEN = 4096
_SMALL_ROWS = 1024


@triton.jit
def _atom_quant_kernel(w_ptr, q_ptr, s_ptr, rows, cols, n_rb, n_cb, fp8_max, inv_fp8_max, tiny,
                       BLOCK: tl.constexpr, SHUF: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    r = rb * BLOCK + tl.arange(0, BLOCK)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    off = e * rows * cols + r[:, None].to(tl.int64) * cols + c[None, :]
    x = tl.load(w_ptr + off).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    # torch's CUDA divide by a Python scalar multiplies by the fp32 reciprocal.
    scale = tl.maximum(amax * inv_fp8_max, tiny)
    q = tl.div_rn(x, scale)
    q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max).to(q_ptr.dtype.element_ty)
    if SHUF:
        # (E, rows, cols) shuffles as (E * rows, cols): the (16, 16) layout groups 16 rows.
        _store_tile(q_ptr, q, (e * rows + rb * BLOCK).to(tl.int32), cb * BLOCK, cols, True)
    else:
        tl.store(q_ptr + off, q)
    tl.store(s_ptr + (e * n_rb + rb) * n_cb + cb, scale)


def quant_weight_atom(w: torch.Tensor, fp8_dtype=None, shuffle: bool = False):
    """ATOM's per-expert 128x128 FP8 quant of (E, rows, cols).

    Returns (q (E, rows, cols), scale (E, rows/128, cols/128)); ``shuffle=True`` stores q in
    ``shuffle_weight(layout=(16, 16))`` order, the layout ATOM's FusedMoE holds.
    """
    from aiter import dtypes

    fp8_dtype = fp8_dtype or dtypes.fp8
    w = w.contiguous()
    e, rows, cols = w.shape
    if rows % _BLOCK or cols % _BLOCK:
        raise ValueError(f"expert weight {tuple(w.shape)} is not 128-aligned")
    q = torch.empty(e, rows, cols, dtype=fp8_dtype, device=w.device)
    n_rb, n_cb = rows // _BLOCK, cols // _BLOCK
    s = torch.empty(e, n_rb, n_cb, dtype=torch.float32, device=w.device)
    if e:
        _atom_quant_kernel[(e, n_rb, n_cb)](
            w, q, s, rows, cols, n_rb, n_cb, _FP8_MAX, _INV_FP8_MAX, torch.finfo(torch.float32).tiny,
            BLOCK=_BLOCK, SHUF=shuffle, num_warps=8,
        )
    return q, s


def pin_fp8_moe_dispatch() -> None:
    """Fix aiter's dispatch for topk=1 per_1x128 ``fused_moe`` calls to 2-stage CK without split-K.

    block_m 16 (up to ``_SMALL_ROWS`` rows) and block_m 64 give the same bits per row, so the
    tile follows the row count for speed.
    """
    import dataclasses
    import functools

    import aiter.fused_moe as fm
    from aiter import QuantType

    orig = fm.get_2stage_cfgs
    if getattr(orig, "_lumenrl_fp8_pinned", False):
        return

    @functools.lru_cache(maxsize=None)
    def no_splitk(token, args, kwargs):
        held = {}
        for gfx, table in fm.fused_moe_1stage_dict.items():
            held[gfx] = {key: table.pop(key) for key in [key for key in table if key[1] == QuantType.per_1x128]}
        try:
            meta = orig(token, *args, **dict(kwargs))
        finally:
            for gfx, entries in held.items():
                fm.fused_moe_1stage_dict[gfx].update(entries)
        if meta.run_1stage or meta.stage1 is None or "splitk" not in meta.stage1.keywords:
            raise RuntimeError(f"per_1x128 MoE pin expected a 2-stage CK config, got {meta}")
        stage1 = meta.stage1
        stage1 = functools.partial(stage1.func, *stage1.args, **{**stage1.keywords, "splitk": 0})
        return dataclasses.replace(meta, stage1=stage1, ksplit=0)

    def pinned(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, *args, **kwargs):
        if topk != 1 or q_type != QuantType.per_1x128:
            return orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, *args, **kwargs)
        tile_token = _SMALL_TOKEN if token <= _SMALL_ROWS else _PIN_TOKEN
        return no_splitk(
            tile_token, (model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type) + args,
            tuple(sorted(kwargs.items())),
        )

    pinned._lumenrl_fp8_pinned = True
    fm.get_2stage_cfgs = pinned


def _expanded_fp8_experts(x, w13, w2, w13_scale, w2_scale, top_index, ones=None, x_rep=None,
                          expert_mask=None, activation=None):
    """Per-(token, slot) expert outputs, shape (m * k, H), from the pinned per_1x128 CK path."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    pin_fp8_moe_dispatch()
    if activation is None:
        activation = ActivationType.Silu
    m, k = top_index.shape
    ids = top_index.to(torch.int32).contiguous().view(m * k, 1)
    if ones is None:
        ones = torch.ones(m * k, 1, dtype=torch.float32, device=x.device)
    if x_rep is None:
        x_rep = x.repeat_interleave(k, dim=0)
    return fused_moe(
        x_rep, w13, w2, ones, ids,
        expert_mask=expert_mask, activation=activation, quant_type=QuantType.per_1x128,
        w1_scale=w13_scale, w2_scale=w2_scale,
    )


def batch_invariant_fp8_experts_core(x, w13, w2, w13_scale, w2_scale, top_weights, top_index,
                                     ones=None, x_rep=None, expert_mask=None, activation=None):
    """Expanded-row per_1x128 CK experts + fixed-order reduce. ``w13``/``w2`` are ATOM-quantized and shuffled."""
    y = _expanded_fp8_experts(x, w13, w2, w13_scale, w2_scale, top_index, ones, x_rep, expert_mask, activation)
    return _expanded_reduce(y, top_weights)


_GEN = [0]
_HOOK = [None]
_OWNERS = []


def _bump_generation(*_args, **_kwargs) -> None:
    """Free every cached FP8 expert weight after ``optimizer.step()``, before the weight sync."""
    _GEN[0] += 1
    for owner in _OWNERS:
        owner.__dict__.pop("_lumenrl_fp8_w", None)
    _OWNERS.clear()


def _cached_shuffled_weights(owner, gate_up, down):
    """ATOM-quantized, shuffled (w13, s13, w2, s2) of the HF experts, kept until the next optimizer step.

    ``LUMENRL_FP8_MOE_WEIGHT_CACHE=0`` requantizes on every forward (one full FP8 copy of each
    layer's experts lives only for that forward).
    """
    import os

    entry = owner.__dict__.get("_lumenrl_fp8_w")
    if entry is not None and entry[0] == _GEN[0]:
        return entry[1]
    with torch.no_grad():
        w13, s13 = quant_weight_atom(gate_up.detach(), shuffle=True)
        w2, s2 = quant_weight_atom(down.detach(), shuffle=True)
    weights = (w13, s13, w2, s2)
    if os.environ.get("LUMENRL_FP8_MOE_WEIGHT_CACHE", "1") == "1":
        if _HOOK[0] is None:
            from torch.optim.optimizer import register_optimizer_step_post_hook

            _HOOK[0] = register_optimizer_step_post_hook(_bump_generation)
        owner.__dict__["_lumenrl_fp8_w"] = (_GEN[0], weights)
        _OWNERS.append(owner)
    return weights


class _BatchInvariantFp8Experts(torch.autograd.Function):
    """Forward: CK expanded FP8 experts on ATOM-quantized, shuffled copies of the HF parameters.

    Backward: Sonic blockwise FP8 grouped GEMMs on the same parameters. Only the first expert
    GEMM is recomputed; the router-weight gradient uses the saved per-slot CK outputs, the ones
    the forward reduced.
    """

    @staticmethod
    def forward(ctx, x, gate_up, down, top_index, top_weights, w13, s13, w2, s2):
        x = x.contiguous()
        y = _expanded_fp8_experts(x, w13, w2, s13, s2, top_index)
        ctx.save_for_backward(x, gate_up, down, top_index, top_weights, y)
        return _expanded_reduce(y, top_weights)

    @staticmethod
    def backward(ctx, grad_out):
        from aiter.ops.triton.moe.sonicmoe import activation_bwd, activation_fwd

        from lumen.ops.gemm.grouped_gemm import _cu_seqlens_from_group_sizes
        from lumenrl.moe.fp8_experts import _gemm, _linear_backward

        x, gate_up, down, top_index, top_weights, y = ctx.saved_tensors
        t, k = top_index.shape
        h = x.shape[-1]
        g32 = grad_out.float().unsqueeze(1)
        g_w = None
        if ctx.needs_input_grad[4]:
            g_w = (g32 * y.view(t, k, h).float()).sum(-1).to(top_weights.dtype)
        g_rows = (g32 * top_weights.float().unsqueeze(-1)).to(grad_out.dtype).view(t * k, h)
        del y, g32

        flat = top_index.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=gate_up.shape[0]).to(torch.int32)
        cu = _cu_seqlens_from_group_sizes(counts)
        x_sorted = x.unsqueeze(1).expand(t, k, h).reshape(t * k, h).index_select(0, order)
        g_sorted = g_rows.index_select(0, order)
        del g_rows

        q13, s13 = quant_weight_atom(gate_up.detach())
        q2, s2 = quant_weight_atom(down.detach())
        inter = gate_up.shape[1] // 2
        fc1 = _gemm(x_sorted, q13, s13, cu, b_transposed=True)
        act = activation_fwd(fc1, inter, "swiglu", concat_layout=True)
        d_act, d_down = _linear_backward(g_sorted, act, q2, s2, counts, cu)
        del act, g_sorted, q2
        d_fc1 = activation_bwd(fc1, d_act, inter, "swiglu", concat_layout=True)
        del fc1, d_act
        d_x_sorted, d_gate_up = _linear_backward(d_fc1, x_sorted, q13, s13, counts, cu)
        g_x = None
        if ctx.needs_input_grad[0]:
            g_x = torch.empty_like(d_x_sorted).index_copy_(0, order, d_x_sorted).view(t, k, h).sum(1)
        return g_x, d_gate_up.to(gate_up.dtype), d_down.to(down.dtype), None, g_w, None, None, None, None


class BatchInvariantFp8HFExperts(nn.Module):
    """``Qwen3MoeExperts`` replacement: same HF parameters, rollout-identical FP8 forward, Sonic FP8 backward."""

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.num_experts = int(self.gate_up_proj.shape[0])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        gate_up, down = _dense(self.gate_up_proj), _dense(self.down_proj)
        w13, s13, w2, s2 = _cached_shuffled_weights(self, gate_up, down)
        return _BatchInvariantFp8Experts.apply(
            hidden_states, gate_up, down, top_k_index, top_k_weights, w13, s13, w2, s2,
        )


def replace_hf_moe_experts_batch_invariant_fp8(model: nn.Module) -> int:
    """Swap every fused HF ``mlp.experts`` for :class:`BatchInvariantFp8HFExperts`. Returns the count."""
    replaced = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or isinstance(experts, BatchInvariantFp8HFExperts):
            continue
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue
        if not hasattr(module, "gate"):
            continue
        module.experts = BatchInvariantFp8HFExperts(experts)
        replaced += 1
    return replaced
