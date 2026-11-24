"""Deterministic, batch-invariant BF16 MoE experts shared by the rollout and the trainer.

Routing (``fused_route``): one Triton kernel per call does top-k on the gate logits, the
softmax weights, the renormalization and the cast, and writes int32 expert ids plus the
all-ones weights the expanded CK call needs. Ties go to the lower expert id. Both sides run
this kernel on the same logits, row by row, so ids and weights match bit for bit; it does not
try to reproduce torch's softmax bits. With renormalization the top-k weights are exactly a
softmax over the selected logits, which is also what the backward differentiates.

Experts (``batch_invariant_experts_core``): aiter CK 2-stage ``fused_moe`` on expanded rows (each
(token, slot) is its own row, topk=1, weight 1, so stage2 writes every output element once),
then ``_expanded_reduce`` sums the top-k rows of each token in fixed slot order in fp32. This
is the kernel ATOM runs with ``LUMENRL_BATCH_INVARIANT_MOE=1``; it needs ATOM's shuffled layout.

Trainer (``BatchInvariantHFExperts``): forward is ``batch_invariant_experts_core`` on ATOM-shuffled copies of the HF
parameters; backward is Lumen's SonicMoE autograd on the same parameters (there is no CK
backward for BF16 MoE). Parameters stay in HF layout (``gate_up_proj`` (E, 2I, H),
``down_proj`` (E, H, I)), so FSDP2 sharding, ``state_dict`` and weight export are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

_REDUCE_BLOCK_H = 1024


def _dense(weight: torch.Tensor) -> torch.Tensor:
    """Unwrap FSDP2/DTensor views to a dense tensor."""
    to_local = getattr(weight, "to_local", None)
    if not callable(to_local):
        return weight
    local = to_local()
    if tuple(local.shape) == tuple(weight.shape):
        return local
    return weight.full_tensor()


@triton.jit
def _route_kernel(logits_ptr, w_ptr, ids_ptr, ones_ptr, stride_row,
                  x_ptr, xrep_ptr, H,
                  E: tl.constexpr, K: tl.constexpr, NORM: tl.constexpr,
                  EXPAND: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    if EXPAND:
        # The expanded CK call wants x repeated K times along rows (token t -> rows t*K..t*K+K-1).
        # Doing the copy here saves a separate repeat_interleave kernel per MoE layer.
        for h0 in range(0, H, BLOCK_H):
            hoffs = h0 + tl.arange(0, BLOCK_H)
            hmask = hoffs < H
            xv = tl.load(x_ptr + row * H + hoffs, mask=hmask, other=0.0)
            for j in tl.static_range(K):
                tl.store(xrep_ptr + (row * K + j) * H + hoffs, xv, mask=hmask)
    offs = tl.arange(0, E)
    x0 = tl.load(logits_ptr + row * stride_row + offs).to(tl.float32)
    x = x0
    ko = tl.arange(0, K)
    vals = tl.zeros([K], dtype=tl.float32)
    idxs = tl.zeros([K], dtype=tl.int32)
    for j in tl.static_range(K):
        mx = tl.max(x, axis=0)
        ix = tl.min(tl.where(x == mx, offs, E), axis=0)
        vals = tl.where(ko == j, mx, vals)
        idxs = tl.where(ko == j, ix, idxs)
        x = tl.where(offs == ix, float("-inf"), x)
    top = tl.max(vals, axis=0)
    e = tl.exp(vals - top)
    if NORM:
        denom = tl.sum(e, axis=0)
    else:
        denom = tl.sum(tl.exp(x0 - top), axis=0)
    w = e / denom
    tl.store(w_ptr + row * K + ko, w.to(w_ptr.dtype.element_ty))
    tl.store(ids_ptr + row * K + ko, idxs)
    tl.store(ones_ptr + row * K + ko, tl.full([K], 1.0, dtype=tl.float32))


def _route_launch(logits: torch.Tensor, top_k: int, renormalize: bool, x: torch.Tensor | None = None):
    logits = logits.contiguous()
    m, e = logits.shape
    w = torch.empty(m, top_k, dtype=logits.dtype, device=logits.device)
    ids = torch.empty(m, top_k, dtype=torch.int32, device=logits.device)
    ones = torch.empty(m * top_k, 1, dtype=torch.float32, device=logits.device)
    x_rep = None
    if x is not None:
        x = x.contiguous()
        x_rep = torch.empty(m * top_k, x.shape[1], dtype=x.dtype, device=x.device)
    if m:
        _route_kernel[(m,)](
            logits, w, ids, ones, logits.stride(0),
            x if x is not None else logits, x_rep if x_rep is not None else logits,
            x.shape[1] if x is not None else 0,
            E=e, K=top_k, NORM=renormalize, EXPAND=x is not None, BLOCK_H=1024,
            # One row per program with E=128 and K=8: 1 warp is ~3 us/launch faster than the default 4
            # (logs/bench-route-reduce-tune.log) and bitwise identical; rollout and trainer share this launch.
            num_warps=1,
        )
    return w, ids, ones, x_rep


class _FusedRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, top_k, renormalize):
        w, ids, ones, _ = _route_launch(logits, top_k, renormalize)
        ctx.save_for_backward(logits, ids)
        ctx.renormalize = renormalize
        ctx.mark_non_differentiable(ids, ones)
        return w, ids, ones

    @staticmethod
    def backward(ctx, grad_w, _grad_ids, _grad_ones):
        logits, ids = ctx.saved_tensors
        lf = logits.float()
        idx = ids.long()
        g = grad_w.float()
        if ctx.renormalize:
            w = torch.softmax(lf.gather(1, idx), dim=-1)
            g_sel = w * (g - (w * g).sum(dim=-1, keepdim=True))
            g_logits = torch.zeros_like(lf).scatter_(1, idx, g_sel)
        else:
            p = torch.softmax(lf, dim=-1)
            wg = p.gather(1, idx) * g
            g_logits = torch.zeros_like(lf).scatter_(1, idx, wg) - p * wg.sum(dim=-1, keepdim=True)
        return g_logits.to(logits.dtype), None, None


def fused_route(router_logits: torch.Tensor, top_k: int, renormalize: bool = True):
    """Top-k routing on gate logits (M, E). Returns (weights (M, k) in the logits dtype,
    int32 ids (M, k), fp32 ones (M*k, 1)). E must be a power of two."""
    if torch.is_grad_enabled() and router_logits.requires_grad:
        return _FusedRoute.apply(router_logits, top_k, renormalize)
    w, ids, ones, _ = _route_launch(router_logits, top_k, renormalize)
    return w, ids, ones


def fused_route_expand(router_logits: torch.Tensor, x: torch.Tensor, top_k: int, renormalize: bool = True):
    """Inference-only: ``fused_route`` plus the expanded activations ``x`` repeated top_k times
    per token (row t*k+j = x[t]), written by the same kernel. Returns (weights, ids, ones, x_rep)."""
    return _route_launch(router_logits, top_k, renormalize, x=x)


@triton.jit
def _expanded_reduce_kernel(y_ptr, w_ptr, out_ptr, H, K: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j in tl.static_range(K):
        wj = tl.load(w_ptr + row * K + j).to(tl.float32)
        yj = tl.load(y_ptr + (row * K + j) * H + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wj * yj
    tl.store(out_ptr + row * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _expanded_reduce(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """out[m] = sum_j weights[m, j] * y[m*k + j], fp32 accumulation in slot order."""
    m, k = weights.shape
    h = y.shape[-1]
    y = y.contiguous()
    weights = weights.contiguous()
    out = torch.empty(m, h, dtype=y.dtype, device=y.device)
    if m:
        grid = (m, triton.cdiv(h, _REDUCE_BLOCK_H))
        _expanded_reduce_kernel[grid](y, weights, out, h, K=k, BLOCK_H=_REDUCE_BLOCK_H)
    return out


def batch_invariant_experts_core(x, w13, w2, top_weights, top_index, ones=None, expert_mask=None, activation=None, x_rep=None):
    """Expanded-row CK experts + fixed-order reduce. ``w13``/``w2`` are ATOM-shuffled.
    ``ones`` is the (M*k, 1) fp32 all-ones buffer from ``fused_route`` (allocated if None).
    ``x_rep`` is ``x`` repeated top_k times per token, if the route kernel already wrote it."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe as aiter_fused_moe

    if activation is None:
        activation = ActivationType.Silu
    m, k = top_index.shape
    ids = top_index.to(torch.int32).contiguous().view(m * k, 1)
    if ones is None:
        ones = torch.ones(m * k, 1, dtype=torch.float32, device=x.device)
    if x_rep is None:
        x_rep = x.repeat_interleave(k, dim=0)
    y = aiter_fused_moe(
        x_rep,
        w13,
        w2,
        ones,
        ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=QuantType.No,
    )
    return _expanded_reduce(y, top_weights)


class _BatchInvariantExperts(torch.autograd.Function):
    """Forward: CK expanded experts on ATOM-shuffled copies of the HF parameters.
    Backward: SonicMoE autograd on the same parameters, recomputed from the saved inputs."""

    @staticmethod
    def forward(ctx, x, gate_up, down, top_index, top_weights):
        from aiter.ops.shuffle import shuffle_weight

        x = x.contiguous()
        w13 = shuffle_weight(gate_up.contiguous(), layout=(16, 16))
        w2 = shuffle_weight(down.contiguous(), layout=(16, 16))
        out = batch_invariant_experts_core(x, w13, w2, top_weights, top_index)
        ctx.save_for_backward(x, gate_up, down, top_index, top_weights)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from lumen.modules.sonic_hf_experts import _pack_gate_up, sonic_experts_forward

        x, gate_up, down, top_index, top_weights = ctx.saved_tensors
        need_x, need_gu, need_dn, _, need_w = ctx.needs_input_grad
        with torch.enable_grad():
            xg = x.detach().requires_grad_(need_x)
            gu = gate_up.detach().requires_grad_(True)
            dn = down.detach().requires_grad_(True)
            wg = top_weights.detach().requires_grad_(need_w)
            out = sonic_experts_forward(
                xg, top_index.long(), wg, _pack_gate_up(gu), dn.transpose(1, 2).contiguous()
            )
            wrt = [t for t, need in ((xg, need_x), (gu, True), (dn, True), (wg, need_w)) if need]
            grads = iter(torch.autograd.grad(out, wrt, grad_out.contiguous()))
        g_x = next(grads) if need_x else None
        g_gu = next(grads)
        g_dn = next(grads)
        g_w = next(grads).to(top_weights.dtype) if need_w else None
        return g_x, g_gu, g_dn, None, g_w


class BatchInvariantHFExperts(nn.Module):
    """``Qwen3MoeExperts`` replacement: same HF parameters, deterministic CK forward, Sonic backward."""

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.num_experts = int(self.gate_up_proj.shape[0])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        return _BatchInvariantExperts.apply(
            hidden_states,
            _dense(self.gate_up_proj),
            _dense(self.down_proj),
            top_k_index,
            top_k_weights,
        )


def replace_hf_moe_experts_batch_invariant(model: nn.Module) -> int:
    """Swap every fused HF ``mlp.experts`` for :class:`BatchInvariantHFExperts`. Returns the count."""
    replaced = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or isinstance(experts, BatchInvariantHFExperts):
            continue
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue
        if not hasattr(module, "gate"):
            continue
        module.experts = BatchInvariantHFExperts(experts)
        replaced += 1
    return replaced
