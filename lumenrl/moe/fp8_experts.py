"""Blockwise FP8 HF MoE experts for the trainer (``LUMENRL_FP8_MOE=1``).

The official Qwen3-FP8 recipe on SonicMoE grouped GEMMs, run on the HF parameter layout
(``gate_up_proj`` (E, 2I, H), ``down_proj`` (E, H, I)) so FSDP2 sharding, ``state_dict`` and
weight export are unchanged. Each weight is quantized once per forward with 128x128 blocks;
the blocks are square, so the same FP8 tensor serves the forward GEMM (B transposed) and the
dgrad GEMM (B as stored) without a layout copy. Activations are 1x128 along K; wgrad quantizes
each expert's token segment along M. SwiGLU stays BF16.

The (token, slot) assignments are stable-sorted by expert id, run expert-major, gathered back
to (token, slot) order and summed over the slots in fp32 in slot order.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lumenrl.moe.batch_invariant_experts import _dense

_BLOCK = 128


def _fp8_dtype():
    from lumen.ops.gemm.grouped_gemm import _default_fp8_dtype

    return _default_fp8_dtype()


def _gemm(a, w8, w_scale, cu, b_transposed):
    """a (TK, K) BF16 x per-expert w8 -> (TK, N). ``b_transposed``: w8 is (E, N, K)."""
    from aiter.ops.triton.moe.sonicmoe import grouped_gemm
    from lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    a8, a_scale = quant_fp8_blockwise_impl(a, _fp8_dtype(), axis=1, block_size=_BLOCK)
    return grouped_gemm(
        a8, w8, cu, B_is_transposed=b_transposed, A_scale=a_scale, B_scale=w_scale,
        block_size=_BLOCK, out_dtype=torch.bfloat16,
    )


def _linear_backward(grad, inp, w8, w_scale, counts, cu):
    """y = inp @ W.T with W (E, N, K): returns (d_inp, dW in W's layout)."""
    from lumen.ops.gemm.grouped_gemm import _sonic_fp8_wgrad_direct
    from lumen.ops.quantize.ops import quant_fp8_blockwise_segment_m_dual

    d_inp = _gemm(grad, w8, w_scale, cu, b_transposed=False)
    g_col, g_scale, i_col, i_scale = quant_fp8_blockwise_segment_m_dual(
        grad, inp, counts.numel(), counts, cu, _fp8_dtype(), _BLOCK
    )
    d_w = _sonic_fp8_wgrad_direct(g_col, i_col, cu, g_scale, i_scale, _BLOCK)
    if d_w is None:
        raise RuntimeError("this aiter build has no SonicMoE _grouped_gemm_dw_kernel")
    return d_inp, d_w


class _Fp8ExpertMlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_up, down, counts):
        from aiter.ops.triton.moe.sonicmoe import activation_fwd
        from lumen.ops.gemm.grouped_gemm import _cu_seqlens_from_group_sizes
        from lumen.ops.quantize.ops import quant_fp8_blockwise_weight_3d

        cu = _cu_seqlens_from_group_sizes(counts)
        gu8, gu_scale = quant_fp8_blockwise_weight_3d(gate_up, _fp8_dtype(), _BLOCK)
        dn8, dn_scale = quant_fp8_blockwise_weight_3d(down, _fp8_dtype(), _BLOCK)
        fc1 = _gemm(x, gu8, gu_scale, cu, b_transposed=True)
        inter = gate_up.shape[1] // 2
        act = activation_fwd(fc1, inter, "swiglu", concat_layout=True)
        out = _gemm(act, dn8, dn_scale, cu, b_transposed=True)
        ctx.save_for_backward(x, fc1, act, gu8, gu_scale, dn8, dn_scale, counts, cu)
        ctx.inter = inter
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from aiter.ops.triton.moe.sonicmoe import activation_bwd

        x, fc1, act, gu8, gu_scale, dn8, dn_scale, counts, cu = ctx.saved_tensors
        d_act, d_down = _linear_backward(grad_out.contiguous(), act, dn8, dn_scale, counts, cu)
        d_fc1 = activation_bwd(fc1, d_act, ctx.inter, "swiglu", concat_layout=True)
        d_x, d_gate_up = _linear_backward(d_fc1, x, gu8, gu_scale, counts, cu)
        return d_x, d_gate_up, d_down, None


def fp8_experts_forward(hidden_states, top_k_index, top_k_weights, gate_up, down):
    """Differentiable blockwise FP8 experts on HF-layout ``gate_up`` (E, 2I, H) and ``down`` (E, H, I)."""
    t, k = top_k_index.shape
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=gate_up.shape[0]).to(torch.int32)
    # Expand, then permute: the backward is a sum over slots plus a scatter to unique rows,
    # with no atomics.
    x = hidden_states.unsqueeze(1).expand(t, k, hidden_states.shape[-1]).reshape(t * k, -1).index_select(0, order)
    y = _Fp8ExpertMlp.apply(x, gate_up, down, counts)
    y = y.index_select(0, torch.argsort(order)).view(t, k, -1)
    out = (y.float() * top_k_weights.unsqueeze(-1).float()).sum(dim=1)
    return out.to(hidden_states.dtype)


class Fp8HFExperts(nn.Module):
    """``Qwen3MoeExperts`` replacement: same HF parameters, blockwise FP8 grouped GEMMs."""

    def __init__(self, experts: nn.Module):
        super().__init__()
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.num_experts = int(self.gate_up_proj.shape[0])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        return fp8_experts_forward(
            hidden_states, top_k_index, top_k_weights, _dense(self.gate_up_proj), _dense(self.down_proj)
        )


def replace_hf_moe_experts_fp8(model: nn.Module) -> int:
    """Swap every fused HF ``mlp.experts`` for :class:`Fp8HFExperts`. Returns the count."""
    replaced = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or isinstance(experts, Fp8HFExperts):
            continue
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue
        if not hasattr(module, "gate"):
            continue
        module.experts = Fp8HFExperts(experts)
        replaced += 1
    return replaced
