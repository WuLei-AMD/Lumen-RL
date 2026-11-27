"""Run ATOM's routed experts on the deterministic expanded path (LUMENRL_BATCH_INVARIANT_MOE=1).

aiter ``fused_moe`` (2-stage CK) accumulates expert outputs with atomics, so
one input run twice differs by up to 6e-2, and a token's output changes with
the batch around it. The patched unquantized FusedMoE instead runs the experts
on expanded rows with a fixed-order reduce (``lumenrl.moe.batch_invariant_experts``), on
ATOM's stock shuffled weights. Routing is ``fused_route_expand``, the kernel the
trainer's router runs in the same mode, so ids and weights match bit for bit.
Per_1x128 FP8 experts (``per_block_fp8``) take the same route through
``lumenrl.moe.batch_invariant_fp8_experts`` on ATOM's quantized, shuffled weights.
"""

import importlib.machinery
import os
import sys


def install():
    """Install the deterministic MoE patch on ATOM's unquantized and per_1x128 FP8 FusedMoE."""
    if getattr(sys, "_lumenrl_align_moe_installed", False):
        return
    sys._lumenrl_align_moe_installed = True
    patchers = {"atom.model_ops.moe": _patch_moe}
    if os.environ.get("LUMENRL_BI_ROUTER_GEMM", "0") == "1":
        patchers["atom.models.qwen3_moe"] = _patch_qwen3_moe_gate

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            patch = patchers.get(fullname)
            if patch is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _patch=patch):
                _orig(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _patch_moe(module):
    _patch_moe_batch_invariant(module)
    _patch_fp8_moe_batch_invariant(module)


def _patch_qwen3_moe_gate(module):
    """LUMENRL_BI_ROUTER_GEMM=1: every Qwen3 MoE gate computes its logits with ``router_gemm``,
    the kernel the trainer's router runs under the same flag, instead of ``tgemm.mm``."""
    from lumenrl.moe.batch_invariant_experts import router_gemm

    block = module.Qwen3MoeSparseMoeBlock
    orig_init = block.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        gate = self.gate

        def forward(x, *_args, _gate=gate, **_kwargs):
            if getattr(_gate.weight, "is_shuffled", False):
                raise RuntimeError(f"{_gate.prefix}: router_gemm reads a plain (E, K) gate weight, got a shuffled one")
            return router_gemm(x, _gate.weight)

        gate.forward = forward

    block.__init__ = __init__
    print("[lumenrl] ATOM Qwen3 MoE gates run router_gemm (LUMENRL_BI_ROUTER_GEMM=1)", flush=True)


def _plain_softmax(layer, use_grouped_topk, custom_routing_function, scoring_func, e_score_correction_bias):
    return (
        not use_grouped_topk
        and custom_routing_function is None
        and scoring_func == "softmax"
        and e_score_correction_bias is None
        and not getattr(layer, "num_fused_shared_experts", 0)
    )


def _patch_moe_batch_invariant(module):
    """LUMENRL_BATCH_INVARIANT_MOE=1: ATOM's stock shuffled weights, deterministic experts.

    Experts are ``lumenrl.moe.batch_invariant_experts.batch_invariant_experts_core`` (the kernel the trainer's
    ``BatchInvariantHFExperts`` runs in forward). Routing is ``fused_route_expand``, the kernel the
    trainer's router runs in the same mode, so ids and weights match the trainer bit for bit.
    """
    from aiter import ActivationType

    from lumenrl.moe.batch_invariant_experts import batch_invariant_experts_core, fused_route_expand

    method = module.UnquantizedFusedMoEMethod
    orig_apply = method.apply

    def apply(
        self,
        layer,
        x,
        router_logits,
        top_k,
        renormalize,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        global_num_experts=-1,
        expert_map=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        fused_shared_experts_scoring_func=None,
        apply_router_weight_on_input=False,
        activation=ActivationType.Silu,
        prefix="",
    ):
        plain_softmax = _plain_softmax(
            layer, use_grouped_topk, custom_routing_function, scoring_func, e_score_correction_bias
        )
        if getattr(self, "fused_experts", None) is not None or apply_router_weight_on_input or not plain_softmax:
            return orig_apply(
                self, layer, x, router_logits, top_k, renormalize,
                use_grouped_topk, topk_group, num_expert_group, global_num_experts,
                expert_map, custom_routing_function, scoring_func, e_score_correction_bias,
                fused_shared_experts_scoring_func, apply_router_weight_on_input, activation, prefix,
            )
        topk_weights, topk_ids, ones, x_rep = fused_route_expand(router_logits, x, top_k, renormalize)
        return batch_invariant_experts_core(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            ones=ones,
            x_rep=x_rep,
            expert_mask=layer.expert_mask,
            activation=activation,
        )

    method.apply = apply
    print("[lumenrl] ATOM unquantized FusedMoE runs deterministic expanded fused_moe (LUMENRL_BATCH_INVARIANT_MOE=1)", flush=True)


def _patch_fp8_moe_batch_invariant(module):
    """Per_1x128 ``Fp8MoEMethod``: same routing, ``batch_invariant_fp8_experts_core`` on ATOM's FP8 weights.

    The trainer's ``BatchInvariantFp8HFExperts`` quantizes its BF16 experts with ATOM's quantizer and
    shuffle, so both sides hand the pinned CK path the same bytes and scales.
    """
    from aiter import ActivationType, QuantType

    from lumenrl.moe.batch_invariant_experts import fused_route_expand
    from lumenrl.moe.batch_invariant_fp8_experts import batch_invariant_fp8_experts_core

    method = module.Fp8MoEMethod
    orig_apply = method.apply

    def apply(
        self,
        layer,
        x,
        router_logits,
        top_k,
        renormalize,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        global_num_experts=-1,
        expert_map=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        apply_router_weight_on_input=False,
        fused_shared_experts_scoring_func=None,
        activation=ActivationType.Silu,
        prefix="",
    ):
        supported = (
            self.quant_type == QuantType.per_1x128
            and getattr(self, "fused_experts", None) is None
            and not apply_router_weight_on_input
            and getattr(layer, "w13_input_scale", None) is None
            and getattr(layer, "w2_input_scale", None) is None
            and not getattr(layer, "swiglu_limit", 0.0)
            and _plain_softmax(layer, use_grouped_topk, custom_routing_function, scoring_func, e_score_correction_bias)
        )
        if not supported:
            return orig_apply(
                self, layer, x, router_logits, top_k, renormalize,
                use_grouped_topk, topk_group, num_expert_group, global_num_experts,
                expert_map, custom_routing_function, scoring_func, e_score_correction_bias,
                apply_router_weight_on_input, fused_shared_experts_scoring_func, activation, prefix,
            )
        topk_weights, topk_ids, ones, x_rep = fused_route_expand(router_logits, x, top_k, renormalize)
        return batch_invariant_fp8_experts_core(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            ones=ones,
            x_rep=x_rep,
            expert_mask=layer.expert_mask,
            activation=activation,
        )

    method.apply = apply
    print("[lumenrl] ATOM per_1x128 FP8 FusedMoE runs deterministic expanded fused_moe (LUMENRL_BATCH_INVARIANT_MOE=1)", flush=True)
