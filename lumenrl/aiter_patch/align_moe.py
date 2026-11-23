"""Run ATOM's BF16 routed experts on the trainer's SonicMoE path.

aiter ``fused_moe`` (2-stage CK) accumulates expert outputs with atomics, so
one input run twice differs by up to 6e-2, and a token's output changes with
the batch around it. No trainer can match that. With ``LUMENRL_ALIGN_MOE=1``
the unquantized FusedMoE instead runs Lumen's ``sonic_experts_forward``, the
function the trainer's ``SonicHFExperts`` calls (``LUMENRL_SONIC_MOE=1``),
which is deterministic and independent of the token count. Routing is the HF
Qwen3-MoE router math on ATOM's gate logits: fp32 softmax, top-k, renormalize,
cast to the logits dtype. ``topk_softmax`` picks the same experts in a
different order, and the reduction runs in that order.

The expert buffers keep their shape, so ATOM's loader and weight updater still
match parameters by shape. Their bytes are Sonic's layout: ``w13_weight``
(E, 2I, H) holds ``w1`` (E, H, 2I) and ``w2_weight`` (E, H, I) holds ``w2``
(E, I, H). That conversion replaces aiter's shuffle both after the initial
load and in the updater's per-expert relayout.
"""

import importlib.machinery
import sys

_SONIC_ATTR = "_lumen_sonic_layout"


def install():
    if getattr(sys, "_lumenrl_align_moe_installed", False):
        return
    sys._lumenrl_align_moe_installed = True
    patchers = {
        "atom.model_ops.moe": _patch_moe,
        "atom.model_ops.utils": _patch_utils,
    }

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


def _to_sonic(param, kind, expert_ids):
    from lumen.modules.sonic_hf_experts import down_to_sonic_, gate_up_to_sonic_

    convert = gate_up_to_sonic_ if kind == "w13" else down_to_sonic_
    convert(param.data, expert_ids)
    setattr(param, _SONIC_ATTR, kind)


def _patch_utils(module):
    orig = module.shuffle_expert_slices

    def shuffle_expert_slices(tensor, expert_ids, layout=(16, 16)):
        kind = getattr(tensor, _SONIC_ATTR, None)
        if kind is None:
            return orig(tensor, expert_ids, layout)
        _to_sonic(tensor, kind, list(expert_ids))

    module.shuffle_expert_slices = shuffle_expert_slices


def _patch_moe(module):
    import torch
    import torch.nn.functional as F

    method = module.UnquantizedFusedMoEMethod
    orig_process = method.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        def to_sonic(w13, w2, layout=(16, 16)):
            _to_sonic(w13, "w13", range(w13.shape[0]))
            _to_sonic(w2, "w2", range(w2.shape[0]))

        orig_shuffle = module.shuffle_weights
        module.shuffle_weights = to_sonic
        try:
            orig_process(self, layer)
        finally:
            module.shuffle_weights = orig_shuffle

    def apply(self, layer, x, router_logits, top_k, renormalize, *args, **kwargs):
        from lumen.modules.sonic_hf_experts import sonic_experts_forward
        from lumen.ops.moe.sonic_configs import install_qwen3_sonic_configs

        install_qwen3_sonic_configs()
        # Qwen3MoeTopKRouter.forward, as the trainer's router patch runs it.
        probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_value, top_index = torch.topk(probs, top_k, dim=-1)
        if renormalize:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        top_value = top_value.to(router_logits.dtype)
        w13, w2 = layer.w13_weight, layer.w2_weight
        num_experts, two_i, hidden = w13.shape
        return sonic_experts_forward(
            x,
            top_index,
            top_value,
            w13.view(num_experts, hidden, two_i),
            w2.view(num_experts, two_i // 2, hidden),
        )

    method.process_weights_after_loading = process_weights_after_loading
    method.apply = apply
    print("[lumenrl] ATOM unquantized FusedMoE runs SonicMoE experts (LUMENRL_ALIGN_MOE=1)", flush=True)
