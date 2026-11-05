"""Regression coverage for repeated actor/ref packing-patch initialization."""

import sys
import types


def test_attention_patch_is_idempotent(monkeypatch):
    registry = {"sdpa": object()}
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    modeling_utils.ALL_ATTENTION_FUNCTIONS = registry
    transformers = types.ModuleType("transformers")

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)

    from lumenrl.engine.training import packing

    original = registry["sdpa"]
    saved_original = packing._original_attn_fn
    try:
        packing._original_attn_fn = None
        assert packing.patch_attention_for_packing()
        assert registry["sdpa"] is packing._varlen_attention_forward
        assert packing._original_attn_fn is original

        assert packing.patch_attention_for_packing()
        assert registry["sdpa"] is packing._varlen_attention_forward
        assert packing._original_attn_fn is original
    finally:
        packing._original_attn_fn = saved_original


def test_rowwise_attention_requires_zero_dropout(monkeypatch):
    from lumenrl.engine.training import packing

    monkeypatch.setenv("LUMENRL_ROWWISE_TRAIN_ATTN", "1")
    packing._rowwise_dropout_warned = False
    assert packing._use_rowwise_attention(0.0)
    assert not packing._use_rowwise_attention(0.1)
    assert not packing._use_rowwise_attention(0.1)
    assert packing._rowwise_dropout_warned


def test_vocab_tail_mask_does_not_write_the_source(monkeypatch):
    import torch

    from lumenrl.engine.training.packing import _mask_vocab_tail

    monkeypatch.setenv("LUMENRL_TRAIN_VOCAB_MASK", "3")
    source = torch.arange(8, dtype=torch.float32).view(1, 8)
    masked = _mask_vocab_tail(source)
    assert torch.equal(source, torch.arange(8, dtype=torch.float32).view(1, 8))
    assert torch.equal(masked[0, :3], source[0, :3])
    assert torch.isneginf(masked[0, 3:]).all()


def test_entropy_stays_finite_with_vocab_mask(monkeypatch):
    import torch

    from lumenrl.engine.training.packing import packed_token_entropy

    monkeypatch.setenv("LUMENRL_TRAIN_VOCAB_MASK", "4")
    logits = torch.randn(3, 8)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    entropy = packed_token_entropy(logits, cu)
    assert entropy.numel() == 2
    assert torch.isfinite(entropy).all()
