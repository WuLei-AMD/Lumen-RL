"""Frozen reference policy for KL and baseline log-prob targets."""

from __future__ import annotations

import logging
from typing import Any

from lumenrl.workers.base_worker import BaseWorker, get_nested_config

import torch

from lumenrl.core.protocol import DataProto
from lumenrl.engine.training.fsdp_backend import FSDP2Backend

logger = logging.getLogger(__name__)


class RefPolicyWorker(BaseWorker):
    """Reference model in eval mode for stable log-probability estimates."""

    def __init__(self, rank: int, world_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(rank, world_size, config)
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self) -> None:
        """Load a frozen reference policy (same architecture path as actor)."""
        policy = get_nested_config(self.config, "policy", default={}) or {}
        training_cfg = policy.get("training", {}) or {}
        self._model = FSDP2Backend.build_model(str(policy.get("model_name", "")), training_cfg)
        self._model = FSDP2Backend.apply_lumen_optimizations(
            self._model,
            get_nested_config(self.config, "quantization", "training", default={}) or {},
        )
        self._model.to(self._device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        self._log.info("RefPolicyWorker: frozen reference ready.")

    def compute_log_probs(self, batch: DataProto) -> DataProto:
        """Return detached reference log-probs on the actor's packed forward.

        The padded ``log_softmax`` path scores left-pad tokens and keeps the
        extra lm_head columns. Actor old/train log-probs use packed varlen
        attention and, when set, ``LUMENRL_TRAIN_VOCAB_MASK`` plus q_len=1
        attention. The KL term has to use that same denominator and reduction.
        """
        if self._model is None:
            raise RuntimeError("init_model() must be called before compute_log_probs().")
        if "input_ids" not in batch.tensors:
            raise KeyError("batch must contain 'input_ids'")

        from lumenrl.engine.training.packing import (
            PackingContext,
            pack_sequences,
            packed_token_log_probs,
            unpack_log_probs,
        )

        sequences = batch["input_ids"].to(self._device)
        batch_size, seqlen = sequences.shape
        if batch_size == 0 or seqlen == 0:
            empty = torch.zeros(batch_size, max(seqlen - 1, 0), dtype=torch.float32)
            return DataProto(
                tensors={"ref_log_probs": empty, "input_ids": batch["input_ids"]},
                meta=dict(batch.meta),
            )
        attention_mask = batch.tensors.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = torch.ones_like(sequences)
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        policy = get_nested_config(self.config, "policy", default={}) or {}
        try:
            max_tokens = int(policy.get("max_token_len_per_gpu", 0) or 0)
        except (TypeError, ValueError):
            max_tokens = 0
        row_lens = attention_mask.sum(dim=1).long().tolist()
        chunks: list[tuple[int, int]] = []
        if max_tokens > 0:
            start = 0
            while start < batch_size:
                tokens, end = 0, start
                while end < batch_size:
                    length = int(row_lens[end])
                    if tokens + length > max_tokens and end > start:
                        break
                    tokens += length
                    end += 1
                chunks.append((start, end))
                start = end
        else:
            chunks = [(row, row + 1) for row in range(batch_size)]

        import torch.distributed as dist

        n_rows = len(chunks)
        if dist.is_initialized() and dist.get_world_size() > 1:
            count = torch.tensor([n_rows], device=self._device)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            n_iters = int(count.item())
        else:
            n_iters = n_rows

        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for index in range(n_iters):
                lo, hi = chunks[min(index, n_rows - 1)]
                packed = pack_sequences(sequences[lo:hi], attention_mask[lo:hi])
                with PackingContext(packed.cu_seqlens, packed.max_seqlen):
                    outputs = self._model(
                        input_ids=packed.input_ids,
                        position_ids=packed.position_ids,
                        attention_mask=None,
                        use_cache=False,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    flat = packed_token_log_probs(
                        logits.squeeze(0),
                        packed.input_ids.squeeze(0),
                        packed.cu_seqlens,
                        temperature=temperature,
                    )
                    token_log_probs = unpack_log_probs(
                        flat, packed.cu_seqlens, packed.seq_lens, seqlen
                    )
                if index < n_rows:
                    parts.append(token_log_probs)
                del outputs, logits

        log_probs = torch.cat(parts, dim=0)
        return DataProto(
            tensors={
                "ref_log_probs": log_probs.cpu(),
                "input_ids": batch["input_ids"],
            },
            meta=dict(batch.meta),
        )

    def cleanup(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().cleanup()
