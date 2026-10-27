"""Align ATOM prefill and decode attention through AITER varlen attention.

ATOM_FORCE_ATTN_TRITON=1 sends decode through Triton RoPE and gluon paged
attention. Prefill, once its RoPE is kept on aiter, uses flash_attn_varlen.
This patch makes decode do the same two things: aiter RoPE plus a gather of
the paged cache into the varlen kernel. It does not run the assembly paged
kernel. Enable with LUMENRL_ALIGN_DECODE=1.

Full-sequence causal varlen and the q_len=1 varlen call disagree on some
prompt lengths (the 354-token step-30 prompt differs by about 0.06–0.18 on the
first response token; a 117-token prompt matches bitwise). Training scores
that token with q_len=1, so alignment keeps prefill on the same call.
``LUMENRL_ALIGN_DECODE=1`` enables it. ``LUMENRL_PREFILL_DECODE_ATTN=1`` enables
it on its own. The prefill path is one varlen launch per query row.
"""

import os
import importlib.machinery
import sys

import torch


def install():
    """Patch ATOM attention when this directory is injected into PYTHONPATH."""
    if getattr(sys, "_lumenrl_align_attention_installed", False):
        return
    sys._lumenrl_align_attention_installed = True

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != "atom.model_ops.attention_mha":
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                patch_attention(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _install_logprob_snapshot()


def _install_logprob_snapshot():
    """Materialize a sampled logprob on the host before deferred D2H.

    ``tokenIDProcessor`` queues ``gpu_logprobs.to("cpu", non_blocking=True)``
    on a side stream and reads it one step later. The source is a one-element
    gather. Cloning it and synchronizing the compute stream still leaves that
    copy outstanding: after a few hundred decode steps the caching allocator
    reuses the clone's block, and the late copy observes that recycled float.
    The align path therefore finishes a blocking host copy before returning.
    ``send_to_cpu_async`` then sees a CPU tensor, whose ``.to("cpu")`` keeps
    the same storage. Enable with ``LUMENRL_ALIGN_DECODE=1``.
    """
    busy = False

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            nonlocal busy
            if busy or fullname != "atom.model_engine.model_runner":
                return None
            busy = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                busy = False
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                processor = getattr(module, "tokenIDProcessor", None)
                if processor is None or getattr(processor, "_lumenrl_logprob_snapshot", False):
                    return
                orig_init = processor.__init__
                orig_send = processor.send_to_cpu_async

                def __init__(self, *args, **kwargs):
                    orig_init(self, *args, **kwargs)
                    if os.environ.get("LUMENRL_ALIGN_SYNC_OUTPUT", "0") == "1":
                        # Diagnostic alignment path: return this step's token id
                        # and logprob together instead of overwriting the prior
                        # step's scheduler placeholder. This removes deferred
                        # queueing and its one-step state from the comparison.
                        self.is_deferred_out = False

                def send_to_cpu_async(
                    self, gpu_tensor, cpu_tensor_handle, data_ready,
                    copy_done=None, gpu_logprobs=None,
                ):
                    if (
                        gpu_logprobs is not None
                        and os.environ.get("LUMENRL_ALIGN_DECODE", "0") == "1"
                    ):
                        ready = gpu_logprobs.detach().contiguous().clone()
                        torch.cuda.current_stream().synchronize()
                        gpu_logprobs = ready.to("cpu")
                    return orig_send(
                        self, gpu_tensor, cpu_tensor_handle, data_ready,
                        copy_done, gpu_logprobs,
                    )

                processor.__init__ = __init__
                processor.send_to_cpu_async = send_to_cpu_async
                processor._lumenrl_logprob_snapshot = True
                print("aligned decode snapshots sampled logprobs", flush=True)
                if os.environ.get("LUMENRL_ALIGN_SYNC_OUTPUT", "0") == "1":
                    print("aligned decode disables deferred output", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _aiter_rope(prefill, align):
    if os.environ.get("ATOM_FORCE_ATTN_TRITON", "0") != "1":
        return False
    return prefill or align


def patch_attention(module):
    impl = module.PagedAttentionImpl
    orig_rope = impl.rope_cache
    orig_dispatch = impl.dispatch_backend

    def rope_cache(self, q, k, v, qkv, position, fwd_ctx):
        prefill = bool(getattr(getattr(fwd_ctx, "context", None), "is_prefill", False))
        align = os.environ.get("LUMENRL_ALIGN_DECODE", "0") == "1"
        if not _aiter_rope(prefill, align):
            return orig_rope(self, q, k, v, qkv, position, fwd_ctx)
        os.environ["ATOM_FORCE_ATTN_TRITON"] = "0"
        try:
            result = orig_rope(self, q, k, v, qkv, position, fwd_ctx)
        finally:
            os.environ["ATOM_FORCE_ATTN_TRITON"] = "1"
        if (
            align
            and os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1"
        ):
            q_out, k_out, v_out, *_ = result
            rows = int(k_out.shape[0])
            if prefill:
                capacity = max(
                    rows,
                    int(os.environ.get("LUMENRL_ALIGN_CONTIG_KV_CAPACITY", "8192")),
                )
                self._lumenrl_k_history = torch.empty(
                    (capacity, *k_out.shape[1:]),
                    dtype=k_out.dtype,
                    device=k_out.device,
                )
                self._lumenrl_v_history = torch.empty(
                    (capacity, *v_out.shape[1:]),
                    dtype=v_out.dtype,
                    device=v_out.device,
                )
                self._lumenrl_k_history[:rows].copy_(k_out)
                self._lumenrl_v_history[:rows].copy_(v_out)
                self._lumenrl_k_history_len = rows
            else:
                start = int(getattr(self, "_lumenrl_k_history_len", 0))
                end = start + rows
                if end > int(self._lumenrl_k_history.shape[0]):
                    raise RuntimeError(
                        f"contiguous KV history overflow: {end} > "
                        f"{self._lumenrl_k_history.shape[0]}"
                    )
                self._lumenrl_k_history[start:end].copy_(k_out)
                self._lumenrl_v_history[start:end].copy_(v_out)
                self._lumenrl_k_history_len = end
        return result

    def paged_attention_varlen(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
        import aiter

        meta = fwd_ctx.attn_metadata
        lengths = meta.context_lens.to(dtype=torch.int32)
        nseq = int(lengths.shape[0])
        cu_k = torch.empty(nseq + 1, dtype=torch.int32, device=q.device)
        cu_k[0] = 0
        torch.cumsum(lengths, dim=0, out=cu_k[1:])
        total = int(cu_k[-1].item())
        if meta.cu_seqlens_q is not None and int(meta.cu_seqlens_q.shape[0]) == nseq + 1:
            cu_q = meta.cu_seqlens_q
        else:
            cu_q = torch.arange(nseq + 1, device=q.device, dtype=torch.int32)
        batch_id = torch.repeat_interleave(
            torch.arange(nseq, device=q.device, dtype=torch.int32),
            lengths,
        )
        seq_starts = torch.zeros(nseq, dtype=torch.int32, device=q.device)
        gather_meta = type("GatherMeta", (), {})()
        gather_meta.cu_seqlens_k = cu_k
        gather_meta.total_kv = total
        gather_meta.block_tables = meta.block_tables
        gather_meta.batch_id_per_k_token = batch_id
        gather_meta.seq_starts = seq_starts
        use_contiguous = (
            os.environ.get("LUMENRL_ALIGN_CONTIG_KV", "0") == "1"
            and nseq == 1
            and hasattr(self, "_lumenrl_k_history")
        )
        if use_contiguous:
            history_len = int(self._lumenrl_k_history_len)
            if history_len != total:
                raise RuntimeError(
                    f"contiguous KV length mismatch: history={history_len}, metadata={total}"
                )
            k_full = self._lumenrl_k_history[:history_len]
            v_full = self._lumenrl_v_history[:history_len]
        else:
            _, k_full, v_full, _, _, _, _ = self._gather_prefix_and_concat_kv(
                q, k, v, k_cache, v_cache, k_scale, v_scale, gather_meta
            )
        max_k = int(lengths.max().item())
        sliding = (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)
        result = aiter.flash_attn_varlen_func(
            q,
            k_full,
            v_full,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(meta.max_seqlen_q or 1),
            max_seqlen_k=max_k,
            min_seqlen_q=0,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=sliding,
            sink_ptr=self.sinks,
            # Training's row-wise forward needs LSE for backward, which selects
            # AITER's return-LSE variant. Request it here too so aligned rollout
            # and training use the same q_len=1 forward variant.
            return_lse=True,
        )
        return result[0]

    def prefill_attention_decode_varlen(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
    ):
        """Run prefill as one q_len=1 varlen call per query row.

        For a prefix-cache hit, ``q`` has only the new rows while the gathered
        K/V include cached and new rows.  The bottom-right causal alignment
        means query row ``j`` attends through ``prefix_len + j``.  Supplying
        exactly that K/V prefix lets q_len=1 use the same no-mask attention
        specialization that decode uses.
        """
        import aiter

        meta = fwd_ctx.attn_metadata
        if meta.has_cached:
            q, k, v, _, _, _, _ = self._gather_prefix_and_concat_kv(
                q, k, v, k_cache, v_cache, k_scale, v_scale, meta
            )

        cu_q = meta.cu_seqlens_q
        cu_k = meta.cu_seqlens_k
        out = torch.empty_like(q)
        one = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
        sliding = (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)

        for seq in range(int(cu_q.numel()) - 1):
            q_start, q_end = (int(x) for x in cu_q[seq : seq + 2].tolist())
            k_start, k_end = (int(x) for x in cu_k[seq : seq + 2].tolist())
            prefix_len = k_end - k_start - (q_end - q_start)
            for row, q_index in enumerate(range(q_start, q_end)):
                key_end = k_start + prefix_len + row + 1
                result = aiter.flash_attn_varlen_func(
                    q[q_index : q_index + 1],
                    k[k_start:key_end],
                    v[k_start:key_end],
                    cu_seqlens_q=one,
                    cu_seqlens_k=torch.tensor(
                        [0, key_end - k_start], dtype=torch.int32, device=q.device
                    ),
                    max_seqlen_q=1,
                    max_seqlen_k=key_end - k_start,
                    min_seqlen_q=0,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=sliding,
                    sink_ptr=self.sinks,
                    return_lse=True,
                )
                out[q_index : q_index + 1] = result[0]
        return out

    def _use_qlen1_prefill():
        return (
            os.environ.get("LUMENRL_ALIGN_DECODE", "0") == "1"
            or os.environ.get("LUMENRL_PREFILL_DECODE_ATTN", "0") == "1"
        )

    def dispatch_backend(self, fwd_ctx, q, k, v):
        prefill = bool(getattr(getattr(fwd_ctx, "context", None), "is_prefill", False))
        if prefill:
            if _use_qlen1_prefill():
                return self.prefill_attention_decode_varlen
            return orig_dispatch(self, fwd_ctx, q, k, v)
        if os.environ.get("LUMENRL_ALIGN_DECODE", "0") != "1":
            return orig_dispatch(self, fwd_ctx, q, k, v)
        return self.paged_attention_varlen

    impl.rope_cache = rope_cache
    impl.paged_attention_varlen = paged_attention_varlen
    impl.prefill_attention_decode_varlen = prefill_attention_decode_varlen
    impl.dispatch_backend = dispatch_backend
    print(
        "aligned attention uses aiter RoPE and q_len=1 flash_attn_varlen "
        "for prefill and decode when LUMENRL_ALIGN_DECODE=1",
        flush=True,
    )
