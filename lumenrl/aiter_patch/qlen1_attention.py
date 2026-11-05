"""Decode-shaped attention for many query rows in one launch.

ATOM decode scores a token with one q_len=1 ``flash_attn_varlen_func`` call:
one query row over its key prefix. Prefill and training reproduce that value
bit for bit only through the same q_len=1 specialization, which costs one
launch per token per layer when done row by row. This runs every row in one
call while keeping each row's arithmetic:

- ``cu_seqlens_k_padded`` holds each row's physical key start and
  ``cu_seqlens_k`` its real length, so rows read overlapping prefixes of one
  K/V buffer without copying them.
- The query heads that share a KV head attend to the same keys, so they enter
  as the rows of one non-causal sequence of length ``hq // hk``. K/V is then
  read once per KV head instead of once per query head.

Both were checked bit for bit against the one-row calls at lengths 127 to 4096
(``atom_stability/test_qlen1_batched_attn.py``).

The launch goes through ``FlashAttnVarlenFunc`` rather than
``flash_attn_varlen_func``. The public wrapper tries FlyDSL first, and that
kernel does not read ``cu_seqlens_k_padded``. On gfx1250 it would treat the
cumsum of row lengths as contiguous K and score the wrong prefixes. CK and
fmha v3 both receive the physical starts.
"""

import torch


def _one_row_prefixes(q, k, v, row_starts, row_lens, softmax_scale, window_size, sink_ptr):
    """Score each query row on its own key slice.

    Used when CK is off. Slicing the prefix keeps the arithmetic of one
    q_len=1 call. The packed launch is unavailable there because the Triton
    varlen fallback ignores physical key starts.
    """
    from aiter import flash_attn_varlen_func

    device = q.device
    one = torch.tensor([0, 1], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    starts = row_starts.tolist()
    lens = row_lens.tolist()
    for index, (start, length) in enumerate(zip(starts, lens)):
        start = int(start)
        length = int(length)
        cu_k = torch.tensor([0, length], dtype=torch.int32, device=device)
        result = flash_attn_varlen_func(
            q[index : index + 1],
            k[start : start + length],
            v[start : start + length],
            cu_seqlens_q=one,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=length,
            min_seqlen_q=0,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            sink_ptr=sink_ptr,
            return_lse=True,
        )
        out[index : index + 1] = result[0] if isinstance(result, tuple) else result
    return out


def _prefix_varlen(
    grouped,
    k,
    v,
    cu_q,
    cu_k,
    starts,
    max_seqlen_q,
    max_len,
    softmax_scale,
    window_size,
    sink_ptr,
):
    """One varlen launch that honors ``starts`` as each row's physical key offset."""
    from aiter.ops.mha import FlashAttnVarlenFunc

    return FlashAttnVarlenFunc.apply(
        grouped,
        k,
        v,
        cu_q,
        cu_k,
        int(max_seqlen_q),
        int(max_len),
        0,
        0.0,
        softmax_scale,
        0.0,
        False,
        window_size,
        None,
        None,
        False,
        True,
        False,
        None,
        None,
        torch.is_grad_enabled(),
        None,
        starts,
        True,
        1,
        sink_ptr,
        None,
        None,
        None,
    )


def row_prefix_attention(
    q,
    k,
    v,
    row_starts,
    row_lens,
    max_len,
    softmax_scale,
    window_size=(-1, -1, 0),
    sink_ptr=None,
):
    """Attention of query row ``i`` over ``k[row_starts[i] : row_starts[i] + row_lens[i]]``.

    ``q`` is ``(rows, hq, d)``; ``k``/``v`` are ``(keys, hk, d)``.
    ``row_starts`` and ``row_lens`` are int32 device tensors of length ``rows``.
    ``max_len`` is the host-side maximum of ``row_lens``.
    """
    rows, hq, dim = q.shape
    if rows == 0:
        return q
    from aiter.ops.mha import ENABLE_CK

    if not ENABLE_CK:
        return _one_row_prefixes(
            q, k, v, row_starts, row_lens, softmax_scale, window_size, sink_ptr
        )
    hk = k.shape[1]
    if hk == 0 or hq % hk != 0:
        raise RuntimeError(f"query heads {hq} must be a positive multiple of KV heads {hk}")
    rep = hq // hk
    device = q.device
    cu_k = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(row_lens, 0, out=cu_k[1:])
    starts = torch.empty(rows + 1, dtype=torch.int32, device=device)
    starts[:rows] = row_starts
    starts[rows] = k.shape[0]
    cu_q = torch.arange(0, rep * rows + 1, rep, dtype=torch.int32, device=device)
    grouped = q.contiguous().view(rows, hk, rep, dim).transpose(1, 2).reshape(rows * rep, hk, dim)
    result = _prefix_varlen(
        grouped,
        k,
        v,
        cu_q,
        cu_k,
        starts,
        rep,
        max_len,
        softmax_scale,
        window_size,
        sink_ptr,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out.view(rows, rep, hk, dim).transpose(1, 2).reshape(rows, hq, dim)


def packed_causal_rows(cu_seqlens, total):
    """Row starts and lengths of causal attention over packed sequences.

    Row ``t`` of sequence ``s`` attends to ``[cu_seqlens[s], t]``. Everything is
    computed on the device, so no host synchronization is needed.
    """
    device = cu_seqlens.device
    index = torch.arange(total, dtype=torch.int32, device=device)
    seq = torch.searchsorted(cu_seqlens[1:].to(torch.int32), index, right=True)
    starts = cu_seqlens.to(torch.int32)[seq]
    return starts, index - starts + 1
