"""router_gemm: each row's logits are bit-identical whatever M or the surrounding rows."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(m=4096, k=2048, n=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    return x, w


def test_rows_invariant_to_m_and_offset():
    from lumenrl.moe.batch_invariant_experts import router_gemm

    x, w = _inputs()
    ref = router_gemm(x, w)
    for m in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 128, 255, 256, 511, 1000, 2048):
        for start in (0, 1, 13, 4096 - m):
            out = router_gemm(x[start:start + m], w)
            assert torch.equal(out, ref[start:start + m]), (m, start)
    perm = torch.randperm(x.shape[0], device="cuda")
    assert torch.equal(router_gemm(x[perm], w), ref[perm])


def test_matches_fp32_reference():
    from lumenrl.moe.batch_invariant_experts import router_gemm

    x, w = _inputs()
    ref = F.linear(x.float(), w.float())
    err = (router_gemm(x, w).float() - ref).abs().max().item()
    assert err <= ref.abs().max().item() * 2 ** -7


def test_cuda_graph_replay_matches_eager():
    from lumenrl.moe.batch_invariant_experts import router_gemm

    x, w = _inputs()
    buf = x[:16].clone()
    router_gemm(buf, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = router_gemm(buf, w)
    buf.copy_(x[100:116])
    graph.replay()
    assert torch.equal(out, router_gemm(x, w)[100:116])


def test_backward_matches_linear():
    from lumenrl.moe.batch_invariant_experts import router_gemm

    x, w = _inputs(m=512)
    x1, w1 = x.clone().requires_grad_(), w.clone().requires_grad_()
    x2, w2 = x.clone().requires_grad_(), w.clone().requires_grad_()
    g = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    router_gemm(x1, w1).backward(g)
    F.linear(x2, w2).backward(g)
    torch.testing.assert_close(x1.grad, x2.grad)
    torch.testing.assert_close(w1.grad, w2.grad)
