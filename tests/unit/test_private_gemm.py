"""LUMENRL_PRIVATE_GEMM selects the public table without touching other paths."""

import os

from lumenrl.engine.private_gemm import private_bf16_gemm_table, use_private_bf16_gemm_table


def test_zero_drops_the_shipped_table_and_keeps_a_custom_path(monkeypatch):
    shipped = private_bf16_gemm_table()
    assert shipped
    monkeypatch.setenv("LUMENRL_PRIVATE_GEMM", "0")
    monkeypatch.setenv("AITER_CONFIG_GEMM_BF16", shipped)
    use_private_bf16_gemm_table()
    assert "AITER_CONFIG_GEMM_BF16" not in os.environ

    monkeypatch.setenv("LUMENRL_PRIVATE_GEMM", "0")
    monkeypatch.setenv("AITER_CONFIG_GEMM_BF16", "/tmp/caller-gemm.csv")
    use_private_bf16_gemm_table()
    assert os.environ["AITER_CONFIG_GEMM_BF16"] == "/tmp/caller-gemm.csv"


def test_default_installs_the_shipped_table_only_when_unset(monkeypatch):
    shipped = private_bf16_gemm_table()
    assert shipped
    monkeypatch.setenv("LUMENRL_PRIVATE_GEMM", "1")
    monkeypatch.delenv("AITER_CONFIG_GEMM_BF16", raising=False)
    use_private_bf16_gemm_table()
    assert os.environ["AITER_CONFIG_GEMM_BF16"] == shipped

    monkeypatch.setenv("AITER_CONFIG_GEMM_BF16", "/tmp/caller-gemm.csv")
    use_private_bf16_gemm_table()
    assert os.environ["AITER_CONFIG_GEMM_BF16"] == "/tmp/caller-gemm.csv"
