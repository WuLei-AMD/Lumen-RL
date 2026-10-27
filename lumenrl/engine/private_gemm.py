"""Select the private BF16 GEMM table before aiter caches its config lookup."""

from __future__ import annotations

import os

_TABLE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "configs",
        "qwen3_8b_bf16_tuned_gemm.private.csv",
    )
)


def private_bf16_gemm_table() -> str | None:
    return _TABLE if os.path.isfile(_TABLE) else None


def use_private_bf16_gemm_table() -> None:
    """Keep a caller-provided path. Otherwise use the table shipped with Lumen-RL."""
    path = private_bf16_gemm_table()
    if path:
        os.environ.setdefault("AITER_CONFIG_GEMM_BF16", path)
