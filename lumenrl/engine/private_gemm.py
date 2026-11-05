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
    """Install the shipped BF16 table unless the caller already chose one.

    Call this before aiter caches ``get_config_file``. ``LUMENRL_PRIVATE_GEMM=0``
    requests the public table. aiter treats an unset or empty
    ``AITER_CONFIG_GEMM_BF16`` the same way and merges the public CSV with
    ``model_configs/``. A value that points at the shipped private table is
    removed, so a shell that exported it first still follows the flag. Any
    other caller-provided path is left in place.
    """
    shipped = private_bf16_gemm_table()
    if os.getenv("LUMENRL_PRIVATE_GEMM", "1").strip() == "0":
        current = os.environ.get("AITER_CONFIG_GEMM_BF16")
        if current is not None and (
            not current.strip()
            or (shipped is not None and os.path.normpath(current) == shipped)
        ):
            os.environ.pop("AITER_CONFIG_GEMM_BF16", None)
        return
    if shipped:
        os.environ.setdefault("AITER_CONFIG_GEMM_BF16", shipped)
