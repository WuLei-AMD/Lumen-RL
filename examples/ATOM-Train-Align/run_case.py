"""Check the ATOM train-align contract on the Qwen3-8B case.

Loads ``configs/atom_train_align_qwen3_8b.yaml`` and checks the operators in
``lumenrl.core.atom_train_align``. Pass ``--gpu`` to also launch the prefill
``split_k1`` QKV kernel on one row and on M=93.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_cases():
    import importlib.util

    path = Path(__file__).resolve().parent / "cases.py"
    spec = importlib.util.spec_from_file_location("atom_train_align_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_contract() -> list[str]:
    import torch

    from lumenrl.core.atom_train_align import (
        PAD_WEIGHTS,
        QKV_SPLITK1,
        QKV_WEIGHT,
        place_decode_row,
        should_pad_weight,
        should_pin_qkv,
    )
    from lumenrl.core.config import LumenRLConfig

    cases = _load_cases()
    cfg = LumenRLConfig.from_yaml(ROOT / "configs" / "atom_train_align_qwen3_8b.yaml")
    lines = []
    atom = cfg.policy.generation.atom_cfg
    lines.append(f"train_align={atom.train_align} m={atom.train_align_m}")
    lines.append(f"ATOM_TRAIN_ALIGN={os.environ.get('ATOM_TRAIN_ALIGN')} M={os.environ.get('ATOM_TRAIN_ALIGN_M')}")
    lines.append(f"LUMEN_ROLLOUT={os.environ.get('LUMEN_ROLLOUT')}")
    if not atom.train_align or atom.train_align_m != 93:
        raise SystemExit("config did not enable train_align M=93")
    if os.environ.get("LUMEN_ROLLOUT") != "ATOM":
        raise SystemExit("config load did not export LUMEN_ROLLOUT=ATOM")

    row = torch.ones(1, 4)
    padded, pos = place_decode_row(row, 93, 2)
    if padded.shape != (93, 4) or pos != 2 or padded[2, 0].item() != 1 or padded[0, 0].item() != 0:
        raise SystemExit("place_decode_row did not put the token at row 2")
    if not should_pin_qkv(QKV_WEIGHT, 1):
        raise SystemExit("M=1 QKV was not pinned")
    if should_pin_qkv(QKV_WEIGHT, 93):
        raise SystemExit("full-M QKV must stay on the normal GEMM")
    if not should_pad_weight((24576, 4096)) or not should_pad_weight((4096, 12288)):
        raise SystemExit(f"pad set changed: {PAD_WEIGHTS}")
    if "split_k1" not in QKV_SPLITK1 or "split_k4" in QKV_SPLITK1:
        raise SystemExit("QKV kernel is not the prefill split_k1 kernel")
    lines.append("place_decode_row row 2: ok")
    lines.append("qkv pin and pad shapes: ok")
    lines.append(
        "recorded decode ms/token "
        f"original={cases.DECODE_MS_PER_TOKEN['original']} "
        f"train_align={cases.DECODE_MS_PER_TOKEN['train_align']}"
    )
    return lines


def check_gpu(model_path: str) -> list[str]:
    import torch
    from aiter.tuned_gemm import flydsl_gemm

    from lumenrl.core.atom_train_align import QKV_SOLIDX, QKV_SPLITK1, pin_qkv

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP is not available")
    weight = torch.randn(6144, 4096, device="cuda", dtype=torch.bfloat16)
    full = torch.randn(93, 4096, device="cuda", dtype=torch.bfloat16)
    cfg = {"kernelName": QKV_SPLITK1}
    with torch.inference_mode():
        ref = flydsl_gemm(full, weight, QKV_SOLIDX, config=cfg)
        one = pin_qkv(full[:1], weight)
    mismatch = int((one.view(torch.int16) != ref[:1].view(torch.int16)).sum().item())
    return [f"gpu qkv M=1 vs M=93 row0 mismatch={mismatch}/{one.numel()} model={model_path}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", default="/home/leiwu/models/Qwen3-8B")
    args = parser.parse_args()
    lines = check_contract()
    if args.gpu:
        lines.extend(check_gpu(args.model))
    text = "\n".join(lines) + "\n"
    print(text, end="")


if __name__ == "__main__":
    main()
