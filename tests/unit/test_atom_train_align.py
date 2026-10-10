from __future__ import annotations

import os

import torch

from lumenrl.core.atom_train_align import (
    PAD_WEIGHTS,
    QKV_SPLITK1,
    QKV_WEIGHT,
    activate,
    place_decode_row,
    should_pad_weight,
    should_pin_qkv,
)


def test_place_decode_row_writes_the_requested_index() -> None:
    row = torch.ones(1, 4)
    padded, pos = place_decode_row(row, 93, 2)
    assert pos == 2
    assert padded.shape == (93, 4)
    assert padded[2, 0].item() == 1
    assert padded[0, 0].item() == 0
    assert padded[3, 0].item() == 0


def test_qkv_pin_is_only_the_decode_row() -> None:
    os.environ["ATOM_TRAIN_ALIGN"] = "1"
    os.environ["ATOM_TRAIN_ALIGN_M"] = "93"
    os.environ["ATOM_BITWISE_M"] = "93"
    try:
        assert should_pin_qkv(QKV_WEIGHT, 1)
        assert not should_pin_qkv(QKV_WEIGHT, 93)
        assert should_pad_weight((24576, 4096))
        assert should_pad_weight((4096, 12288))
        assert (4096, 4096) not in PAD_WEIGHTS
        assert "split_k1" in QKV_SPLITK1
    finally:
        os.environ.pop("ATOM_TRAIN_ALIGN", None)
        os.environ.pop("ATOM_TRAIN_ALIGN_M", None)
        os.environ.pop("ATOM_BITWISE_M", None)


def test_activate_exports_the_lumen_rollout_flag(monkeypatch) -> None:
    for key in ("ATOM_TRAIN_ALIGN", "ATOM_TRAIN_ALIGN_M", "ATOM_BITWISE_M", "LUMEN_ROLLOUT"):
        monkeypatch.delenv(key, raising=False)
    activate(93)
    assert os.environ["ATOM_TRAIN_ALIGN"] == "1"
    assert os.environ["ATOM_TRAIN_ALIGN_M"] == "93"
    assert os.environ["ATOM_BITWISE_M"] == "93"
    assert os.environ["LUMEN_ROLLOUT"] == "ATOM"
    for key in ("ATOM_TRAIN_ALIGN", "ATOM_TRAIN_ALIGN_M", "ATOM_BITWISE_M", "LUMEN_ROLLOUT"):
        monkeypatch.delenv(key, raising=False)
