"""Static decode's entry budget (LUMENRL_ALIGN_STATIC_BLOCKS) keeps every real entry where full width puts it."""

import os
import random
import sys

import pytest

import lumenrl

# ATOM workers import these modules by their top-level names (aiter_patch is on their PYTHONPATH).
sys.path.insert(0, os.path.join(os.path.dirname(lumenrl.__file__), "aiter_patch"))

import align_decode as ad  # noqa: E402
import qlen1_attention as qa  # noqa: E402

CAP, SPLIT, REP = 6144, 256, 8
WIDTH = -(-CAP // SPLIT)
ROWS = 256
CAPTURES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 128, 256]


@pytest.fixture(autouse=True)
def _split(monkeypatch):
    monkeypatch.setattr(qa, "SPLIT_KEYS", SPLIT)


def _lists(lens, entries):
    starts = [i * CAP for i in range(len(lens))]
    return ad._static_lists(starts, lens, list(range(len(lens))), ROWS, WIDTH, entries, ROWS * CAP, SPLIT, REP)


@pytest.mark.parametrize("per_row", [0, 6, 8])
def test_budget_keeps_real_entries_and_padding_room(monkeypatch, per_row):
    monkeypatch.setenv("LUMENRL_ALIGN_STATIC_BLOCKS", str(per_row))
    budget = lambda rows: qa.static_entries(rows, WIDTH, per_row)
    rng = random.Random(0)
    laid_out = 0
    for _ in range(500):
        n = rng.choice([1, 2, 3, 5, 8, 16, 31, 32, 64])
        hi = rng.choice([300, 1500, 4300, CAP])
        lens = [rng.randint(1, hi) for _ in range(n)]
        real = sum(-(-x // SPLIT) for x in lens)
        if not ad._static_fits(lens, CAP):
            assert per_row > 0 and real + 1 > budget(n)
            with pytest.raises(RuntimeError):
                _lists(lens, real - 1)
            continue
        full = _lists(lens, ROWS * WIDTH)
        bud = _lists(lens, budget(ROWS))
        # gather, k_start and key counts of the real entries; merge lists of the real rows
        assert full[1][:real] == bud[1][:real]
        assert full[3][:real] == bud[3][:real]
        assert [full[2][i + 1] - full[2][i] for i in range(real)] == [bud[2][i + 1] - bud[2][i] for i in range(real)]
        assert full[4][: n * WIDTH] == bud[4][: n * WIDTH] and full[5][:n] == bud[5][:n]
        for b in (c for c in CAPTURES if c >= n):
            # A larger graph pads rows n..b onto entry `real`, which must be launched.
            assert real + (1 if b > n else 0) <= budget(b)
            k_start = bud[3][: budget(b) + 1]
            assert all(x <= y for x, y in zip(k_start, k_start[1:]))
        laid_out += 1
    assert laid_out > 0


def test_budget_shape():
    assert qa.static_entries(32, WIDTH, 0) == 32 * WIDTH
    assert qa.static_entries(1, WIDTH, 8) == WIDTH
    assert qa.static_entries(qa.STATIC_FULL_ROWS, WIDTH, 8) == qa.STATIC_FULL_ROWS * WIDTH
    assert qa.static_entries(32, WIDTH, 8) == 32 * 8 + 1
