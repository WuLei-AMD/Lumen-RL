# ATOM-Train-Align

Qwen3-8B BF16 case. Lumen training uses `rollout: ATOM`. ATOM decode uses the operators in `lumenrl/core/atom_train_align.py` so the selected-token logprob matches that forward.

Config: [`configs/atom_train_align_qwen3_8b.yaml`](../../configs/atom_train_align_qwen3_8b.yaml).

```bash
python examples/ATOM-Train-Align/run_case.py
```

`--gpu` also runs the pinned QKV kernel. It needs the gfx950 image and one GPU. The default check does not.

Measured eager TP=1 decode on this case is 13.84 ms/token without the contract and 23.93 ms/token with it. Prefill stays about 14.7 ms. The extra time is the M=93 placement and the syncs around it, not the LM head.
