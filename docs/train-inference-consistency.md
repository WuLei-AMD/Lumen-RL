# Train-inference consistency

This page describes the Qwen3-8B dense BF16 path that makes a Lumen
training forward and an ATOM rollout agree on the same token. FP8 is out
of scope.

Training is one Lumen teacher-forcing forward (FSDP2). The online score is
the log-probability ATOM returns from `generate`. Position `i` scores token
`i+1`. The last token of each sequence is dropped. Both sides use GPU
float32 `log_softmax`. The logged field `mismatch/abs_diff` is the mean of
`|rollout − training|` on the response mask. `frac_abs_gt_0.1` is the
fraction of response tokens whose absolute gap exceeds 0.1.

`LUMENRL_PREFILL_RESCORE=0` compares training with online decode (s^D).
Setting it to 1 replaces the log-probability that enters the loss with one
prefill of the finished sequence (s^P). Those two numbers are not the same
comparison.

## What the feature changes

ATOM BF16 in `examples/DAPO/run_dapo.sh` turns the alignment on. The gather
inside decode allocates every step, so that path stays eager
(`enforce_eager=true`, `compilation_config.level=0`). A level-3 CUDA graph
replays fixed addresses and aborts.

| Piece | Where | What it does |
| --- | --- | --- |
| Six training ops | `LUMEN_ATOM_OPS=rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up` | Training norm, QKV, RoPE, attention, and gate/up follow the ATOM kernels. An empty tag list leaves HF RMSNorm, HF RoPE, and HF attention in place. |
| Decode attention | `LUMENRL_ALIGN_DECODE=1`, `lumenrl/aiter_patch/align_decode.py` | Decode uses aiter RoPE and gathers the paged cache into `flash_attn_varlen`. It does not run the assembly paged-decode kernel. |
| Same prefix reduction | `LUMENRL_PREFILL_DECODE_ATTN=1`, `LUMENRL_ROWWISE_TRAIN_ATTN=1`, `lumenrl/aiter_patch/qlen1_attention.py`, `lumenrl/engine/training/packing.py` | Prefill and the packed training forward score a token with the same q_len=1 reduction decode uses. A full causal varlen call diverges from that reduction once a query reads past 2048 keys. |
| Vocabulary tail | `LUMENRL_TRAIN_VOCAB_MASK`, `packing.py` | Training blanks the padded lm_head columns ATOM drops before `log_softmax`. For Qwen3-8B-Base that width is 151669. Leaving it unset scores every rollout token high. |
| Repeatable GEMM | `LUMENRL_STABLE_GEMM=1`, set by default once `LUMEN_ATOM_OPS` is non-empty (`fsdp_backend.py`) | Pads M so the BF16 GEMM runs at splitK=1. splitK>1 FlyDSL BF16 is not repeatable on one input. |
| GEMM table | `AITER_CONFIG_GEMM_BF16` | `LUMENRL_PRIVATE_GEMM=0` keeps the public table. Otherwise, when the variable is unset and `configs/qwen3_8b_bf16_tuned_gemm.private.csv` exists, `run_dapo.sh` points the variable at that file. |
| Assembly decode | `ATOM_FORCE_ATTN_TRITON=1` | The assembly paged-decode kernel returns a finite wrong value when a context fills exactly 16 pages and the last page is partial. The mean absolute gap stays flat; `mismatch/chi2_token` jumps. The flag is the default on every ATOM mode in `run_dapo.sh`. Keep prefill RoPE on aiter: the flag otherwise switches it to Triton while training stays on aiter. |
| Reference model | `lumenrl/workers/ref_worker.py` | The reference log-probabilities use the same packed forward, including the vocabulary mask and the q_len=1 attention when those switches are set. |

`run_dapo.sh` defaults `LUMENRL_PREFILL_RESCORE=1` on ATOM BF16, because a
packed prefill and decode diverge once a query reads past 2048 keys. The
loss then uses one prefill of the finished sequence. The online s^D numbers
below were taken with that switch forced to 0.

Contiguous K/V (`LUMENRL_ALIGN_CONTIG_KV=1`, capacity 8192) is optional. On
the 2176-token protocol it moved the mean absolute gap from 3.84e-9 to
3.43e-10 before rescore.

## How to run

```bash
MODE=atombf16 bash examples/DAPO/run_dapo.sh
```

The online measurements below also set `LUMENRL_PREFILL_RESCORE=0`,
`LUMENRL_ROWWISE_TRAIN_ATTN=1`, `LUMENRL_ALIGN_CONTIG_KV=1`,
`ATOM_FORCE_ATTN_TRITON=1`, and the private GEMM table. Seed and
`reward.shuffle_seed` were 1234. `kl_coeff=0` (a coefficient of 0.001 loads
a second model and does not fit on 8 GPUs). Filter groups were off.

## Measured results

Hardware is 8× AMD Instinct MI355X (gfx950). Model is Qwen3-8B-Base, BF16.
Rollout is ATOM at tensor parallel 1. Training is Lumen FSDP2.

| Pin | Value |
| --- | --- |
| Lumen-RL | `dev/qwen3-8b-bf16-align` |
| Lumen | `dev/ATOM_rollout` `0c2810c` |
| ATOM | `6e9b3e1` (unchanged during the measurement) |
| aiter | `4eb7dcac` (unchanged during the measurement) |
| Image | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917` |

Logs live on the training machine under `lumen-atom-rollout/logs/` and are
not in git. The image aiter fails to import flydsl; the measured process
puts host aiter `4eb7dcac` first on `PYTHONPATH` and sets
`LUMENRL_ATOM_AITER_SRC`. `examples/DAPO/atom_aiter_shim/sitecustomize.py`
fills the modules the image lacks.

### Same weights, fixed tokens

Teacher-forcing against a full ATOM prefill, six ops, stable GEMM, prefill
RoPE on aiter:

- 16 and 24 tokens: selected log-probabilities differ by 0.
- 4096 tokens, one GPU and 8-GPU FSDP2: max abs of the 4095 selected
  log-probabilities is 0, and hidden states of layers 0–35 differ by 0.
- Positions 0–2047 of one eager aligned decode match bitwise. From position
  2048, full `flash_attn_varlen` and the q_len=1 call differ (max abs
  2.87e-1, mean abs 2.57e-2 on 4095 log-probabilities).
- With `LUMENRL_ROWWISE_TRAIN_ATTN=1` and q_len=1 prefill, 2176 tokens give
  2175 selected logits and GPU float32 log-probabilities of 0. On 24 tokens
  the two backwards match bitwise, including all 399 parameter gradients.

With the private GEMM table forced to splitK=1, the same BF16 input run
twice differs by 0. That table is
`configs/qwen3_8b_bf16_tuned_gemm.private.csv`.

### Online decode (`LUMENRL_PREFILL_RESCORE=0`)

| Run | Response tokens | `mismatch/abs_diff` | Notes |
| --- | --- | --- | --- |
| Reply cap 512, 8 sequences | 4096 | 0 | Trace nonzero count 0. `logs/ex5-logprob-snap.log` |
| Reply filled to 2176 | 17408 | 8.26e-10 | Positions 0–511 are 0. 54 nonzero tokens, max abs 9.54e-7. `logs/ex5-snap2176.log` |
| Model stops on its own, lengths 238–4670 | 10060 | 2.43e-10 | 8 nonzero tokens. `logs/ex5-snapmix.log` |
| Reply filled to 7168, pack cap 4096 per GPU | 57344 | 9.42e-10 | `frac_abs_gt_0.1` is 0, max abs 3.81e-6. `logs/ex5-snap7168.log` |

The 200-step online run (2026-10-20 14:49:51Z–22:23:41Z, exit 0, prefill
still a full varlen, rescore off) is `logs/ex5-blog200-online.log`:

| | |
| --- | --- |
| Mean / median / min / max `mismatch/abs_diff` | 4.771e-4 / 1.506e-10 / 0 / 0.015071 |
| Steps below 1e-4 | 192 / 200, mean 1.425e-9, 47 of them exactly 0 |
| Outlier steps | 30, 57, 83, 120, 126, 160, 186, 199 (`frac_abs_gt_0.1` from 0.025 to 0.056) |

Those 8 outliers are the first response token on a few prompt lengths,
where a full causal varlen prefill and the q_len=1 reduction disagree by
about 0.01. After prefill moved to that same q_len=1 reduction
(`LUMENRL_PREFILL_DECODE_ATTN=1`), the first 30 steps of the same seed
(`logs/ex5-blog30-q1fix.log`, 2026-10-25, exit 0) have mean 6.71e-9, max
1.72e-7, and `frac_abs_gt_0.1` of 0 on every step. The 8 outlier step
indices were not re-run under that prefill.

Turning rescore back on after contiguous K/V makes the 2176-token protocol
bitwise (17408/17408). That zero is s^P, not the 200-step s^D number.

## Limits

The packed trainer's backward is still one full causal varlen on the same
Q, K, and V (`packing.py`). It matches the forward on short sequences and
diverges once a query reads past 2048 keys. A forward gap of 0 does not
mean the actor update is the gradient of the forward that produced the
log-probability.

Reply tokens can differ across two runs. `SamplingParams` on the pinned
ATOM commit has no seed field, so the trainer seed does not enter sampling.
Each run is compared only with the log-probabilities it saved.
