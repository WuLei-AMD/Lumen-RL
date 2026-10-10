# ATOM train align

Lumen `rollout: ATOM` and ATOM prefill already share one BF16 forward on Qwen3-8B. ATOM decode does not. This feature makes decode use that same contract when a config asks for it.

## What is aligned

| Op | Training / prefill | Decode without the contract | Decode with the contract |
|---|---|---|---|
| QKV | flydsl `split_k1`, solidx 4696 | flydsl `split_k4`, atomic BF16 add | same `split_k1` kernel |
| gate/up, down | full M | M=1 | row placed into the prefill M, then sliced back |
| o_proj, LM head | full M | M=1 | same placement |
| attention | `flash_attn_varlen_func` | `pa_fwd_asm` | varlen over the tokens decoded so far |

`split_k4` is not a value training can match. Two launches of the same M=1 input differ in 1222/4096 elements, max abs 1.953e-3, because the four K partials are atomic-added as BF16.

## Config

```yaml
policy:
  generation_backend: atom
  generation:
    atom_cfg:
      train_align: true
      train_align_m: 93   # 0 uses policy.max_total_sequence_length
```

Loading that config exports `ATOM_TRAIN_ALIGN=1`, `ATOM_TRAIN_ALIGN_M`, `ATOM_BITWISE_M`, and `LUMEN_ROLLOUT=ATOM`. The training actor then takes Lumen's existing `rollout: ATOM` patches. ATOM decode reads `ATOM_BITWISE_M` and, when `install()` runs, uses the operators in `lumenrl/core/atom_train_align.py`.

The flag defaults off. A normal ATOM rollout keeps `split_k4` and paged attention.

## Qwen3-8B case

Runbook: [`examples/ATOM-Train-Align/README.md`](../examples/ATOM-Train-Align/README.md). Config: `configs/atom_train_align_qwen3_8b.yaml`.
