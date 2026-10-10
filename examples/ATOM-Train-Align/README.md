# ATOM-Train-Align

Qwen3-8B BF16。同一段 93 个冻结 token，eager，TP=1。训练前向走 Lumen `rollout: ATOM`。ATOM decode 打开 `train_align` 之后，selected-token logprob 和这段 prefill 对上。

配置是 [`configs/atom_train_align_qwen3_8b.yaml`](../../configs/atom_train_align_qwen3_8b.yaml)：

```yaml
policy:
  generation_backend: atom
  max_total_sequence_length: 93
  generation:
    atom_cfg:
      train_align: true
      train_align_m: 93
```

`train_align` 默认关。打开后进程里是 `ATOM_TRAIN_ALIGN=1`、`ATOM_TRAIN_ALIGN_M=93`、`ATOM_BITWISE_M=93`、`LUMEN_ROLLOUT=ATOM`。

## 环境

镜像 `lumen/mxfp4-atom:gfx950`，一张卡，`ATOM_USE_TORCH_RMSNORM=0`。权重 `/home/leiwu/models/Qwen3-8B`。token 是 `compare-out/frozen_tokens.pt`。比较脚本在对齐工作区根目录，不在这个包里。

```bash
docker run --rm --init \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --network=host --security-opt=seccomp=unconfined --shm-size=32G \
  --env HIP_VISIBLE_DEVICES=0 \
  --env ATOM_USE_TORCH_RMSNORM=0 \
  --volume "$LUMEN":/workspace/Lumen \
  --volume "$LUMEN_RL/third_party/ATOM":/app/ATOM \
  --volume "$ALIGN":/home/leiwu/lumen-atom-rollout \
  --volume /home/leiwu/models/Qwen3-8B:/home/leiwu/models/Qwen3-8B:ro \
  --workdir /home/leiwu/lumen-atom-rollout \
  --entrypoint bash \
  lumen/mxfp4-atom:gfx950 -lc '<command>'
```

`$ALIGN` 是放着 `compare_*.py` 的目录。换 `MASTER_PORT`，避开已经占用的端口。

## 验收

selected-token logprob 的 `mismatch_count = 0` 且 `max_abs = 0`。CPU `log_softmax`。

训练前向对 ATOM TP=2 prefill，93 token：

```bash
python compare_train_vs_tp2.py \
  --model /home/leiwu/models/Qwen3-8B \
  --tokens compare-out/frozen_tokens.pt \
  --atom-save compare-out/tp2-rollout-rank0.pt \
  --report compare-out/report-train-vs-tp2.txt
```

hidden 0/380928，logits 0/14130048，logprob 0/92。

decode 打开合同之后对同一段 prefill：

```bash
python compare_prefill_decode.py \
  --model /home/leiwu/models/Qwen3-8B \
  --tokens compare-out/frozen_tokens.pt \
  --landed \
  --master-port 29691 \
  --report compare-out/report-decode-landed.txt
```

`--landed` 把 `ATOM_BITWISE_M` 设成 93。93 个 hidden 是 0，logits 0/14130048，logprob 0/92。不带这个开关时，hidden 大约 349k/380928，logprob 92/92 都对不上。decode 的 QKV `split_k4` 对同一输入两次 launch 也不相同，所以合同把 QKV 钉在 prefill 的 `split_k1`。

## 性能

同一条 eager 路径，prefill 是 93 个 token 一次前向加 lm_head。decode 是逐 token 的层前向加 lm_head。没有采样，也没有训练那一次前向。

| | 原始 decode | `ATOM_BITWISE_M=93` |
|---|---|---|
| prefill | 14.73 ms | 14.78 ms |
| 每个 token | 13.84 ms | 23.93 ms |
| 36 层前向 | 13.58 ms | 23.60 ms |
| lm_head | 0.25 ms | 0.32 ms |

93 步墙钟从 1286.7 ms 到 2225.6 ms。多出来的时间在 36 层前向。lm_head 大约多 0.08 ms/token。

```bash
MASTER_PORT=29721 python compare-out/bench_bitwise_perf.py
MASTER_PORT=29731 python compare-out/profile_bitwise_decode.py
```
