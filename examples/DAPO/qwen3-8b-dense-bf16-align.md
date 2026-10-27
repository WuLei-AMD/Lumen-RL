# Qwen3-8B dense BF16：训推一致性

给后续 agent。这条线已经测完的路径不要重做。FP8 不在这里。

训练是 Lumen 的一次 teacher-forcing。在线打分是 ATOM 逐步 decode 返回的 logprob。比较的是同一份权重、更新前，`|rollout − 训练|` 的均值，日志字段 `mismatch/abs_diff`。位置 i 给 token i+1 打分。logprob 用 GPU float32 `log_softmax`。不要把 Lumen 的训练 forward 叫成 prefill。Lumen 没有 decode，不要加。

博客公式是 `q/μ = (q/s^P) × (s^P/s^D) × (s^D/μ)`。Figure 5 画的是 q 对 s^D，也就是 rescore 关闭时的在线 decode。整段 prefill 重打分是 s^P，可以到逐位 0，不能拿来当这个数。

## 结论

同一条序列、同一个计算形状，Lumen 对 ATOM 整段 prefill 可以逐位。在线 decode 对齐到大约 1e-9。把 200 步均值抬到 4.77e-4 的 0.01 离群，来自整段 causal varlen 和逐 token varlen 在部分 prompt 长度上不是同一个数。`LUMENRL_ALIGN_DECODE=1` 之后 prefill 也走逐 token varlen，step 30 从 0.014–0.026 降到 2.61e-10。1e-9 的稀疏 ulp 还在。step 57 及以后的离群下标没有用修复后的 prefill 重跑。

栈是 Lumen FSDP2 加 8 路 ATOM TP1，MI350X gfx950。不是博客的 Megatron TP4/CP2 加 vLLM、MI300X。不要画成博客的 G10/G11 曲线。

## 版本

| 项 | 位置 |
| --- | --- |
| Lumen-RL | 本分支 `dev/qwen3-8b-bf16-align`，`dev_wul` |
| Lumen | `dev/ATOM_rollout` `7d725c6`，另加未提交的 `lumen/config.py`：stable GEMM 和 fused residual。没有这份，训练侧 GEMM 和 checkpoint 重算对不齐 |
| ATOM | ROCm `main` `6e9b3e1c`，只读，不改源码 |
| aiter | ROCm `main` `4eb7dcac`，只读，不改源码 |
| 模型 | `Qwen3-8B-Base` BF16 |
| 镜像 | `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260917` |

镜像里的 aiter 和这台机器的 flydsl 对不上。进程的 `PYTHONPATH` 要把主机 `/home/leiwu/aiter` 放在前面，并设 `LUMENRL_ATOM_AITER_SRC`。`examples/DAPO/atom_aiter_shim/sitecustomize.py` 只补缺的模块，不要用整份主机 aiter 换掉镜像里能用的那份，除非导入已经失败。

## 改动

ATOM 和 aiter 源码没有改。

训练侧，Lumen `lumen/config.py`：

- `LUMENRL_STABLE_GEMM=1` 时，QKV、gate/up、linear 的 `tgemm.mm` 把 M 补到 splitK=1。FlyDSL splitK>1 的 bf16 原子加两次结果不一样。
- fused residual 跟 hidden 一起返回，不再放在 `backbone._atom_residual`。activation checkpoint 重算一层时读不到那个属性。

训练侧，本仓库：

- `lumenrl/engine/training/fsdp_backend.py`：空的 quant 配置也能读 `LUMEN_ATOM_OPS`，并在有 atom ops 时默认打开 stable GEMM。
- `lumenrl/engine/training/packing.py`：`LUMENRL_ROWWISE_TRAIN_ATTN=1` 时每个 token 用 q_len=1 `flash_attn_varlen_func` 看自己的 KV 前缀，logprob 改用 GPU float32 `log_softmax`。这是 O(tokens) 次 launch，默认关闭。它消掉了 2048 key 之后大约 0.1 的归约差。
- `lumenrl/engine/private_gemm.py` 和 `configs/qwen3_8b_bf16_tuned_gemm.private.csv`：Qwen3-8B 的 BF16 形状查这张表。进程启动前就要设 `AITER_CONFIG_GEMM_BF16`。

Rollout 侧，`lumenrl/aiter_patch/`，由 `atom_ray_server.py` 放进 ATOM worker 的 `PYTHONPATH`：

- prefill 的 RoPE 留在 aiter。`ATOM_FORCE_ATTN_TRITON=1` 否则会把 prefill RoPE 带去 Triton，Q/K 和训练对不上。
- `LUMENRL_ALIGN_DECODE=1`：decode 用 aiter RoPE 加 `flash_attn_varlen`，不用汇编 paged attention。prefill 同样改成逐 token varlen。整段 causal kernel 是 `fmha_fwd_hd128_bf16_causal_group`，q_len=1 是 `fmha_fwd_hd128_bf16_group`。354 token 的 prompt 上两者差 0.06–0.18，117 token 上逐位相同。
- `LUMENRL_ALIGN_CONTIG_KV=1`：`nseq==1` 时 decode 读连续 K/V，不从 paged cache gather。paged KV 是早先 1e-9 的主要来源。
- `LUMENRL_ALIGN_SYNC_OUTPUT=1`：deferred D2H 之前 clone 这一步的 logprob 并同步。否则下一步 forward 会复用那块显存。
- `LUMENRL_PREFILL_RESCORE=1` 才把生成结束后的整段 prefill 写回 loss。博客的 s^D 不要开。`run_dapo.sh` 的 BF16 ATOM 模式目前把这个默认设成 1，复现在线数字时必须显式设成 0。

`atom_ray_server.py` 把上述环境变量转给 worker。多桶 IPC 包在 `weight_sync_session()` 里。`LUMENRL_ALIGN_SYNC_WEIGHTS=1` 只在最后一桶之后 synchronize 并打印指纹，默认关闭。它没有去掉 step 30 的离群，不要再加 synchronize 当修复。ATOM 侧指纹当时打印 `missing`，因为参数名对不上 `layers.0.self_attn.q_proj.weight`；训练侧 8 卡 crc 相同。不要把 missing 当成权重分叉。

六个 tag，不要改成默认开启：

```text
rmsnorm,residual_rmsnorm,qkv,rope,varlen_attn,gate_up
```

对齐 decode 每次 forward 的地址不固定。`LUMENRL_ALIGN_DECODE=1` 时保持 eager、`compilation_config.level=0`。level 3 的 FULL graph 会重放固定地址然后中止。

## 在线比较怎么配

seed 1234，`reward.shuffle_seed=1234`。`gen_batch_size=1`，`num_generations=8`，`train_global_batch_size=8`。回复上限 7168，总长上限 8192，`max_token_len_per_gpu=4096`，`max_num_batched_tokens=16384`。`kl_coeff=0`。不要设成 0.001：那会再加载一份 reference，8 张卡放不下。`dynamic_sampling=false`，`filter_groups.enable=false`。否则一组准确率没有方差时整步会被丢掉。`gpu_memory_utilization=0.38`，sleep level 2。

必须开：`LUMEN_ATOM_OPS` 六个 tag，`LUMENRL_ALIGN_DECODE=1`，`LUMENRL_ALIGN_SYNC_OUTPUT=1`，`LUMENRL_ALIGN_CONTIG_KV=1`（容量 8192），`LUMENRL_ROWWISE_TRAIN_ATTN=1`，`LUMENRL_STABLE_GEMM=1`，`ATOM_FORCE_ATTN_TRITON=1`，`ATOM_H2D_BACKEND=direct`，`ATOM_USE_TORCH_RMSNORM=0`，私有 GEMM 表。

必须关：`LUMENRL_PREFILL_RESCORE=0`。不要开 `LUMENRL_IGNORE_EOS`，除非就是要打满回复长度。`LUMENRL_TRAIN_DECODE_TRACE_DIR` 只在要保存逐 token trace 时设。

温度是 1，`SamplingParams` 没有 seed。ATOM 采样不跟 trainer seed 走。两次运行的 token 可以不同。每条运行只和它自己的 trace 比。callback 显示的 step 是循环下标加 1。trace 文件用 0-based 下标，显示 step 30 是 `step-00029.pt`。

## 已经测过的结果

同形状。Lumen 一次 teacher-forcing 对 ATOM 整段 prefill：16 和 24 token 的选中 logprob 是 0。4096 token 在单卡和 8 卡 FSDP2 上，4095 个选中 logprob 的 max abs 是 0。

2048。整段 varlen 对 q_len=1，位置 2047 逐位，位置 2048 起 logprob 差到大约 0.1。逐行训练加 q_len=1 prefill 之后，2176 token 的 2175 个选中 logits 和 GPU float32 logprob 是 0。24 token 两次 backward，399/399 参数梯度逐位。

在线 decode，rescore 关闭，修复 prefill 之前。200 step，约 7.5 小时，退出码 0。均值 4.77e-4，中位数 1.51e-10。192 步低于 1e-4，这 192 步均值 1.43e-9，其中 47 步是 0。另外 8 步是 0.0073–0.0151，绝对差超过 0.1 的比例 0.025–0.056：step 57、83、30、199、126、120、186、160。日志 `logs/ex5-blog200-online.log`，在训练机上，不在 git 里。

这 8 步的误差从第一个回复 token 就开始，这个 token 是 prefill 打的。8 条序列都偏，同一个 token 在不同卡上的 logprob 相同，所以不是漏同步，也不是一张卡。前后一步是 1e-10。权重同步 fence 之后 step 30 变成 0.0256，fence 排除。

基座权重上把 trace 里的 token 强制喂回去：prompt 117 的两条 prefill 逐位相同；prompt 354 的第一个回复 token 差 0.058、0.067、0.183。绝对值对不上训练当时的 0.123，因为权重不是 step 30 的。

修复之后。同样的 200 step 配置，prefill 改为逐 token varlen，只跑 30 step。退出码 0，约 1 小时。30 步最小 0，最大 1.72e-7（step 16），均值 6.71e-9。step 30 是 2.61e-10。绝对差超过 0.1 的比例全是 0。日志 `logs/ex5-blog30-q1fix.log`。

1e-9 还在。它是稀疏的 float32 ulp，rollout 更高。同一条 token 再喂给引擎可以是 0/2176；训练当时存下来的 rollout 仍可能有几处 1 个 ulp。512 token 一步在 clone 之后可以是 0。回复打满 2176 或 7168 的单步均值在 1e-10 到 1e-9。完整序列 q_len=1 rescore 可以到逐位 0，那是换掉在线返回值。

## 不要再做

- 不要改 ATOM 或 aiter 源码来修这两个差。
- 不要把训练 GEMM 按 M=1 逐段切开。Qwen3-8B 这些 BF16 形状从 M=1 到 M=2310 已经是同一个 splitK=1 kernel。切开之后 2176 的均值仍是 3.50e-9，代码已撤。
- 不要再加一次 IPC 之后的 `cuda.synchronize` 当 0.01 的修复。测过，step 30 更大。
- 不要用单步或短 `max_response` 复现 0.01 离群。它跟 seed 1234 的数据顺序和 prompt 长度走，短回复会换成另一批 token。
- 不要把 `run_dapo.sh` 默认的 rescore 打开值写成博客 s^D。
- 不要把 KL 设成 0.001，不要为了博客比较打开 filter groups。
- 不要把 1e-9 说成已经消掉，也不要把 step 57 之后说成已经用新 prefill 复测过。
