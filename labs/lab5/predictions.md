# Lab 5 · predictions.md

规则同前。对照文件：labs/mini/lab5_grpo.py。policy = sft50 (d24, step 233)，
harness 配置：`GRPOConfig(kl_beta=0.1)`（其余默认：num_samples 16, device_batch_size 8,
max_new_tokens 64, temperature 1.0, top_k 50），task = `ArithmeticTask(64, seed=0)` 的第 0 条，
step=0 的一次 rollout + 第 0 个 micro-batch（8 行）的一次 forward/backward。
✏ 行要求写推导/理由。E 组是动力学题，harness 用一个 ~1 分钟的短跑兑现。

## A. rollout batch 的形状与 mask

| # | 问题 | 你的预测 |
|---|---|---|
| A1 ✏ | `render_for_completion` 给出的 prompt 长度 P（"What is a+b?" 一位数）。列出每个 special token 与内容 token 的顺序 | |
| A2 | rollout 后 `inputs` 的 shape (S, T)。T 与 P、max_new_tokens 的关系（假设至少一个 sample 没在 64 token 内终止） | |
| A3 | 一行 sample 采了 g 个 token（含或不含终止符？）后终止：该行 `targets` 里 −1 的个数，用 T、P、g 表示 | |
| A4 | `<|assistant_end|>` 这个 id 会不会出现在 `targets >= 0` 的位置上？为什么 | |
| A5 | `(targets >= 0).sum()` 与 Σ_i g_i 的关系 | |

## B. group 与 advantage

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | 16 个 sample 里 p 个对：正确/错误 sample 的 advantage 各是多少（用 p 表示）；`advantages.sum()` 是精确 0 还是 fp32 尾数 | |
| B2 | 把 advantage 全置 0 再 backward：`pg_obj` 的值、以及所有参数梯度的总 L2 norm 是精确 0.0、还是 1e-x 的小数？grad 是 None 还是零张量 | |
| B3 | step 0（policy == ref）：k3 KL 的值是精确 0 还是 1e-x？说原因 | |
| B4 ✏ | 同一 prompt，group 大小 16 → 4：当 pass@1 = 0.3 时，"全对或全错 → 零梯度" 的 group 占比各多少（写公式和数） | |

## C. 归一化与长短 sample 的权重

| # | 问题 | 你的预测 |
|---|---|---|
| C1 ✏ | 同一 micro-batch 里两条 sample，长度 g=10 与 g=60、\|A\| 相同：token-level 归一化下两者对梯度的贡献比；若改成 sequence-level（每条先 mean over 自己的 token）呢 | |
| C2 | 训练 forward 算出的、采样 token 的 per-token logp 均值（nats）。提示：temperature 1.0、top-k 50 采出来的 token 不是 argmax | |
| C3 ✏ | `pg_obj` 在 step 0 的符号与量级。提示：Σ_i A_i = 0，所以它不是 "advantage × 平均 logp"；想它到底在度量什么 | |
| C4 | 第 0 个 micro-batch 的 normalizer = num_valid × num_passes × examples_per_step 里，后两个因子各是多少 | |

## D. 采样 vs 训练：一致性与显存

| # | 问题 | 你的预测 |
|---|---|---|
| D1 ✏ | 同一批采样 token：采样时的 logp（softmax 只在 top-50 内归一化）与训练 forward 的 logp（全 32768 词表）—— 哪个大？差多少 nats/token（量级）？这对 "on policy 所以 ratio 恒为 1" 意味着什么 | |
| D2 | top_k 关掉（=0）后重算 D1，差值是精确 0 还是 1e-x？（mini 没有 KV cache，两条路径的 forward 相同） | |
| D3 ✏ | sft50 参数量（按 config 算：n_embd 1536, n_layer 24, vocab 32768, 无 GQA）与 bf16 权重 GiB；kl_beta>0 时 policy+ref 常驻 GiB | |
| D4 | 三个峰值显存的大小顺序并估量级：(i) 采样 8 行 × ~75 token（no_grad）；(ii) 训练 micro-batch 8 行的 forward+backward；(iii) 一次 optimizer.step 后新增的 optimizer state | |
| D5 ✏ | 一步里的时间：采 16 个 sample × 64 token（无 KV cache，两趟）vs 两个 micro-batch 的 forward+backward。谁大、大约几倍？无 KV cache 的采样 FLOPs 随 max_new_tokens 是线性还是二次 | |

## E. 动力学（短跑兑现）

| # | 问题 | 你的预测 |
|---|---|---|
| E1 ✏ | 已知：reward = "最后一个整数正确" 时 30 步后平均长度 68 → 36。若改成 "回答里任何位置出现正确整数即 1"，长度会升、降、还是不动？reward 会更高还是更低？说机制 | |
| E2 | 单步 reward 均值（32 个 sample）的 step-to-step 标准差量级，当真实 pass@1 ≈ 0.3 | |
