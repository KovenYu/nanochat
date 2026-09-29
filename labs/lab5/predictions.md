# Lab 5 · predictions.md

规则同前。对照文件：labs/mini/lab5_grpo.py。policy = sft50 (d24, step 233)，
harness 配置：`GRPOConfig(kl_beta=0.1)`（其余默认：num_samples 16, device_batch_size 8,
max_new_tokens 64, temperature 1.0, top_k 50），task = `ArithmeticTask(64, seed=0)` 的第 0 条，
step=0 的一次 rollout + 第 0 个 micro-batch（8 行）的一次 forward/backward。
✏ 行要求写推导/理由。E 组是动力学题，harness 用一个 ~1 分钟的短跑兑现。

## A. rollout batch 的形状与 mask

| # | 问题 | 你的预测 |
|---|---|---|
| A1 ✏ | `render_for_completion` 给出的 prompt 长度 P（"What is a+b?" 一位数）。列出每个 special token 与内容 token 的顺序 |special token 包括开头的user start user end 和 assistant start |
| A2 | rollout 后 `inputs` 的 shape (S, T)。T 与 P、max_new_tokens 的关系（假设至少一个 sample 没在 64 token 内终止） | T=P+max_new_tokens-1|
| A3 | 一行 sample 采了 g 个 token（含或不含终止符？）后终止：该行 `targets` 里 −1 的个数，用 T、P、g 表示 |P |
| A4 | `<|assistant_end|>` 这个 id 会不会出现在 `targets >= 0` 的位置上？为什么 |会，这个也要预测的 |
| A5 | `(targets >= 0).sum()` 与 Σ_i g_i 的关系 | 相等|

## B. group 与 advantage

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | 16 个 sample 里 p 个对：正确/错误 sample 的 advantage 各是多少（用 p 表示）；`advantages.sum()` 是精确 0 还是 fp32 尾数 | 1-p/16和 -p/16; 精确是0|
| B2 | 把 advantage 全置 0 再 backward：`pg_obj` 的值、以及所有参数梯度的总 L2 norm 是精确 0.0、还是 1e-x 的小数？grad 是 None 还是零张量 | pg_obj为0，norm精确为0，因为A都是直接乘在上面的。grad是零tensor|
| B3 | step 0（policy == ref）：k3 KL 的值是精确 0 还是 1e-x？说原因 | 精确是0，因为d=0，exp(d)-d-1就是精确0|
| B4 ✏ | 同一 prompt，group 大小 16 → 4：当 pass@1 = 0.3 时，"全对或全错 → 零梯度" 的 group 占比各多少（写公式和数） | group=4的时候，全对是0.3^4, 大约是1%; 全错是0.7^4, 大约是25%。group=16的时候，全对概率是1%^4基本上忽略不计，全错的概率是25%^4=1/256=0.3%左右，也几乎忽略不计了。|

## C. 归一化与长短 sample 的权重

| # | 问题 | 你的预测 |
|---|---|---|
| C1 ✏ | 同一 micro-batch 里两条 sample，长度 g=10 与 g=60、\|A\| 相同：token-level 归一化下两者对梯度的贡献比；若改成 sequence-level（每条先 mean over 自己的 token）呢 | token-level norm下1:6, sequence level 1:1|
| C2 | 训练 forward 算出的、采样 token 的 per-token logp 均值（nats）。提示：temperature 1.0、top-k 50 采出来的 token 不是 argmax | 约 −0.23 nats。估法：大约三分之一的 token 在做选择，每次平均 2 个选项，H ≈ 1/3 × ln2 ≈ 0.23（其余 token 几乎确定）。|
| C3 ✏ | `pg_obj` 在 step 0 的符号（不用估量级）。提示：Σ_i A_i = 0，所以它等于 Σ A_i·(L_i − L̄)/N，L_i = 第 i 条的 logp 之和；答对的 sample 更短还是更长决定符号。附问：为什么 TRL 打印的 GRPO `loss` 训练全程围着 0 转、不往下走 | 答对的sample在step0大概率是更短的，更长的有可能复读机。所以pg obj是正的。之后围着0打转是因为如果答对明显更短，那么整个model就会被push往生成更短的答案，反之亦然。|
| C4 | 第 0 个 micro-batch 的 normalizer = num_valid × num_passes × examples_per_step 里，后两个因子各是多少 | 4|

## D. 采样 vs 训练：一致性与显存

| # | 问题 | 你的预测 |
|---|---|---|
| D1 ✏ | 同一批采样 token：采样时的 logp（softmax 只在 top-50 内归一化）与训练 forward 的 logp（全 32768 词表）—— 哪个大？差多少 nats/token（量级）？这对 "on policy 所以 ratio 恒为 1" 意味着什么 | 训练的logp大。按照上面的估算，假设1/3的token在做选择，不算top50的话每次选项可能稍微多一点，比如变成3个，那这里就变成1/3*ln3=0.36左右，所以量级上就是只差0.1~0.2. 这意味着”on policy 所以ratio 恒为1“在实践中并不正确，有一些偏差。|
| D2 | top_k 关掉（=0）后重算 D1，差值是精确 0 还是 1e-x？（2026-09-28 更正题面：采样时第 t 步 forward 的是长度 P+t 的前缀，训练 forward 的是整行长度 T；causal 下数学上相同，但不是同一次计算） | 直觉上是1e-x吧，因为计算涉及到的p很小，总会有rounding error|
| D3 ✏ | sft50 参数量（按 config 算：n_embd 1536, n_layer 24, vocab 32768, 无 GQA）与权重 GiB；kl_beta>0 时 policy+ref 常驻 GiB（2026-09-28 更正题面：权重不是全 bf16，见 CC 的更正说明） | 这是1.3B模型对吧，那就是ref是2.6G显存，policy本身是16倍所以是20G左右的显存。|
| D4 | 三个峰值显存的大小顺序并估量级：(i) 采样 8 行 × ~75 token（no_grad）；(ii) 训练 micro-batch 8 行的 forward+backward；(iii) 一次 optimizer.step 后新增的 optimizer state | |
| D5 ✏ | 一步里的时间：采 16 个 sample × 64 token（无 KV cache，两趟）vs 两个 micro-batch 的 forward+backward。谁大、大约几倍？无 KV cache 的采样 FLOPs 随 max_new_tokens 是线性还是二次 | |

## E. 动力学（短跑兑现）

| # | 问题 | 你的预测 |
|---|---|---|
| E1 ✏ | 已知：reward = "最后一个整数正确" 时 30 步后平均长度 68 → 36。若改成 "回答里任何位置出现正确整数即 1"，长度会升、降、还是不动？reward 会更高还是更低？说机制 | |
| E2 | 单步 reward 均值（32 个 sample）的 step-to-step 标准差量级，当真实 pass@1 ≈ 0.3 | |
