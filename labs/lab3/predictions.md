# Lab 3 · predictions.md

规则同前。对照文件：labs/mini/lab3_dpo.py。policy = sft50 (d24, step 233)，ref = 同权重冻结副本。
✏ 行要求写推导/理由。B4/B5 是纯推理题（实测在 Stage 3 的 β 扫描里兑现）。

## A. 双前向的计算图与显存

| # | 问题 | 你的预测 |
|---|---|---|
| A1 ✏ | 训练时 policy 侧的显存(参数+梯度+optimizer state) vs ref 侧的显存，各多少 GiB？（lab1 C 组的账本直接用） | |
| A2 | 四个 logp 向量 pi_c, pi_r, ref_c, ref_r 中哪些在 autograd graph 上？loss 的梯度经由哪几个流回模型？ | |
| A3 | `logp.gather(-1, safe.unsqueeze(-1))` —— 如果误写成 `gather(1, ...)`，会报错还是静默出错？错的话错在哪一维的语义上？ | |
| A4 ✏ | 为什么 prompt 的 token 必须排除在 sequence_logprob 之外？（想：留着它，margin 会多出什么项？） | |

## B. β 与 margin 的量级/动力学

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | 第 0 步（policy == ref）：implicit reward r_c、r_r、margin、以及 loss 的精确值 | |
| B2 ✏ | β 从 0.1 调到 0.5：policy 被拴得更紧还是更松？说机制 | |
| B3 | sft50 对一条 ~30 token 的 chosen response 的 sequence logp 量级（提示：sft50 训练末期 loss ≈ 0.97） | |
| B4 ✏ | 训练中 logπ(chosen) 会单调上升吗？margin 上升时 chosen/rejected 的 logp 各自可能怎么走？ | |
| B5 ✏ | sum（而非 mean）的 sequence logp：如果 rejected 系统性地比 chosen 长 2 倍，margin 和梯度会有什么系统性偏差？ | |

## C. 实测锚点（harness 在 sft50 上测）

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | `sequence_logprob` 输出的 shape 和 dtype（B=4 对，即输入 8 行） | |
| C2 | 对 "What is 2+2?" → chosen "4."：per-token 平均 logp 大约多少 nats？rejected "Seven, probably."（不通顺+错）的会更低多少个量级？ | |
| C3 | 显存：载入 policy 后再载入 ref，`memory_allocated` 增量 ≈ 多少 GiB？ | |
