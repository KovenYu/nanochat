# Lab 3 · predictions.md

规则同前。对照文件：labs/mini/lab3_dpo.py。policy = sft50 (d24, step 233)，ref = 同权重冻结副本。
✏ 行要求写推导/理由。B4/B5 是纯推理题（实测在 Stage 3 的 β 扫描里兑现）。

## A. 双前向的计算图与显存

| # | 问题 | 你的预测 |
|---|---|---|
| A1 ✏ | 训练时 policy 侧的显存(参数+梯度+optimizer state) vs ref 侧的显存，各多少 GiB？（lab1 C 组的账本直接用） | 训练时policy是16Byte/param, ref 只有模型本体bf16，所以是2B/param；我不记得具体是多少GB因为参数量我忘记了，但是比例是这样的|
| A2 | 四个 logp 向量 pi_c, pi_r, ref_c, ref_r 中哪些在 autograd graph 上？loss 的梯度经由哪几个流回模型？ | pi_c和pi_r在autograd。loss梯度也是经由这两个流回policy。|
| A3 | `logp.gather(-1, safe.unsqueeze(-1))` —— 如果误写成 `gather(1, ...)`，会报错还是静默出错？错的话错在哪一维的语义上？ |不会报错。会在同一个位置的、来自于不同example的token做gather，但由于mask都是0或1所以最后会变成整个batch都replicate了第一个和第二个example的tokens |
| A4 ✏ | 为什么 prompt 的 token 必须排除在 sequence_logprob 之外？（想：留着它，margin 会多出什么项？） |留着的话，margin里会包括ref(prompt)和policy(prompt)的概率，虽然这俩本身不会搞出梯度，但是会引入错误的scale甚至是错误的方向 |

## B. β 与 margin 的量级/动力学

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | 第 0 步（policy == ref）：implicit reward r_c、r_r、margin、以及 loss 的精确值 | reward都是0，margin也是0，loss是0.3多一点 |
| B2 ✏ | β 从 0.1 调到 0.5：policy 被拴得更紧还是更松？说机制 |beta越大，越紧。因为sigmoid会让reward差得大的时候，gradient变小，大的beta会把reward撑得虚高，那policy就没法漂太远。 |
| B3 | sft50 对一条 ~30 token 的 chosen response 的 sequence logp 量级（提示：sft50 训练末期 loss ≈ 0.97） | 0.97*30 左右，可能比这个稍微高一些|
| B4 ✏ | 训练中 logπ(chosen) 会单调上升吗？margin 上升时 chosen/rejected 的 logp 各自可能怎么走？ |logp chosen大概不会单调上升，margin上升有可能通过让reject 下降比chosen下降更多来实现 |
| B5 ✏ | sum（而非 mean）的 sequence logp：如果 rejected 系统性地比 chosen 长 2 倍，margin 和梯度会有什么系统性偏差？ | 如果reject长2倍，那么reject这边的reward就会偏大（项数更多），chosen那边的reward偏小；margin没什么偏差，但是梯度会偏向于压reject的概率，分给其他tokens。最后的结果可能是，chosen本身没学会，回答的随机性反而增加了。|

## C. 实测锚点（harness 在 sft50 上测）

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | `sequence_logprob` 输出的 shape 和 dtype（B=4 对，即输入 8 行） | shape=(8,) dtype=fp32|
| C2 | 对 "What is 2+2?" → chosen "4."：per-token 平均 logp 大约多少 nats？rejected "Seven, probably."（不通顺+错）的会更低多少个量级？ | SFT结束时是0.97，那logp就是差不多-0.97；rejected 每个token接近于random，32K vocab size 下可能是-10左右，差了一个量级|
| C3 | 显存：载入 policy 后再载入 ref，`memory_allocated` 增量 ≈ 多少 GiB？ | 就是上面说的啊|
