# Lab 2 · predictions.md

规则同前：只填「你的预测」列。数值题给量级或区间，标 ✏ 的行要求写出推导/理由。
对照文件：labs/mini/lab2_sft.py。

## A. mask 语义（render + shift 之后，监督落在哪）

对这条对话（harness 逐字一致）：

```python
CONV = {"messages": [
    {"role": "user",      "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user",      "content": "Bye"},
    {"role": "assistant", "content": "Ok"},
]}
```

设它单独占一行（无 packing 干扰），`inputs = row[:-1]`，`targets = row[1:]`。
考虑**input 位置 p 上的 token 是什么**，预测该位置的 target 是否被监督（-1 还是真 id）：

| # | input 位置 p 上的 token | target[p] 被监督吗 | 你的预测 |
|---|---|---|---|
| A1 | `<\|user_end\|>`（第一轮） | assistant start | no |
| A2 | `<\|assistant_start\|>`（第一轮） | Hello的第一个token |yes |
| A3 | "Hello" 的最后一个 token | assistant end |yes |
| A4 | `<\|assistant_end\|>`（第一轮） | user start |no |
| A5 | 整行被监督的 target 总数 = ？（用 "Hello"=1 token、"Ok"=1 token、"Hi"/"Bye"=1 token 假设算精确值） | 4，就是hello, assistant end, ok, assistant end| |

## B. packing 行为

row_capacity = 33（即 T=32）。对话流依次给出长度 [30, 12, 20, 8, 3]，之后全部是长度 31。

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | 第 1 行装入的对话长度序列（按装入顺序）、padding 数 | 30，3|
| B2 | 第 2 行装入的长度序列、padding 数、`padded_at` | 31，padding=2|
| B3 | 第 2 行 targets 里由 padding 贡献的 -1 个数 | 2|
| B4 ✏ | `for i, content_len in enumerate(row_lengths)` 那段显式 padding 屏蔽——它屏蔽的位置里，有没有哪个是 `mask[1:]` 那条路**没有**覆盖到的？（提示：padding token 的 mask 是什么） | 没看懂问题是什么|

## C. grad accumulation 等价性 ✏

设 grad_accum=2，micro-batch 1 有 100 个有效 target（非 -1），micro-batch 2 有 300 个。
对照组：把两个 micro-batch 拼成一个大 batch 一次 forward/backward。

| # | 问题 | 你的预测 |
|---|---|---|
| C1 ✏ | `(loss/2).backward()` 累积出的梯度与大 batch 的梯度**严格相等**吗？不等的话，两条路各自等价于什么加权？ | 不严格相等。第一个micro里各target对应的loss是 1/100的权重，loss/2后等价于1/200的权重；第二个micro则是1/600的权重。但是在大batch里，每个target都得到1/400的权重。换句话说，在前者的情况里，有效target少的micro batc里的几个conversation的重要性被放大了|
| C2 | 若把 loss_reduction 换成 'sum'、最后统一除以总有效 target 数，等价性恢复吗 | 应该基本等价，如果不考虑数值精度上的微小损失|
| C3 | 两个 micro-batch 的先后顺序影响累积结果吗（数学上） | 数学上不影响，实现上会受精度有微小影响|

## D. 显存账（±10% 闭合，物理题重头）

配置（harness 逐字一致，单 H200，bf16）：tiny 模型
`GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)`，
**B=16, T=128**，grad_accum=1，MuonAdamW，训 3 步后读 `torch.cuda.max_memory_allocated()`。

提示：先列参数清单（wte/lm_head/ve/blocks/scalars 的数量与 dtype），再分四块记账。

WTE=32768*64=2^21, dtype=bf16
lm_head=2^21, dtype=fp32
ve (skip becoz it is scale-dependent)
blocks: qkv+proj: 2*64*64*4=2^15
linear: 2*64*4*64*2=2^16
dtype=fp32
scalars also skip

| # | 记账项 | 你的预测（MiB） |
|---|---|---|
| D1 | 参数本体（注意 bf16/fp32 混合） |wte-> 2^21*2byte=2^22B=4GB; lm_head->8GB; blocks: 1.5*2^16*4B=375MB. 总共12GB出头 |
| D2 | 梯度 | 都是bf16，所以是差不多8GB出头|
| D3 | optimizer state（AdamW m+v 的 dtype 跟随参数；Muon momentum + factored 二阶） | wte, lm_head 是adamw，所以是2*4GB+2*8GB，就是24GB左右；blocks都是muon，就还是375MB，加起来就是24GB出头|
| D4 ✏ | 峰值 activation 的**最大单项**是什么 tensor？多少 MiB？（想想 fused-CE 那次讨论） | 是从bf16升到fp32之后的logits，是16*128*32768*4B=2^28B=256GB|
| D5 | `max_memory_allocated` 总峰值 | 感觉300GB左右？这样H200不是爆炸了吗|
