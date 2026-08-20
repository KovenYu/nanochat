# Lab 1 · predictions.md

规则同 lab0：只填「你的预测」列，填完前不跑 harness。
数值题允许写量级或区间；shape 题要求精确。
按 SPEC 的 scale-relevance filter，本表只考 scale-invariant 的机制与量级。

## A. tiny 配置下的 shape / dtype 流

配置（harness 逐字一致）：`GPTConfig(sequence_len=16, vocab_size=256, n_layer=2,
n_head=4, n_kv_head=2, n_embd=64)`，输入 `idx: (B=2, T=16) int64`，带 targets。
（head_dim = 64/4 = 16，GQA 2 组。）

| # | tensor | shape 预测 | dtype 预测 |
|---|---|---|---|
| A1 | `wte(idx)` 的输出 | | |
| A2 | `c_q(x)` view 之后的 q | | |
| A3 | `c_k(x)` view 之后的 k（注意 GQA） | | |
| A4 | forward 里实际用到的 `cos`（切片后） | | |
| A5 | 单层单头的 attention 权重矩阵（softmax 后、乘 v 前，SDPA 内部） | | （不用填 dtype）|
| A6 | MLP 的隐层（relu² 之后那个 tensor） | | |
| A7 | 返回的 `logits`（无 targets 时） | | |
| A8 | 返回的 `loss`（有 targets 时） | | |

## B. RoPE 与 attention logits 的量级（物理题）

对上述 tiny 配置、`init_weights` 后的真实前向（harness 会逐条实测）：

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | `norm(q) * 1.2` 之后，q 的每个 head 向量的 RMS 是多少 | |
| B2 | `apply_rotary_emb` 前后，k 向量的 L2 范数的相对变化量级（\|Δ‖k‖\|/‖k‖） | |
| B3 | 为什么是 B2 那个量级？（一句话，说机制） | |
| B4 | attention logit（`q·k/√D`，softmax 前）的**理论上界**（用 B1 的结果推，D=16） | |
| B5 | attention logit 的**典型值**（随机初始化、q k 近似不相关时的 std 量级） | |
| B6 | `lm_head` 输出经 softcap 后 logits 的可能取值范围 | |
| B7 | 训练 5568 步之后（d24 真模型），B5 的典型值会变大还是变小？为什么？ | |

## C. d24 checkpoint 解剖（真实制品，数量级会考精确一些）

已知：`--depth=24`，其余用 scripts/base_train.py 默认值（aspect_ratio=64,
head_dim=128, max_seq_len=2048, vocab=32768）。冻结目录
`base_checkpoints/d24-baseline-26413dc/` 内有 model_005568.pt、meta_005568.json、
optim_005568_rank{0..3}.pt。

先推配置（C1-C3），再算账（C4-C8）：

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | d24 的 n_embd（按 base_train.py:129-135 的规则推） | |
| C2 | d24 的 n_head 和 head_dim | |
| C3 | d24 是 MHA 还是 GQA？ | |
| C4 | 总参数量（±20% 内给个数，写出你的分项账：wte + lm_head + blocks + 其他） | |
| C5 | model_005568.pt 的文件大小（±20%；注意不同参数组的存储 dtype 不同） | |
| C6 | checkpoint 的 state_dict 里有没有 norm 层的权重？有没有 cos/sin？ | |
| C7 | 为什么 optim 文件有 4 个 rank 分片？里面大致存的是什么？ | |
| C8 | 4 个 optim 分片的**总大小**相对 model 文件的倍数（想想 Muon 和 AdamW 各存几份状态、什么 dtype） | |

## D. KV cache 的量级（挑战 ③ 的前置感知）

对 d24 真实配置（用你 C1-C3 的答案），bf16 cache：

| # | 问题 | 你的预测 |
|---|---|---|
| D1 | 缓存一个 token 的 K+V（全部 24 层）要多少字节 | |
| D2 | 一条 2048 token 的对话，KV cache 总共多少 MB | |
| D3 | 如果 d24 改成 GQA n_kv_head=n_head/4，D2 变成多少 | |
