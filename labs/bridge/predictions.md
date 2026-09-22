# 桥接练习 · predictions.md（Qwen3-0.6B vs nanochat）

规则同前几个 lab：只填「你的预测」列，填完前不跑 harness。
数值题允许写量级或区间；shape 题要求精确。参照物是 HF transformers 的
`modeling_qwen3.py`（`labs/mini/bridge_qwen3.py` 里以 `mq3:<line>` 引用）和
Qwen3-0.6B 的 `config.json`。运行环境是 Phase A 的 env（`source /svl/u/koven/sft-phase-a/env.sh`）。

`config.json` 关键字段（逐字）：

```
hidden_size 1024      intermediate_size 3072     num_hidden_layers 28
num_attention_heads 16   num_key_value_heads 8   head_dim 128
vocab_size 151936     rms_norm_eps 1e-6          rope_theta 1000000
tie_word_embeddings true   torch_dtype bfloat16   max_position_embeddings 40960
hidden_act silu       attention_bias false       sliding_window null
```

## A. 从 config 推 checkpoint 里的 tensor（不看文件）

| # | 问题 | 你的预测 |
|---|---|---|
| A1 | 每层 `self_attn.q_proj.weight` 的 shape（注意 head_dim=128 和 hidden_size=1024 的关系） | |
| A2 | 每层 `self_attn.k_proj.weight` 的 shape | |
| A3 | 每层 `self_attn.o_proj.weight` 的 shape | |
| A4 | `self_attn.q_norm.weight` 的 shape（QK-norm 的可学习参数作用在哪个维度上） | |
| A5 | 每层 `mlp` 三个矩阵（gate/up/down）各自的 shape | |
| A6 | 每层有多少个 tensor？整个文件（28 层 + 头尾）总共多少个？ | |
| A7 | `tie_word_embeddings=true`：`model.safetensors` 里有没有 `lm_head.weight` 这个 key？ | |
| A8 | 参数量：按 A1–A5 算账（embedding + 28 层 + final norm），给出 unique 参数总数；它为什么叫 0.6B？ | |
| A9 | `model.safetensors` 的文件大小（±20%；A7 的答案会影响） | |
| A10 | 文件里 norm 的 weight 用什么 dtype 存？（nanochat checkpoint 里非 embedding 参数是 fp32） | |

## B. 机制差异 vs `lab1_gpt.py`（每题 yes/no + 一句机制）

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | RMSNorm 有没有可学习 scale？统计量（均方）在什么 dtype 下算？ | |
| B2 | QK-norm 作用在 RoPE 之前还是之后？如果 norm **没有**可学习 weight，两种顺序等价吗？**有** weight 呢？ | |
| B3 | nanochat 的 rotary 是 y1 = x1·cos + x2·sin, y2 = −x1·sin + x2·cos；HF 是 y = x·cos + rotate_half(x)·sin，rotate_half(x) = cat(−x2, x1)。二者是同一个旋转吗？把一个训练好的 checkpoint 用另一种约定去跑，top-1 预测会怎样？ | |
| B4 | rope_theta 1e6 vs nanochat 1e5：最低频那对通道的波长（走完 2π 需要多少个 position）各是多少？对 max_position_embeddings=40960 意味着什么？ | |
| B5 | MLP：SwiGLU intermediate 3072 = 3C，三个矩阵；nanochat relu² 是 4C 两个矩阵。每层 MLP 参数量之比 Qwen3 / nanochat（同 C）？ | |
| B6 | attention 的 scale 是多少？q/k 还有 nanochat 那样的 ×1.2 吗？QK-norm 带可学习 weight 后，attention logit 还有 lab1 B4 那样的硬上界吗？ | |
| B7 | 输入 embedding 过不过 norm？logits 有没有 softcap？ | |
| B8 | loss 的 ignore_index 是多少？（nanochat 是 −1） | |
| B9 | 全 bf16 权重下，前向里哪些步骤在 fp32 里算？候选：RMSNorm 统计量、RoPE 表、SDPA 内 softmax、logits、loss | |

## C. Chat template 与 loss mask（Phase 1 直接相关）

对话 M = [user "hi", assistant "hello", user "again?", assistant "yes"]。

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | HF `apply_chat_template(M)` 渲染出的字符串（逐字，含换行；提示：template 里 `<think>` 的插入条件和 `loop.last` 有关） | |
| C2 | 序列开头有没有 BOS？`<|im_end|>` 与 `<|endoftext|>` 各是 eos 还是 pad？ | |
| C3 | LLaMA-Factory 的 `qwen3` template（enable_thinking 默认 true）对同一 M 渲染的 token 序列与 C1 相同吗？不同在哪？ | |
| C4 | LLaMA-Factory 下哪些 token 带 loss：assistant header `<|im_start|>assistant\n`？空 think 块？内容？`<|im_end|>`？它后面的 `\n`？ | |
| C5 | `enable_thinking: false` 时，渲染和 mask 各变了什么？ | |
| C6 | 用 C3/C4 的规则算：M 的总 token 数，其中带 loss 的有几个（空 think 块按 4 个 token 计：`<think>` `\n\n` `</think>` `\n\n`；"hi"、"hello"、"again?"、"yes" 各按 1 个 token 计，"again?" 按 2 个） | |

## D. 量级（Phase 1 的 8B 一起算）

| # | 问题 | 你的预测 |
|---|---|---|
| D1 | Qwen3-0.6B bf16 权重占显存（tie 之后） | |
| D2 | KV cache 每 token 字节数（bf16，28 层，用 A2 的 kv 维度） | |
| D3 | 全参 SFT：bf16 权重 + fp32 master + AdamW 两个 fp32 moment + bf16 grad = 每参数多少字节？0.6B 和 8B（8.2B unique）各需要多少 GB（不含激活） | |
| D4 | 8B 在 4×H200（4×141GB）上全参可不可以？LoRA 呢？（一句话） | |
