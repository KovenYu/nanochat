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
| A1 | 每层 `self_attn.q_proj.weight` 的 shape（注意 head_dim=128 和 hidden_size=1024 的关系） | 1024,2048|
| A2 | 每层 `self_attn.k_proj.weight` 的 shape | 1024,1024|
| A3 | 每层 `self_attn.o_proj.weight` 的 shape | 2048,1024|
| A4 | `self_attn.q_norm.weight` 的 shape（QK-norm 的可学习参数作用在哪个维度上） | hidden dimension上，也就是2048|
| A5 | 每层 `mlp` 三个矩阵（gate/up/down）各自的 shape | gate,up: 1024,3072, down: 3072,1024 |
| A6 | 每层有多少个 tensor？整个文件（28 层 + 头尾）总共多少个？ | q,k,v,o,qnorm,knorm,gate,up,down, 总共9个每层。28*9+1+1=254|
| A7 | `tie_word_embeddings=true`：`model.safetensors` 里有没有 `lm_head.weight` 这个 key？ | 没有，就是embedding|
| A8 | 参数量：按 A1–A5 算账（embedding + 28 层 + final norm），给出 unique 参数总数；它为什么叫 0.6B？ | embedding是151K*1K=150M左右，attn一层是1K*2K+1K*1K*2+2K*1K=6M (norm忽略), 28层就是180M左右，MLP是1K*3K*2+3K*1K=9M，28层就是250M左右。这样总共就是600M，所以是0.6B |
| A9 | `model.safetensors` 的文件大小（±20%；A7 的答案会影响） | 假设optimizer states也都存下来，那就是16byte/param, 那就是16*0.6=10GB左右|
| A10 | 文件里 norm 的 weight 用什么 dtype 存？（nanochat checkpoint 里非 embedding 参数是 fp32） | bf16，我看整个model都是bf16存的，说明用的是你之前说过的master weight 的那个工业格式，这也是我用 16byte/param的原因。|

## B. 机制差异 vs `lab1_gpt.py`（每题 yes/no + 一句机制）

| # | 问题 | 你的预测 |
|---|---|---|
| B1 | RMSNorm 有没有可学习 scale？统计量（均方）在什么 dtype 下算？ | 有，fp32 |
| B2 | QK-norm 作用在 RoPE 之前还是之后？如果 norm **没有**可学习 weight，两种顺序等价吗？**有** weight 呢？ |之前。没有weight等价，有weight不等价。 |
| B3 | nanochat 的 rotary 是 y1 = x1·cos + x2·sin, y2 = −x1·sin + x2·cos；HF 是 y = x·cos + rotate_half(x)·sin，rotate_half(x) = cat(−x2, x1)。二者是同一个旋转吗？把一个训练好的 checkpoint 用另一种约定去跑，top-1 预测会怎样？ | 不是同一个旋转，两者相反。预测会不对，因为qk乘积在norm了之后其实就是cos两者的夹角。假设这个夹角是10度，rope原本要把q不动，k转-10，现在会变成q不动，k转10，那角度就从0变成了20，所以不对了。|
| B4 | rope_theta 1e6 vs nanochat 1e5：最低频那对通道的波长（走完 2π 需要多少个 position）各是多少？对 max_position_embeddings=40960 意味着什么？ | 角频率是1e-6, 走完6.28需要 6.28*1e6,这意味着40960在最低频连1%波长都走不完。|
| B5 | MLP：SwiGLU intermediate 3072 = 3C，三个矩阵；nanochat relu² 是 4C 两个矩阵。每层 MLP 参数量之比 Qwen3 / nanochat（同 C）？ |9c/8c=9/8 |
| B6 | attention 的 scale 是多少？q/k 还有 nanochat 那样的 ×1.2 吗？QK-norm 带可学习 weight 后，attention logit 还有 lab1 B4 那样的硬上界吗？ |scale就是sqrtD乘上可学的scale，没有1.2了。没有硬上界了。|
| B7 | 输入 embedding 过不过 norm？logits 有没有 softcap？ | 都没有。|
| B8 | loss 的 ignore_index 是多少？（nanochat 是 −1） | -100|
| B9 | 全 bf16 权重下，前向里哪些步骤在 fp32 里算？候选：RMSNorm 统计量、RoPE 表、SDPA 内 softmax、logits、loss |RMSNorm 统计量，SDPA 内 softmax、logits、loss |

## C. Chat template 与 loss mask（Phase 1 直接相关）

对话 M = [user "hi", assistant "hello", user "again?", assistant "yes"]。

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | HF `apply_chat_template(M)` 渲染出的字符串（逐字，含换行；提示：template 里 `<think>` 的插入条件和 `loop.last` 有关） |这个我们不是已经讲过了吗，就是im start 跟 role，im end 跟换行；think 只在最后一个turn出现（这里是空的think内容） |
| C2 | 序列开头有没有 BOS？`<|im_end|>` 与 `<|endoftext|>` 各是 eos 还是 pad？ | 无bos；不知道这个问题是什么意思|
| C3 | LLaMA-Factory 的 `qwen3` template（enable_thinking 默认 true）对同一 M 渲染的 token 序列与 C1 相同吗？不同在哪？ |不同；HF只有最后一轮有think |
| C4 | LLaMA-Factory 下哪些 token 带 loss：assistant header `<|im_start|>assistant\n`？空 think 块？内容？`<|im_end|>`？它后面的 `\n`？ |内容+im end+换行；enable thinking 时还加上think 块内容 |
| C5 | `enable_thinking: false` 时，渲染和 mask 各变了什么？ | false时不带think块，这部分的mask也随之消失|
| C6 | 用 C3/C4 的规则算：M 的总 token 数，其中带 loss 的有几个（空 think 块按 4 个 token 计：`<think>` `\n\n` `</think>` `\n\n`；"hi"、"hello"、"again?"、"yes" 各按 1 个 token 计，"again?" 按 2 个） | 懒得算了，你直接说有什么caveat|

## D. 量级（Phase 1 的 8B 一起算）

| # | 问题 | 你的预测 |
|---|---|---|
| D1 | Qwen3-0.6B bf16 权重占显存（tie 之后） |1G左右 |
| D2 | KV cache 每 token 字节数（bf16，28 层，用 A2 的 kv 维度） | 一token一层是2K个数，就是4KB，28层就是100KB量级。如果ctx是32K，那就是32K*100KB=3200MB，3.2GB，好像也不多。不过这个大概会随着模型参数几乎线性上涨，考虑一个8B的模型，可能KV cache就是32GB量级了。|
| D3 | 全参 SFT：bf16 权重 + fp32 master + AdamW 两个 fp32 moment + bf16 grad = 每参数多少字节？0.6B 和 8B（8.2B unique）各需要多少 GB（不含激活） | 16byte/param这个早就算过了。0.6B不算activation也有10GB，8B就是130GB左右|
| D4 | 8B 在 4×H200（4×141GB）上全参可不可以？LoRA 呢？（一句话） |可以full，ZERO1切optimizer state应该就够了。LORA那更可以了。 |
