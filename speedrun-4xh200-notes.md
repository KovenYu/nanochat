# nanochat speedrun on 4×H200 — 对话记录

> 记录范围：从 speedrun 跑完后的结果澄清开始，到 tokenizer 原理解释为止。
> 日期：2026-08-12。硬件：4×H200（原版配方为 8×H100）。

---

## 🧑 User

> 怎么才18分钟？上面的估计不是3个小时吗？这个speedrun 本身不是说要8xH100跑4个小时吗？

---

## 🤖 Claude

两个都需要澄清一下，而且有个更重要的事：**整个 speedrun 早就跑完了。**

日志最后写入是 **05:06**，现在 15:51 —— 它在 10 小时前就结束了，screen session 也已正常退出。

### 关于「才 18 分钟」

18 分钟是 **SFT 这一个阶段**，不是整个 run。SFT 只有 467 步，是在已经训好的 base model 上做轻量对齐；pretraining 是 5568 步、从随机初始化开始学，两者不是一个量级。完整时间线：

| 阶段 | 耗时 |
|---|---|
| 数据（已缓存）+ tokenizer 训练 | ~20 min |
| **base_train（pretraining）** | **184.3 min** |
| base_eval | ~14 min |
| chat_sft | 18.0 min |
| chat_eval | ~13 min |
| **总计 00:55 → 05:06** | **4 小时 11 分** |

> ⚠️ 后文修正：第一行实际只有 ~1.5 min，那 ~20 min 在 base_train 内部（周期性评测 + compile）。详见最后一节。

### 关于「8×H100 要 4 小时」

这个前提不对。README 写的是 **~1.5 小时**，leaderboard 上「time to GPT-2」最新纪录是 **1.65 小时**（8×H100）。没有 4 小时的说法。

你的 4×H200 用了 4h11m，相对 8 卡的 ~1.65h 是 2.5 倍——比单纯的「卡数减半 → 时间翻倍」多一些，因为评测阶段（base_eval / chat_eval）不随卡数线性缩放。

我之前估的是「3.5~4 小时」，实际 4h11m，**我低估了评测阶段的开销**。不过仍在你 8 小时预算内。

### 最终结果

**Base model（pretraining）**

| | 结果 | 参考 |
|---|---|---|
| val bpb | 0.7190 | 0.71808 |
| **CORE** | **0.2680** | GPT-2 = 0.256525 ✅ |

**Chat model（SFT 后）**

| 任务 | 分数 |
|---|---|
| ARC-Easy | 60.73% |
| ARC-Challenge | 47.87% |
| MMLU | 35.83% |
| GSM8K | 2.65% |
| HumanEval | 12.80% |
| **ChatCORE** | **0.2161** |

Checkpoint 在 `/svl/u/koven/nanochat_data/chatsft_checkpoints/d24/`。可以直接聊天了：

```bash
cd /svl/u/koven/nanochat
source runs/env_4xh200.sh && source .venv/bin/activate
python -m scripts.chat_cli
```

（GSM8K 2.65% 是这个规模的正常水平——4e19 FLOPs 的模型基本不会做多步算术。）

---

## 🤖 Claude（监控收尾）

监控已关闭。

还有两件之前问过、你没回的小事，现在收尾正好一起处理（不做也不影响任何东西）：

1. 那两个 commit 在 `fix-mfu-fp8` 分支上，master 还没动。要合的话 `git merge fix-mfu-fp8` 是 fast-forward。
2. `runs/env_4xh200.sh` 已提交，但它依赖的 `speedrun.sh` 改动按你的意思没提交，所以在干净 checkout 上它设的 `NANOCHAT_BASE_DIR` 会被覆盖回 `$HOME/.cache/nanochat`。

另外 `runs/speedrun.log`（4.2MB）和 `speedrun.log.attempt1` 还是 untracked 状态，`.gitignore` 里没有 `runs/*.log`，需要的话我加一行。

---

## 🧑 User

> val bpb 是什么东西，跟loss的区别是？CORE是什么？现在的pretrain 5568步的 LR schedule是什么样的？checkpoint存盘时你提到的"模型采样"我在哪里看原始数据？你说的"4e19 FLOPS模型本来就是这水平"是什么意思——怎么得到的4E19，以及4E19的水平应该是如何？更高的FLOP量级后大概怎么对应更强的能力？最后你report了一些eval的分数，但我不认识这些benchmark，你简要给我介绍一下。两件小事的处理到最后等我提到"我要结束这个session"的时候再说。

---

## 🤖 Claude

好问题，一个个来。先把代码和日志翻出来，避免我凭印象说。

### 1. `val bpb` vs `loss`

**loss** 是训练时的交叉熵，单位是 nats/**token**。问题在于它依赖 tokenizer：换个 vocab size，同一个模型的 loss 就变了——token 变长则每个 token 的 loss 变大，但模型没变好也没变坏。所以 loss 只能在同一 tokenizer 内部比较。

**bpb (bits per byte)** 把分母从"token 数"换成"字节数"，单位是 bits/**byte**。`nanochat/loss_eval.py:64`：

```python
bpb = total_nats / (math.log(2) * total_bytes)
```

分子累加 nats 后除以 `ln2` 转成 bits，分母是**目标 token 所代表的原始字节数**。这样 tokenizer 怎么变都不影响——衡量的是"压缩这段文本需要几 bit/字节"，是信息论意义上的绝对量。另外它还做了两件事（`loss_eval.py:19-21`）：special token（如 `<|bos|>`）和被 mask 的位置都不计入。

再加个 **val**：算在 held-out 验证集上，不是训练数据，所以反映泛化而非记忆。

一句话：**loss 是训练信号，val bpb 是可跨配置比较的评价指标**。这就是为什么 leaderboard 用 bpb 而不用 loss。

### 2. CORE 是什么

CORE 来自 DCLM 论文，是 **22 个任务的"去随机基线"平均分**。核心在 `scripts/base_eval.py:110`：

```python
centered_result = (accuracy - 0.01*random_baseline) / (1.0 - 0.01*random_baseline)
```

为什么要 centering：4 选 1 的题瞎猜就有 25%，而 LM 补全类任务瞎猜是 0%。直接平均会让选择题虚高。centering 把每个任务重标定成"**0 = 瞎猜水平，1 = 满分**"，然后 22 个任务等权平均。

你这次的完整结果在 `nanochat_data/base_eval/base_model_005568.csv`：

| 任务 | Accuracy | Centered |
|---|---|---|
| hellaswag | 0.5546 | 0.4061 |
| arc_easy | 0.6864 | 0.5819 |
| piqa | 0.7476 | 0.4951 |
| lambada_openai | 0.4300 | 0.4300 |
| squad | 0.4365 | 0.4365 |
| arc_challenge | 0.4002 | 0.2002 |
| commonsense_qa | 0.2531 | 0.0663 |
| **boolq** | 0.5945 | **−0.0671** |
| bigbench_repeat_copy_logic | 0.0313 | 0.0313 |
| …（共 22 个） | | |
| **CORE** | | **0.267960** |

注意 **boolq 是负的**：它的随机基线是 62%（多数类猜 yes 就有这么高），模型只有 59.4%，**低于瞎猜**。这正是 centering 的价值——不做归一化的话 boolq 的 0.59 看起来还挺像样。

### 3. LR schedule

`scripts/base_train.py:359-368`，三段式：**linear warmup → constant → linear warmdown**。参数在 67-69 行：`warmup_steps=40`, `warmdown_ratio=0.65`, `final_lr_frac=0.05`。

代入 `num_iterations = 5568`：

| 阶段 | 步数区间 | lrm |
|---|---|---|
| warmup | 0 → 40 | `(it+1)/40`，从 0.025 线性升到 1.0 |
| constant | 40 → 1949 | 恒定 1.0 |
| warmdown | 1949 → 5568 | 线性降到 **0.05**（不是 0） |

`warmdown_iters = round(0.65 × 5568) = 3619`，所以 constant 段结束于 `5568 − 3619 = 1949`。**65% 的训练步都在衰减学习率**，这是现代 speedrun 配方的特征。

用日志核对过，公式对得上：step 2500 算得 0.855（日志 0.86），step 4500 算得 0.330（日志 0.33），step 5567 是 0.05。

另外还有两条同步的 schedule：

- **Muon momentum**（371-380 行）：前 400 步从 0.85 升到 0.97，warmdown 期间再降到 0.90
- **weight decay**（383-385 行）：`0.5·(1+cos(π·it/N))` 余弦衰减到 0

（顺便修正我之前一句话：我说过 warmdown 是 "lrm → 0"，实际是 → 0.05。）

### 4. 采样的原始数据在哪

就在 `runs/speedrun.log` 里。`base_train.py:457` 每 2000 步（`--sample-every` 默认 2000）+ 最后一步采样一次，7 个固定 prompt，greedy（`temperature=0`）、`max_tokens=16`。

```bash
cd /svl/u/koven/nanochat
grep -n "^<|bos|>" runs/speedrun.log        # 全部 36 条
sed -n '6059,6065p' runs/speedrun.log       # step 5568（最终）的 7 条
sed -n '2416,2422p' runs/speedrun.log       # step 2000，可以对比进步
sed -n '6089,6132p' runs/speedrun.log       # base_eval 自己那轮采样
```

prompt 列表硬编码在 `base_train.py:459-467`（首都、金的化学符号、星期推理、反义词、太阳系行星、喜欢的颜色、解一元一次方程）。

### 5. FLOPs 是怎么算的，以及 4e19 这个数

**先更正**：我之前那句 "4e19 FLOPs 的模型" 是直接引 README 的，**你这次 run 的实际数字是 ~2.8e19，不是 4e19**。

算法在 `nanochat/gpt.py:338`：

```python
num_flops_per_token = 6 * num_matmul_params + attn_flops
```

- `6 × N`：每个矩阵乘参数，forward 贡献 2 FLOPs（乘 + 加），backward 是 forward 的 2 倍，共 6。这就是经典的 **6ND** 法则。
- `attn_flops`：注意力里 QK 和 AV 那两个矩阵乘，参数量无关、只跟序列长度有关，得单独加。

代入你的 run：

```
N (matmul params) = 729,810,648
flops/token       = 4.775226e9
D (tokens)        = 5,838,471,168
总计 = 4.775e9 × 5.838e9 ≈ 2.79e19 FLOPs
```

（日志里打印的 `2.314208e+18` 是被那个 fp8 bug 压低了 12 倍的错值，已修。）

README 说 "4e19" 大概是针对更早的配置（d26，或不同的 token:param 比）写的，跟当前 `speedrun.sh` 的 d24 + ratio 8 对不上。以你自己的日志为准。

#### 2.8e19 是什么水平

有意思的对比：**GPT-2 (1.5B) 自己的训练量大约是 1e20 FLOPs**（估算：1.5B 参数 × ~10B tokens × 6；OpenAI 没公布确切数字）。你用 **~1/4 的算力打平了它的 CORE 分数**——这就是 7 年间 tokenizer、数据质量、架构、优化器（Muon）、精度（fp8）全栈进步的累积效果。

粗略的量级阶梯（前两个是估算，后两个论文有公布）：

| 训练 FLOPs | 代表 | 能力特征 |
|---|---|---|
| ~3e19 | **你这次的 d24** | 语法流畅、常识零散；几乎不能多步推理或算术 |
| ~1e20 | GPT-2 1.5B (2019) | 同上，能写连贯段落但很快跑题 |
| 3.1e23 | GPT-3 175B | few-shot in-context learning 明显起作用 |
| ~4e25 | Llama 3 405B | 可靠的多步推理、代码、工具调用 |

需要说清楚的是：**bpb 随算力是平滑的幂律下降，但下游能力是阶梯式的**。多位数算术、多步推理这类能力在某个规模前基本是零，过了阈值突然可用——这就是为什么你的模型 GSM8K 只有 2.65%，它不是"算得不太准"，而是**根本没到能做多步算术的规模**。

想自己量化这条曲线的话，repo 里有现成的：`runs/scaling_laws.sh` 和 `runs/miniseries.sh`，扫 `--depth` 就能画出 bpb-vs-FLOPs 曲线（README 顶部那张 `dev/scaling_laws_jan26.png` 就是这么来的）。d12 只要 ~5 分钟。

### 6. Benchmark 简介

你的 chat model 最终成绩，配上瞎猜基线看才有意义：

| Benchmark | 你的分数 | 瞎猜 | 是什么 |
|---|---|---|---|
| **ARC-Easy** | 60.73% | 25% | 小学科学选择题，4 选 1。"哪个是导体？" 这种。**明显高于瞎猜** |
| **ARC-Challenge** | 47.87% | 25% | 同来源但专挑检索式方法做错的题，需要一点推理 |
| **MMLU** | 35.83% | 25% | 57 个学科（法律/医学/数学/历史…）的 4 选 1，衡量知识广度。**只比瞎猜高 11 个点，基本等于不会** |
| **GSM8K** | 2.65% | 0% | 小学应用题，要写出多步算式。开放式作答，**这个规模基本做不了** |
| **HumanEval** | 12.80% | 0% | 164 道 Python 函数补全，跑单元测试判对错（pass@1）。12.8% 意味着 164 题对了 21 题 |
| **ChatCORE** | 0.2161 | 0 | 上述任务做完 centering 后的汇总，同 CORE 的思路 |

对照一下 CORE 里那 22 个任务的主要类型：

- **常识推理**：hellaswag（选合理的句子续写）、piqa（物理常识）、copa（因果）、winograd / winogrande（代词消歧）
- **知识问答**：jeopardy、bigbench_qa_wikidata、squad、coqa（阅读理解）
- **语言建模**：lambada_openai（预测长段落的最后一个词，考长程依赖）
- **符号/算法**：bigbench_dyck_languages（括号匹配）、bigbench_operators、bigbench_repeat_copy_logic——你在这几项上分数都极低（0.03~0.18），符合上面说的"阈值未到"

一个实用的读法：**看 centered 而不是 accuracy**。piqa 的 0.7476 看着比 lambada 的 0.4300 高得多，但 centered 后是 0.4951 vs 0.4300，差距其实很小——因为 piqa 是二选一，一半分是白送的。

---

两件小事按你说的先搁着，等你说结束 session 再处理。

---

## 🧑 User

> 为什么会有一部分时间去train tokenizer了？tokenizer这个东西也是一个deep network吗？具体的输入输出形式是什么？输入输出的shape是什么样的？

---

## 🤖 Claude

先给个直接的答案：**tokenizer 训练只花了 69.42 秒**（日志 301 行）。它不是耗时大头。

tokenizer 训练 **69.42 秒**（00:56:17 结束），base_train 在 **00:56:32** 就开始了——launch 后 90 秒。所以我之前那张表里"数据+tokenizer ~20 min"是错的，那 20 分钟实际在 base_train 内部（周期性 CORE/bpb 评测 + compile，不计入它自报的 184.31m）。

### 不是 neural network

tokenizer 用的是 **BPE (Byte Pair Encoding)**，一个纯粹的**贪心频率统计算法**。没有参数、没有梯度、没有 GPU、不做反向传播。`nanochat/tokenizer.py:42` 调的是 `rustbpe`——一个 Rust 写的库，全程跑在 CPU 上。

"训练"在这里的意思是**从语料里统计出一张合并规则表**，不是拟合权重。算法就三步循环：

1. 把文本按正则切成小块（`SPLIT_PATTERN`，`tokenizer.py:26`），合并永远不跨块
2. 统计所有**相邻 token 对**的出现频率
3. 把最高频的那一对合并成一个新 token，记进表里；回到第 2 步

重复 32,503 次。日志里能直接看到这个过程（296-300 行）：

```
Progress: 97%  (31528/32503 merges) - Last merge: (807, 1568) -> 31783 (frequency: 992)
Progress: 100% (32503/32503 merges) - Last merge: (3896, 542) -> 32758 (frequency: 942)
```

`(3896, 542) -> 32758` 就是"token 3896 后面跟着 token 542"这个组合出现了 942 次，于是给它分配新 id 32758。**越往后频率越低**（结尾只有 942 次），这就是贪心的特征——好料先挑走。

vocab 的构成是精确对得上的：

```
256 (所有单字节)  +  32,503 (学到的合并)  +  9 (special tokens)  =  32,768 = 2^15
```

### 花了多少时间，为什么要花

**69.42 秒**。耗时来自：流式读 **20 亿字符**的语料（`--max-chars 2e9`，每篇文档截断到 10,000 字符），统计全部相邻对频率，然后跑 32,503 轮"找最大值 + 合并 + 更新计数"。数据量大，但算法本身很轻，Rust 实现一分钟出头。

相比 base_train 的 3 小时 22 分，这一步可以忽略不计。

**为什么必须先做**：模型的 embedding 表 `wte` 是 `(32768, 1536)`，lm_head 是 `(1536, 32768)`——vocab_size 是模型结构的一部分。tokenizer 不定下来，模型的形状就定不下来，训练数据也没法转成 token id。所以它必须排在 pretraining 前面。

顺带，`tok_eval` 还比较了压缩率（日志 313-330 行）。在 ClimbMix 语料上 **4.74 bytes/token**，比 GPT-2 的 4.67 好 1.4%——因为它是在真实训练分布上训的。代码上更明显：**+31.1%**。但韩语比 GPT-4 差一倍（1.19 vs 2.45 bytes/token），因为 32K 的 vocab 不够分给多语言。这是刻意的取舍。

### 输入输出形式

**训练时**（`tok_train.py`）：

| | |
|---|---|
| 输入 | `Iterator[str]`——文档字符串流，共 2e9 字符 |
| 输出 | `mergeable_ranks: dict[bytes, int]`，32,503 条 `token字节 → 优先级` 的映射，存成 `tokenizer.pkl` |

**没有任何 tensor**。产物是一张查找表。

**使用时**——`str ↔ list[int]` 双向映射。实测：

```
输入:  'The capital of France is Paris. 12345'     37 chars / 37 bytes
输出:  [483, 6968, 285, 7265, 306, 9682, 46, 32, 1179, 5260, 53]     len = 11

  483  b'The'          9682  b' Paris'
 6968  b' capital'       46  b'.'
  285  b' of'            32  b' '
 7265  b' France'      1179  b'12'
  306  b' is'          5260  b'34'
                         53  b'5'
```

两个细节值得注意：

- **空格归属前一个词**：`b' capital'` 带前导空格，这样解码时不需要额外规则拼空格。
- **`12345` 被切成 `12` / `34` / `5`**：这不是巧合。`SPLIT_PATTERN` 里写的是 `\p{N}{1,2}`——数字最多 2 位一组。代码注释（`tokenizer.py:22-24`）说明了原因：GPT-4 用的是 `{1,3}`，但在 32K 这种小 vocab 下不值得把 token 预算浪费在数字上，实测 2 是最优。**这也部分解释了为什么你的模型 GSM8K 只有 2.65%**——多位数在输入层就被切碎了，做算术天然吃亏。

### 关于 shape

**tokenizer 本身没有固定 shape**，输出是变长的 `list[int]`。tensor 形状是下游才出现的。完整链路：

| 阶段 | 数据形态 |
|---|---|
| 原始文档 | `str`，任意长度 |
| `tokenizer.encode()` | `list[int]`，**变长** |
| dataloader 拼接打包 | `x: (B, T) = (16, 2048)` int64，`y: (16, 2048)` |
| `wte` 查表 `(32768, 1536)` | `(16, 2048, 1536)` bfloat16 |
| 24 层 transformer | 保持 `(16, 2048, 1536)` |
| `lm_head` `(1536, 32768)` | `(16, 2048, 32768)` logits |

`(16, 2048)` 就是日志里那行 `Tokens / micro-batch / rank: 16 x 2048 = 32,768`——每卡 16 条序列、每条 2048 token。dataloader（`nanochat/dataloader.py`）负责把变长的文档流用 best-fit 打包塞进固定的 2048 长度窗口，这是 tokenizer 和 tensor 世界之间的转换层。

---

## 附录：本次 run 的关键事实速查

| 项目 | 值 |
|---|---|
| 硬件 | 4×H200（原配方 8×H100） |
| 唯一必要改动 | `--nproc_per_node=8` → `4`（grad_accum 由 `base_train.py:410` 自动 4 → 8） |
| 模型 | d24：24 层，n_embd 1536，12 heads，总参数 1.38B |
| total batch size | 1,048,576 tokens（自动推导，与 GPU 数无关） |
| 训练步数 | 5,568（由 `--target-param-data-ratio=8` 推导） |
| 训练 token | 5,838,471,168 |
| 训练 FLOPs | ~2.79e19 |
| 稳态 dt | ~2.0 s/step，~528k tok/sec |
| 真实 MFU | ~62%（日志显示 5.2% 是 fp8 bug，已修） |
| 总耗时 | 4 小时 11 分（00:55 → 05:06） |
| val bpb | 0.7190 |
| **CORE** | **0.2680**（GPT-2 = 0.2565 ✅） |
| ChatCORE | 0.2161 |

### 环境相关的两个坑

1. `$HOME` (`/sailhome/koven`) 只有 ~13G，所有会增长的缓存都重定向到 `/svl/u/koven`（见 `runs/env_4xh200.sh`）。
2. 系统 `LD_LIBRARY_PATH` 里的 `/usr/local/cuda-12.2/lib64` 带 `libcudnn.so.9.7.1`，会盖掉 torch wheel 自带的 9.10.2 并导致 hard error（`LD_LIBRARY_PATH` 的搜索顺序在 `DT_RUNPATH` 之前）。解法是只保留 OpenBLAS。

### 修掉的 bug

`nanochat/gpt.py:348`，`num_matmul_params()` 原本匹配 `isinstance(m, Linear)`（nanochat 自己的子类）。`--fp8` 会把 145/158 层换成 `Float8Linear`，它继承 `nn.Linear` 而非 nanochat 的 `Linear`，于是计数静默归零，`estimate_flops()` 只剩 attention 项，MFU 被低估 12 倍。改成匹配 `nn.Linear` 即可，顺带修好 `estimate_decode_flops` 和 prefill 估算。提交在分支 `fix-mfu-fp8`。
