# labs/SPEC.md — 蒸馏规则与 lab 定义

## 目标
通过 4 个活跃 lab 建立 post-training（SFT + offline preference optimization）的 mental model。
Exit criterion：随机指参照实现中一行训练相关代码，Koven 能说出该行处理的 tensor 的
shape/dtype 与该行存在的理由。

## 出题原则（scale-relevance filter）
Koven 的目标：build mental model + 数值量级感知，服务于后续 post-train 工业级大模型。
- Stage 2 预测与 Stage 3 挑战**不考** speedrun-specific 组件（smear、value embeddings、
  backout、resid/x0 lambdas、relu² 等）的细节；考题聚焦 scale-invariant 的机制与量级。
- 讲解 ad-hoc 组件后必须提醒：这是 ad-hoc 的，scale up 后被抛弃。
- 判别基准：Qwen3/Gemma 级工业模型仍在用 = 存活；只存在于 modded-nanogpt/nanochat 谱系 = ad-hoc。
- mini 的蒸馏**不受此过滤影响**（checkpoint 对齐要求全部组件在场）；过滤只作用于考什么。

## 蒸馏规则
- 从参照实现（nanochat/ 与 scripts/，tag speedrun-4xh200-baseline）蒸馏，语义一致，禁止顺手优化。
- 砍掉：tokenizer 训练（直接加载下方现成 tokenizer）、inference engine 与 web UI、
  checkpoint 健壮性工程、eval bundle（只留 val loss 或该 lab 指定的 metric）。
- 每个 lab 单文件；每个 tensor 操作行内注释预期 shape。
- "不做 pretraining" 指不跑大规模预训练；eyeball 级（单卡、秒—分钟、tiny 配置）的机制验证短跑不受限。

## 资源与路径
- `NANOCHAT_BASE_DIR=/svl/u/koven/nanochat_data`（由 runs/env_4xh200.sh 导出）。
- eyeball config：单进程单卡；tiny 合成配置 B=2, T=16, n_embd=64, L=2, vocab=256，
  或真数据的小切片。
- shadow config：4×H200（`source runs/env_4xh200.sh && torchrun --nproc_per_node=4`），
  单 run ≤ 20h（实际均为分钟级）。
- 现成 tokenizer（只读）：`/svl/u/koven/nanochat_data/tokenizer/`
- 只读 checkpoint（冻结副本，与 tag speedrun-4xh200-baseline 同源）：
  - base：`/svl/u/koven/nanochat_data/base_checkpoints/d24-baseline-26413dc/`
    （model_005568.pt, meta_005568.json, optim_005568_rank{0..3}.pt）
  - SFT：`/svl/u/koven/nanochat_data/chatsft_checkpoints/d24-baseline-26413dc/`
    （model_000467.pt, meta_000467.json, optim_000467_rank{0..3}.pt）
  - 未冻结的 `base_checkpoints/d24/`、`chatsft_checkpoints/d24/` 视为不存在，不读不写。
- 全部输出：`/svl/u/koven/nanochat_data/labs/<lab名>_<日期>/`。

## 外部概念定义（使 labs 自含全部所需背景）

**Phase 1（labs 之后的下一阶段）**：用成熟框架（LLaMA-Factory）对 Qwen3-8B 级模型做 SFT。
Lab 1 选做项载入 Qwen3-0.6B（HF: Qwen/Qwen3-0.6B）的意义：用 mini 提前验证对 Qwen3
架构系的理解——它是 Phase 1 要训的那族模型的最小成员，故称"桥"。

**Coda trajectory（真实业务数据的形态；labs 中一律用合成模拟，不接真数据）**：
text2cad agent 的交互日志，双层 loop——
- outer loop：user 给 design intent + constraints → agent 生成 CAD code → user 反馈 → agent 修改；
- inner loop：每次生成的内部，agent 对一组 verifiers（compile 是否通过、静态 collision
  检查、VLM vision verdict）迭代修正，全部通过才把结果呈给 user。
trajectory 记录两层 loop 的全部尝试与反馈，含失败 attempt。

**View A / View B（同一批 trajectory 的两种 SFT 导出形态）**：
- View A（shortcut 蒸馏）：(intent + constraints) → 最终通过全部 verifier 且被 user 接受
  的 code。单轮样本，两层 loop 全部 collapse。训练目标：first-pass 质量。
- View B（revision）：保留多轮结构，context = 此前的失败尝试 + verifier/user 反馈，
  loss 只落在成功修正的那个 turn（即 mask_history 语义）。训练目标：incorporate 反馈的能力。
Lab 2 capstone 的 mask 验证 harness 即针对这两种形态的合成样本构建。

## Lab 定义

| Lab | 主题 | Stage 2 预测重点 | Stage 3 挑战（你出题时据此设计 accept_test） |
|---|---|---|---|
| 0 | tokenizer 消费侧（~2h） | 给定文本的 token 数量级；special tokens 在 chat 渲染中的位置与 id | 给定一条多轮对话 dict，预测完整 token 序列结构并逐 token 验证 |
| 1 | 模型 forward + checkpoint 解剖 | 各 module 进出 shape/dtype；RoPE 不改 q/k 范数；attention logits 量级；checkpoint 文件里有哪些 tensor、各多大 | ① MHA→GQA(n_kv=2)，tiny 短跑曲线重合 + KV 显存按预测缩小；② 把 d24-baseline 冻结 checkpoint 载入 mini，与参照实现 logits 对齐到阈值；③ 手写 KV-cache greedy decode，与无 cache 逐 token 一致。选做：载入 Qwen3-0.6B 并对齐 logits（通往 Phase 1 的桥） |
| 2 | SFT = 训练 loop + masking（核心 lab） | 渲染多轮对话后哪些位置 label=-100；packing 后 position/mask 的变化；param/grad/optimizer state 的 dtype 与字节数；tiny 模型显存预算 vs memory_allocated 误差 ≤10%；grad accumulation 的等价性 | ① 实现 mask_history 语义（只训最后一轮），accept_test 证明非零 loss 恰好落在最后一轮 assistant tokens；② View A/B 数据（Coda trajectory 的两种导出形态）的 mask 验证 harness。milestone 仪式：用 mini 对 d24-baseline base checkpoint 跑一遍 SFT（shadow config，分钟级），chat_cli 训前训后各聊一次，肉眼见证 base→chat 的转变 |
| 3 | DPO / KTO（offline RL 本体） | frozen reference 双前向的计算图；per-token logp 的 gather 维度；β 增减对 implicit reward margin 的影响方向 | ① 由你设计合成 preference 数据，Koven spec 出 DPO loss 的实现要点，β 扫描曲线符合预测；② DPO→KTO 改造（去配对，逐样本 good/bad 标签），合成数据上 good↑ bad↓；capstone：从一条模拟 Coda schema 的合成 trajectory 构造 chosen/rejected 对与 KTO 标签。policy 初始化用 lab 2 产出的 SFT checkpoint（这是 post-training pipeline 自身的拓扑 base → SFT → preference：DPO/KTO 的 policy 初始化与 frozen reference 常规上就是 SFT 模型，亲身走一遍拓扑本身是课程内容，故 Lab 2 是硬前置。fallback：若 Lab 2 的 milestone 产物因故不可用，退用冻结的 speedrun chatsft 副本 `chatsft_checkpoints/d24-baseline-26413dc/`（同为 d24 SFT 模型）作初始化，默认仍走硬顺序） |

## Parked（不删除，等触发）
- Lab 5 GRPO：解禁条件 = 决定进入 online RLVR。
- Lab 6 ZeRO/并行：解禁条件 = 大模型（235B 级）run 前一周。
