# labs/ 教学协议（v3，single-repo 版）
适用范围：labs/ 目录下的一切工作。repo 其余部分不受此协议约束。

## 术语
- 参照实现：本 repo 的 nanochat/ 包与 scripts/，即 tag speedrun-4xh200-baseline 所指的树。
  labs 期间禁止 fetch/merge karpathy 上游，保持参照物静止。
- mini：labs/mini/ 下由你（CC）蒸馏的教学版，每个 lab 一个单文件。
- SPEC.md：蒸馏规则与 lab 定义，你的施工图。
- predictions.md：每个 lab 中唯一由 Koven 填写的文件。
- accept_test.py：Stage 3 验收脚本，可执行，判 pass/fail。

## 分工
全部代码由你编写；Koven 全程不写码。他的产出只有四种：
问题、预测、change-spec、对 diff 的解释。

## Stage 1 — READ（你蒸馏，他读）
- 按 SPEC.md 蒸馏本 lab 的单文件 mini；与参照实现语义一致，禁止顺手优化或重构。
- 每个 tensor 操作行内注释预期 shape。
- 他提问时：先讲概念，再指到参照实现的文件与行号。

## Stage 2 — TRACE（他预测，harness 验证）
- 你只维护 labs/mini/trace_harness.py（hooks 打印 shape/dtype/mean/std/max/grad-norm），
  不动模型代码。
- 流程：生成本 lab 的 predictions.md 空表 → 他填完 → 跑 harness → 你出 diff 报告
  → 只讨论他错的行。
- 他填完之前，不得向他展示任何实测值。

## Stage 3 — MODIFY（你出题并执行，他出 high-level spec）（v4 修订，2026-09-18）
- 你给：目标 + accept_test.py。
- 他给 high-level change-spec（核心 idea 层面即可，不必落到文件/函数）；
  决策留白由你以**选择题**形式给出（选项 + trade-off），他选择即拍板（v5，2026-09-20）；
  你**直接、主动列出落地该 idea 的全部 caveat**，以"他的 idea + caveats"为完整 spec 实现。
  caveat 中涉及的决策留白（如异常情形怎么处理）仍由他拍板，不替他决定，不加未议定的防御代码。
- test 失败：报现象与 caveat 层面的原因；实现层 bug 由你自己修。
- 通过后：review 最终 diff，指出 fragile 之处。

## 仪式与配置
- Lab 收尾：mini vs 参照实现对应文件的 diff 阅读，他解释每处差异的工程动机，你纠正补充。
  Lab 2 的收尾 diff 指定为 scripts/base_train.py vs scripts/chat_sft.py。
- eyeball config：单进程单卡，tiny 尺寸（见 SPEC.md），秒—分钟级短跑随时可跑。
- shadow config：source runs/env_4xh200.sh 后 torchrun --nproc_per_node=4，仅用于 milestone。
- checkpoint 纪律：只读一律使用 *-baseline-26413dc 冻结副本；
  一切训练输出写入 $NANOCHAT_BASE_DIR/labs/ 之下，禁止写任何既有 checkpoint 目录。
