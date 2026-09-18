# labs 期间在参照实现里发现的问题（labs 结束后处理；期间参照保持静止）

1. **chat_sft.py `--num-iterations` 单位错误**（2026-09-18 发现）
   帮助文本称 "number of optimization steps"，实现里计数器在 dataloader 的 yield 循环
   （scripts/chat_sft.py:264-266），实际单位是 micro-batch。grad_accum=G 时实训步数少 G 倍。
   epoch 模式（-1，consumed 驱动）不受影响——speedrun 因此没踩坑。
   复现：--num-iterations 233 + grad_accum 8 → 29 个 optimizer step 后退出。

2. **gpt.py:307 注释漂移**：`short_window` 注释写 "2048 -> 768"，代码算出 512。
   （lab1 diff 仪式已讨论：注释处于零验证盲区。）

3. **（已修，历史记录）MFU 低报 12×**：num_matmul_params 按自定义 Linear 子类匹配，
   --fp8 换成 Float8Linear 后计为 0。修复见 commit 1a74d59。
