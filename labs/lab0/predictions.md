# Lab 0 · predictions.md

规则：只填「你的预测」列。填完之前不跑 harness、不问实测值。
预测允许写区间（如 "3~6"），但能推出精确值的行尽量给精确值。
对照文件：labs/mini/lab0_tokenizer.py。已知 vocab_size = 32768。

## A. id 空间

| # | 问题 | 你的预测 |
|---|---|---|
| A1 | `<\|bos\|>` 的 token id | 32768-9|
| A2 | `<\|assistant_start\|>` 的 token id | 32768-8~32768 |
| A3 | `<\|output_end\|>` 的 token id |32768-8~32768 |
| A4 | `token_bytes[A3 的答案]` 的值 | 0|

## B. token 数（`len(tok.encode(s))`，精确值或区间）

| # | 字符串 s | 你的预测 |
|---|---|---|
| B1 | `"hello world"` | 2~11|
| B2 | `" hello world"`（注意开头空格） | 2~12|
| B3 | `"Hello World!"` | 3~12|
| B4 | `"1234567890"` | 5~10|
| B5 | `"3.14159265358979"` | 8~16|
| B6 | `"今天天气很好"` | 1~18|
| B7 | `"👍"` | 1~4|
| B8 | `""`（空串） | 1~2|
| B9 | `"supercalifragilisticexpialidocious"` |? |

## C. render_conversation 的结构

对以下对话（记 `ids, mask = tok.render_conversation(CONV)`，harness 里逐字一致）：

```python
CONV = {"messages": [
    {"role": "system",    "content": "Be brief."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user",      "content": "Now use python to check."},
    {"role": "assistant", "content": [
        {"type": "text",          "text": "Sure."},
        {"type": "python",        "text": "print(2+2)"},
        {"type": "python_output", "text": "4"},
        {"type": "text",          "text": "It is 4."},
    ]},
]}
```

| # | 问题 | 你的预测 |
|---|---|---|
| C1 | 渲染时实际参与循环的 message 数 |4 |
| C2 | `ids` 里 special token 共出现几次 | 15|
| C3 | 这些 special token 中 mask=1 的有几个、各是什么 | 3, python start, python end, assistant end|
| C4 | `ids[0]` 和 `ids[1]` 是哪两个 token |bos, user start |
| C5 | 第一个 mask=1 的位置上的 token 对应什么文本 | Sure|
| C6 | `ids[-1]` 是哪个 token，其 mask 是几 | assistant end, 1|
| C7 | `decode(ids)` 里 "Be brief." 和 "What is 2+2?" 之间隔着什么 |/n/n |

## D. render_for_completion

对同一个 CONV（记 `pids = tok.render_for_completion(CONV)`）：

| # | 问题 | 你的预测 |
|---|---|---|
| D1 | `pids` 的最后两个 token 是哪两个 | ., assistant end|
| D2 | `len(pids)` 与 `len(ids)` 谁大（不必给差值） | len(ids)|
