"""Trace harness for the labs. Per labs/CLAUDE.md this is the only file CC maintains for
Stage 2; it prints measured values and never touches model/mini code.

Usage: python labs/mini/trace_harness.py lab0
Prints the actual value for every row of labs/lab0/predictions.md, numbered identically.
"""

import sys
sys.path.insert(0, "labs/mini")


def lab0():
    from lab0_tokenizer import SPECIAL_TOKENS, get_token_bytes, get_tokenizer
    tok = get_tokenizer()
    tb = get_token_bytes()
    name = {tok.encode_special(s): s for s in SPECIAL_TOKENS}  # id -> "<|...|>"

    print("== A. id 空间 ==")
    print("A1", tok.encode_special("<|bos|>"))
    print("A2", tok.encode_special("<|assistant_start|>"))
    a3 = tok.encode_special("<|output_end|>")
    print("A3", a3)
    print("A4", int(tb[a3]))

    print("== B. token 数 ==")
    B = ["hello world", " hello world", "Hello World!", "1234567890",
         "3.14159265358979", "今天天气很好", "👍", "", "supercalifragilisticexpialidocious"]
    for i, s in enumerate(B, 1):
        ids = tok.encode(s)
        print(f"B{i}", len(ids), [tok.decode([t]) for t in ids])

    print("== C. render_conversation ==")
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
    ids, mask = tok.render_conversation(CONV)
    specials = [(i, name[t], m) for i, (t, m) in enumerate(zip(ids, mask)) if t in name]
    print("C1", "system 被 merge，循环里 4 条 message")
    print("C2", len(specials))
    m1 = [(s, m) for _, s, m in specials if m == 1]
    print("C3", len(m1), [s for s, _ in m1])
    print("C4", name.get(ids[0], tok.decode([ids[0]])), name.get(ids[1], tok.decode([ids[1]])))
    first1 = mask.index(1)
    print("C5", repr(tok.decode([ids[first1]])), f"(位置 {first1})")
    print("C6", name.get(ids[-1], tok.decode([ids[-1]])), "mask", mask[-1])
    dec = tok.decode(ids)
    a, b = dec.index("Be brief.") + len("Be brief."), dec.index("What is 2+2?")
    print("C7", repr(dec[a:b]))
    print("完整可视化：")
    print(tok.visualize_tokenization(ids, mask))

    print("== D. render_for_completion ==")
    pids = tok.render_for_completion(CONV)
    print("D1", name.get(pids[-2], tok.decode([pids[-2]])), name.get(pids[-1], tok.decode([pids[-1]])))
    print("D2", f"len(pids)={len(pids)}  len(ids)={len(ids)}")


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] == "lab0", "usage: trace_harness.py lab0"
    lab0()
