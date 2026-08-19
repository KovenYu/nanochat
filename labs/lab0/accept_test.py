"""Lab 0 · Stage 3 acceptance test.

Challenge: specify, token by token, the exact structure that
render_conversation(CHALLENGE_CONV) produces — ids AND mask.

Your change-spec fills labs/lab0/structure_spec.py, which must define one name:

    STRUCTURE: list of (kind, value, mask)
      kind  = "special" -> value is a special token name, e.g. "<|bos|>"  (1 token)
      kind  = "text"    -> value is an exact string; it contributes len(encode(value))
                           tokens, all carrying this segment's mask value
      mask  = 0 or 1

The test builds (ids, mask) from STRUCTURE and compares against the real
render_conversation output, token by token. Any mismatch: FAIL with the first
diverging position. No partial credit.

Run:  .venv/bin/python labs/lab0/accept_test.py
"""

import sys
sys.path.insert(0, "labs/mini")
sys.path.insert(0, "labs/lab0")

from lab0_tokenizer import get_tokenizer  # noqa: E402

CHALLENGE_CONV = {"messages": [
    {"role": "system",    "content": "You are a terse calculator."},
    {"role": "user",      "content": "What is 17*23?"},
    {"role": "assistant", "content": [
        {"type": "text",          "text": "Let me compute."},
        {"type": "python",        "text": "17*23"},
        {"type": "python_output", "text": "391"},
    ]},
    {"role": "user",      "content": "Now 391+9?"},
    {"role": "assistant", "content": "400"},
]}


def main():
    try:
        from structure_spec import STRUCTURE
    except ImportError:
        print("FAIL: labs/lab0/structure_spec.py with STRUCTURE not found")
        return 1

    tok = get_tokenizer()
    spec_ids, spec_mask = [], []
    for kind, value, m in STRUCTURE:
        assert m in (0, 1), f"bad mask in spec: {m!r}"
        if kind == "special":
            spec_ids.append(tok.encode_special(value))
            spec_mask.append(m)
        elif kind == "text":
            t = tok.encode(value)
            spec_ids.extend(t)
            spec_mask.extend([m] * len(t))
        else:
            raise AssertionError(f"bad kind in spec: {kind!r}")

    real_ids, real_mask = tok.render_conversation(CHALLENGE_CONV)

    n = max(len(spec_ids), len(real_ids))
    for i in range(n):
        s = (spec_ids[i], spec_mask[i]) if i < len(spec_ids) else ("<spec ended>", "-")
        r = (real_ids[i], real_mask[i]) if i < len(real_ids) else ("<real ended>", "-")
        if s != r:
            def show(tid, m):
                if isinstance(tid, str):
                    return tid
                return f"{tok.decode([tid])!r} (id {tid}, mask {m})"
            print(f"FAIL at position {i}: spec has {show(*s)}, real is {show(*r)}")
            print(f"lengths: spec {len(spec_ids)}, real {len(real_ids)}")
            return 1

    print(f"PASS: {len(real_ids)} tokens, ids and mask all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
