"""Lab 2 · Stage 3 · challenge (1): mask_history acceptance test.

GOAL: add a mask_history mode to the Lab 2 mini SFT dataloader — supervise ONLY each
conversation's LAST assistant turn (View B semantics). History turns (earlier assistant
turns included) become context, not training signal.

Required observable interface (how you get there is your change-spec):

    sft_data_generator(conversations, tokenizer, B, T, device,
                       buffer_size=..., mask_history=False)

  - mask_history=False (default): behavior byte-identical to today's.
  - mask_history=True: for every conversation in a packed row, the supervised
    (non -1) targets are exactly those that were supervised before AND belong to
    that conversation's last assistant turn (i.e. sit after its final
    <|assistant_start|>). Everything else is -1. python_output inside the last
    turn stays unsupervised (mask 0 is never resurrected).

Checks:
  T1 3-turn conversation, alone in a row: only turn-3 assistant tokens supervised
  T2 last turn contains [text, python, python_output, text]: python_output stays -1
  T3 single-turn conversation: mask_history output == default output
  T4 two conversations packed in one row: EACH supervises its own last turn
  T5 regression: mask_history=False matches the pre-change behavior on T1's stream

Run:  .venv/bin/python labs/lab2/accept_test.py
"""

import itertools
import sys

sys.path.insert(0, "labs/mini")
from lab0_tokenizer import get_tokenizer  # noqa: E402

CONV_3TURN = {"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "6"},
    {"role": "user", "content": "Sum both answers."},
    {"role": "assistant", "content": "4+6=10"},
]}
CONV_TOOL = {"messages": [
    {"role": "user", "content": "Compute 17*23."},
    {"role": "assistant", "content": "289? No wait."},
    {"role": "user", "content": "Use the tool."},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Sure."},
        {"type": "python", "text": "17*23"},
        {"type": "python_output", "text": "391"},
        {"type": "text", "text": "It is 391."},
    ]},
]}
CONV_1TURN = {"messages": [
    {"role": "user", "content": "Hello there."},
    {"role": "assistant", "content": "Hi."},
]}


def expected_maskhist_positions(tok, conv, offset=0):
    """Ground truth: supervised target positions for one conversation placed at `offset`
    in a row. Supervised <=> original mask==1 AND position after the LAST assistant_start."""
    ids, mask = tok.render_conversation(conv)
    astart = tok.encode_special("<|assistant_start|>")
    last_as = max(i for i, t in enumerate(ids) if t == astart)
    # target position p supervises ids[p+1] (row-local); keep mask==1 tokens after last_as
    return {offset + p for p in range(len(ids) - 1)
            if mask[p + 1] == 1 and (p + 1) > last_as}, len(ids)


def default_positions(tok, conv, offset=0):
    ids, mask = tok.render_conversation(conv)
    return {offset + p for p in range(len(ids) - 1) if mask[p + 1] == 1}, len(ids)


def get_row(gen):
    inputs, targets = next(gen)
    return inputs[0], targets[0]


def supervised_set(targets, limit):
    return {p for p in range(limit) if targets[p].item() != -1}


def main():
    from lab2_sft import sft_data_generator
    tok = get_tokenizer()
    failures = []

    def run_case(name, convs, T, expect_fn_convs, mask_history):
        stream = itertools.cycle(convs)
        try:
            gen = sft_data_generator(stream, tok, B=1, T=T, device="cpu",
                                     buffer_size=4, mask_history=mask_history)
        except TypeError as e:
            failures.append(f"{name}: generator does not accept mask_history ({e})")
            return None
        _, targets = get_row(gen)
        expected, off = set(), 0
        for conv, fn in expect_fn_convs:
            pos, length = fn(tok, conv, offset=off)
            expected |= pos
            off += length
        got = supervised_set(targets, T)
        if got != expected:
            extra, missing = sorted(got - expected), sorted(expected - got)
            failures.append(f"{name}: supervised positions differ. extra={extra} missing={missing}")
        return targets

    ids3, _ = tok.render_conversation(CONV_3TURN)
    ids_tool, _ = tok.render_conversation(CONV_TOOL)
    ids1, _ = tok.render_conversation(CONV_1TURN)

    # T1: 3-turn conv alone fills the row exactly
    run_case("T1", [CONV_3TURN], len(ids3) - 1,
             [(CONV_3TURN, expected_maskhist_positions)], mask_history=True)
    # T2: tool-call conv; python_output must stay -1
    run_case("T2", [CONV_TOOL], len(ids_tool) - 1,
             [(CONV_TOOL, expected_maskhist_positions)], mask_history=True)
    # T3: single-turn: mask_history == default
    run_case("T3", [CONV_1TURN], len(ids1) - 1,
             [(CONV_1TURN, default_positions)], mask_history=True)
    # T4: two convs packed in one row (larger first by bestfit), each keeps its own last turn
    big, small = (CONV_3TURN, CONV_1TURN) if len(ids3) >= len(ids1) else (CONV_1TURN, CONV_3TURN)
    run_case("T4", [big, small], len(ids3) + len(ids1) - 1,
             [(big, expected_maskhist_positions), (small, expected_maskhist_positions)],
             mask_history=True)
    # T5: regression — default path unchanged
    run_case("T5", [CONV_3TURN], len(ids3) - 1,
             [(CONV_3TURN, default_positions)], mask_history=False)

    if failures:
        print("FAIL")
        for f in failures:
            print(" ", f)
        return 1
    print("PASS: T1-T5 all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
