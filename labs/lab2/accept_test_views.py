"""Challenge (2) acceptance: View A/B exports carry exactly the intended mask semantics,
verified through the REAL dataloader (View B via mask_history), plus a numeric demo of
the merged-vs-sliding equivalence claim (decision 4).

Run:  .venv/bin/python labs/lab2/accept_test_views.py
"""

import itertools
import sys

sys.path.insert(0, "labs/mini")
sys.path.insert(0, "labs/lab2")

import torch                                     # noqa: E402
from lab0_tokenizer import get_tokenizer         # noqa: E402
from lab1_gpt import GPT, GPTConfig              # noqa: E402
from lab2_sft import sft_data_generator          # noqa: E402
from coda_views import TRAJ, export_view_a, export_view_b  # noqa: E402

V1, V2, V2B = (TRAJ["outer"][0]["attempts"][0]["code"],
               TRAJ["outer"][1]["attempts"][0]["code"],
               TRAJ["outer"][1]["attempts"][1]["code"])
FAIL_VERDICT = TRAJ["outer"][1]["attempts"][0]["verdict"]
FEEDBACK = TRAJ["outer"][0]["user_feedback"]

failures = []
def check(name, cond, detail=""):
    if not cond:
        failures.append(f"{name}: {detail}")

def rendered_and_supervised(conv, mask_history):
    tok = get_tokenizer()
    ids, _ = tok.render_conversation(conv)
    gen = sft_data_generator(itertools.cycle([conv]), tok, B=1, T=len(ids) - 1,
                             device="cpu", buffer_size=4, mask_history=mask_history)
    inputs, targets = next(gen)
    sup_ids = [int(t) for t in targets[0] if t != -1]
    return tok, ids, tok.decode(ids), tok.decode(sup_ids) if sup_ids else ""

def main():
    # ---- View A ----
    va = export_view_a(TRAJ)
    check("VA structure", len(va["messages"]) == 2 and va["messages"][1]["content"] == V2B,
          f"messages={len(va['messages'])}")
    tok, ids, full, sup = rendered_and_supervised(va, mask_history=False)
    for leak in (V1, V2, FAIL_VERDICT, FEEDBACK):
        check("VA no-leak", leak not in full, f"leaked: {leak!r}")
    check("VA supervised", V2B in sup and sup.strip().startswith(V2B),
          f"supervised={sup!r}")

    # ---- View B ----
    vb = export_view_b(TRAJ)
    tok, ids, full, sup = rendered_and_supervised(vb, mask_history=True)
    for ctx in (V1, V2, V2B, FAIL_VERDICT, FEEDBACK):
        check("VB context complete", ctx in full, f"missing from render: {ctx!r}")
    check("VB supervises v2", V2 in sup, f"supervised={sup!r}")
    check("VB supervises v2b", V2B in sup, "")
    check("VB last-turn text supervised", TRAJ["outer"][1]["present"] in sup, "")
    check("VB verdicts NOT supervised", FAIL_VERDICT not in sup and "PASS" not in sup, f"supervised={sup!r}")
    check("VB v1 NOT supervised", V1 not in sup, "")
    check("VB feedback NOT supervised", FEEDBACK not in sup, "")

    # ---- decision-4 equivalence demo: logits at v2 positions identical whether or not
    # the sequence continues past v2 (causality => merged sample == sliding sample) ----
    torch.manual_seed(0)
    model = GPT(GPTConfig(sequence_len=len(ids) + 8, vocab_size=32768,
                          n_layer=2, n_head=4, n_kv_head=4, n_embd=64))
    model.init_weights()
    # cut right after v2's tokens end: find v2 span end in ids
    v2_tokens = tok.encode(V2)
    end = next(i + len(v2_tokens) for i in range(len(ids))
               if ids[i:i + len(v2_tokens)] == v2_tokens)
    with torch.no_grad():
        lg_full = model(torch.tensor([ids], dtype=torch.long))[0, :end]
        lg_cut = model(torch.tensor([ids[:end]], dtype=torch.long))[0]
    diff = (lg_full - lg_cut).abs().max().item()
    check("VB merged==sliding (causality)", diff < 1e-2, f"max logit diff {diff}")

    if failures:
        print("FAIL")
        for f in failures:
            print(" ", f)
        return 1
    print(f"PASS: View A/B semantics verified; merged-vs-sliding max logit diff {diff:.2e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
