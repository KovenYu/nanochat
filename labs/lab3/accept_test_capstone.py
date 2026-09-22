"""Capstone acceptance: structure + labels + end-to-end plumbing on sft50.

  T1 DPO: exactly 2 pairs; the shared pair's rendered prompt prefixes are
     token-identical up to the fork; the cross-outer pair's are not (flagged)
  T2 KTO labels: verifier {v1:G, v2:B, v2b:G}; user {v1:None, v2:None, v2b:G};
     the user channel carries ONLY the terminal verdict (decision A(b)), so the
     verifier channel has 3 samples per view and the user channel exactly 1
  T3 views: every attempt appears in both A and B contexts; B contexts contain
     the feedback string, A contexts do not
  T4 e2e: dpo_loss on the pairs = ln2 at init; kto_loss (user channel, view B)
     = 0.5 at init; both finite with grad
"""
import math, sys
sys.path.insert(0, "labs/mini"); sys.path.insert(0, "labs/lab3"); sys.path.insert(0, "labs/lab2")
import torch
from lab0_tokenizer import get_tokenizer
from lab3_dpo import load_policy_and_ref, sequence_logprob, dpo_loss, kto_loss
from coda_views import TRAJ
from coda_prefs import dpo_pairs, kto_samples, kto_batch, dpo_batch

def main():
    tok = get_tokenizer()
    fails = []
    def ck(name, cond, detail=""):
        if not cond: fails.append(f"{name}: {detail}")

    pairs = dpo_pairs(TRAJ)
    ck("T1 count", len(pairs) == 2, f"{len(pairs)}")
    kinds = {p[4] for p in pairs}
    ck("T1 kinds", kinds == {"shared", "cross_outer"}, str(kinds))
    from lab3_dpo import render_pair
    (cc, ct, cr, rt, _) = [p for p in pairs if p[4] == "shared"][0]
    ids_c, _ = render_pair(tok, cc, ct); ids_r, _ = render_pair(tok, cr, rt)
    astart = tok.encode_special("<|assistant_start|>")
    fork = max(i for i, t in enumerate(ids_c) if t == astart)
    ck("T1 shared prefix", ids_c[:fork + 1] == ids_r[:max(i for i, t in enumerate(ids_r) if t == astart) + 1],
       "fork prefixes differ")
    (cc, ct, cr, rt, _) = [p for p in pairs if p[4] == "cross_outer"][0]
    ck("T1 cross differs", cc["messages"] != cr["messages"], "")

    samples = kto_samples(TRAJ)
    def lab(code, ch, view="B"):
        return [s[ch] for s in samples if s["response"] == code and s["view"] == view][0]
    v1, v2, v2b = (TRAJ["outer"][0]["attempts"][0]["code"],
                   TRAJ["outer"][1]["attempts"][0]["code"],
                   TRAJ["outer"][1]["attempts"][1]["code"])
    ck("T2 verifier", (lab(v1, "verifier"), lab(v2, "verifier"), lab(v2b, "verifier")) == (True, False, True))
    ck("T2 user", (lab(v1, "user"), lab(v2, "user"), lab(v2b, "user")) == (None, None, True))
    n_user = sum(s["user"] is not None for s in samples if s["view"] == "B")
    ck("T2 user is terminal-only", n_user == 1, f"{n_user} user-labelled samples in view B")

    fb = TRAJ["outer"][0]["user_feedback"]
    for s in samples:
        joined = " ".join(m["content"] if isinstance(m["content"], str) else "" for m in s["conv"]["messages"])
        if s["view"] == "B" and s["response"] in (v2, v2b):
            ck("T3 B has feedback", fb in joined, s["response"])
        if s["view"] == "A":
            ck("T3 A clean", fb not in joined, s["response"])

    policy, ref = load_policy_and_ref("cuda")
    inputs, targets = dpo_batch(pairs, tok, "cuda")
    B = len(pairs)
    pi = sequence_logprob(policy, inputs, targets)
    with torch.no_grad():
        rf = sequence_logprob(ref, inputs, targets)
    l_dpo, _ = dpo_loss(pi[:B], pi[B:], rf[:B], rf[B:], beta=0.2)
    ck("T4 dpo init", abs(l_dpo.item() - math.log(2)) < 1e-4, f"{l_dpo.item():.6f}")
    ki, ktg, klab = kto_batch(samples, tok, "user", "B", "cuda")
    kpi = sequence_logprob(policy, ki, ktg)
    with torch.no_grad():
        krf = sequence_logprob(ref, ki, ktg)
    l_kto, _ = kto_loss(kpi, krf, klab, beta=0.2)
    ck("T4 kto init", abs(l_kto.item() - 0.5) < 1e-4, f"{l_kto.item():.6f}")
    ck("T4 grads", l_dpo.requires_grad and l_kto.requires_grad)

    if fails:
        print("FAIL"); [print(" ", f) for f in fails]; return 1
    print(f"PASS: 2 DPO pairs (1 shared-fork, 1 cross-outer), {len(samples)} KTO samples "
          f"(2 views x 2 channels), e2e init: dpo {l_dpo.item():.4f}=ln2, kto {l_kto.item():.4f}=0.5")
    return 0

if __name__ == "__main__":
    sys.exit(main())
