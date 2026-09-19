"""Lab 3 · Stage 3 · challenge (1): DPO loss acceptance test.

GOAL: fill dpo_loss() in labs/mini/lab3_dpo.py from Koven's spec. Signature stays:

    dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta) -> (loss, stats)

    all inputs (B,) fp32 sequence logps; loss = scalar with grad;
    stats = dict of floats for monitoring (must include "margin": mean implicit-
    reward margin over the batch).

Synthetic data: same-template arithmetic pairs — chosen fluent+correct,
rejected fluent+wrong, IDENTICAL token length (content-only signal; the C2
lesson: at init the policy prices them nearly equally).

Checks (policy = sft50 d24, ref = frozen copy; AdamW lr 1e-5, 60 steps, B=8):
  T1 init: loss == ln2 (3 decimals), |margin| < 1e-4
  T2 beta=0.2 trained: eval margin > 1.0 and train loss < 0.45
  T3 beta sweep {0.05, 0.2, 0.8}: drift = mean |pi_logp - ref_logp| on eval rows
     strictly decreases as beta increases  (Koven's B2, now as a measurement)
  T4 ref params bit-identical before vs after training (frozen means frozen)
  T5 chosen logp direction is NOT required to rise (B4): report it, no assert

Run:  .venv/bin/python labs/lab3/accept_test_dpo.py
"""

import math
import random
import sys

sys.path.insert(0, "labs/mini")

import torch  # noqa: E402
from lab0_tokenizer import get_tokenizer  # noqa: E402
from lab3_dpo import load_policy_and_ref, batch_pairs, sequence_logprob, dpo_loss  # noqa: E402

NUM2WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def make_pairs(n, seed):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        wrong = rng.choice([x for x in range(1, 9) if x != a + b])
        prompt = {"messages": [{"role": "user",
                                "content": f"What is {NUM2WORD[a]} plus {NUM2WORD[b]}?"}]}
        pairs.append((prompt,
                      f"The answer is {NUM2WORD[a + b]}.",     # fluent + correct
                      f"The answer is {NUM2WORD[wrong]}."))    # fluent + wrong, same length
    return pairs

def eval_stats(policy, ref, tok, pairs, beta, device):
    with torch.no_grad():
        inputs, targets = batch_pairs(tok, pairs, device)
        B = len(pairs)
        pi = sequence_logprob(policy, inputs, targets)
        rf = sequence_logprob(ref, inputs, targets)
        margin = (beta * ((pi[:B] - rf[:B]) - (pi[B:] - rf[B:]))).mean().item()
        drift = (pi - rf).abs().mean().item()
        return margin, drift, pi[:B].mean().item()

def train(policy, ref, tok, train_pairs, beta, device, steps=60, bsz=8, lr=1e-5):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    losses = []
    i = 0
    for _ in range(steps):
        batch = [train_pairs[(i + k) % len(train_pairs)] for k in range(bsz)]
        i += bsz
        inputs, targets = batch_pairs(tok, batch, device)
        B = bsz
        pi = sequence_logprob(policy, inputs, targets)
        with torch.no_grad():
            rf = sequence_logprob(ref, inputs, targets)
        loss, stats = dpo_loss(pi[:B], pi[B:], rf[:B], rf[B:], beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def main():
    device = "cuda"
    tok = get_tokenizer()
    train_pairs = make_pairs(64, seed=0)
    eval_pairs = make_pairs(16, seed=1)
    failures = []

    policy, ref = load_policy_and_ref(device)
    ref_fingerprint = sum(p.float().sum().item() for p in ref.parameters())

    # T1: init
    inputs, targets = batch_pairs(tok, eval_pairs, device)
    B = len(eval_pairs)
    with torch.no_grad():
        pi = sequence_logprob(policy, inputs, targets)
        rf = sequence_logprob(ref, inputs, targets)
    loss0, stats0 = dpo_loss(pi[:B], pi[B:], rf[:B], rf[B:], beta=0.2)
    if abs(loss0.item() - math.log(2)) > 1e-3 or abs(stats0["margin"]) > 1e-4:
        failures.append(f"T1: init loss {loss0.item():.6f} (want ln2), margin {stats0['margin']:.2e}")

    # T2 + T3: sweep (fresh policy per beta, shared frozen ref)
    results = {}
    for beta in (0.05, 0.2, 0.8):
        del policy
        torch.cuda.empty_cache()
        policy, _ = load_policy_and_ref(device)   # fresh start; keep the original ref
        losses = train(policy, ref, tok, train_pairs, beta, device)
        margin, drift, chosen_lp = eval_stats(policy, ref, tok, eval_pairs, beta, device)
        results[beta] = (margin, drift, chosen_lp, losses[-1])
        print(f"beta={beta:4}: final train loss {losses[-1]:.4f} | eval margin {margin:+.3f} | "
              f"drift {drift:.3f} | chosen seq logp {chosen_lp:+.2f}")
    if not (results[0.2][0] > 1.0 and results[0.2][3] < 0.45):
        failures.append(f"T2: beta=0.2 margin {results[0.2][0]:.3f} (want >1.0), "
                        f"final loss {results[0.2][3]:.3f} (want <0.45)")
    drifts = [results[b][1] for b in (0.05, 0.2, 0.8)]
    if not (drifts[0] > drifts[1] > drifts[2]):
        failures.append(f"T3: drift not decreasing in beta: {drifts}")

    # T4: frozen means frozen
    if abs(sum(p.float().sum().item() for p in ref.parameters()) - ref_fingerprint) > 0:
        failures.append("T4: reference parameters changed during training")

    # T5: report only
    print(f"T5 (report): chosen seq logp by beta: " +
          ", ".join(f"{b}:{results[b][2]:+.2f}" for b in results))

    if failures:
        print("FAIL")
        for f in failures:
            print(" ", f)
        return 1
    print("PASS: T1-T4 green (T5 informational)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
