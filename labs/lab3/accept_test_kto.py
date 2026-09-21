"""Lab 3 · Stage 3 · challenge (2): KTO acceptance test (unpaired good/bad).

Spec: Koven — implicit reward r = beta*(logpi - logpiref); z0 = batch-mean of r
(detach, clamp>=0); value loss lam*(1 - sigma(+-(r - z0))); lam manual, default 1.

Checks (policy = fresh sft50, ref frozen; AdamW lr 1e-5, 60 steps, batch 7 = odd,
lam_d=1.5 per the official lam_D*n_D = lam_U*n_U rule for the 40/60 mix):
  T1 init: loss == 0.5 exactly (r = 0, z0 = 0, sigma(0) = 0.5), z0 == 0
  T2 trained beta=0.2: eval good seq-logp rises and bad falls vs init (SPEC: good↑ bad↓)
  T3 structurally unpaired: odd batches and an all-good batch both run
  T4 ref params bit-identical after training

Run:  .venv/bin/python labs/lab3/accept_test_kto.py
"""
import random
import sys

sys.path.insert(0, "labs/mini")
import torch  # noqa: E402
from lab0_tokenizer import get_tokenizer  # noqa: E402
from lab3_dpo import load_policy_and_ref, batch_pairs, sequence_logprob, kto_loss  # noqa: E402

NUM2WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def make_unpaired(n, seed, frac_good=0.4):
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        good = i < int(n * frac_good)
        ans = a + b if good else rng.choice([x for x in range(1, 9) if x != a + b])
        prompt = {"messages": [{"role": "user", "content": f"What is {NUM2WORD[a]} plus {NUM2WORD[b]}?"}]}
        samples.append((prompt, f"The answer is {NUM2WORD[ans]}.", bool(good)))
    rng.shuffle(samples)
    return samples

def logps_of(model, tok, samples, device):
    # reuse batch_pairs by feeding each sample as (conv, response, response): take first half
    convs = [(c, resp, resp) for c, resp, _ in samples]
    inputs, targets = batch_pairs(tok, convs, device)
    with torch.no_grad():
        lp = sequence_logprob(model, inputs, targets)
    return lp[:len(samples)]

def main():
    device = "cuda"
    tok = get_tokenizer()
    train_s = make_unpaired(63, seed=0)          # 63: odd batches guaranteed
    eval_s = make_unpaired(20, seed=1)
    labels_eval = torch.tensor([g for _, _, g in eval_s], device=device)
    failures = []

    policy, ref = load_policy_and_ref(device)
    fp0 = sum(p.float().sum().item() for p in ref.parameters())
    lp_init = logps_of(policy, tok, eval_s, device)

    # T1: init loss
    r0 = torch.zeros(8, device=device)
    labels0 = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool, device=device)
    loss0, st0 = kto_loss(r0, r0.clone(), labels0, beta=0.2)
    if abs(loss0.item() - 0.5) > 1e-3 or st0["z0"] != 0.0:
        failures.append(f"T1: init loss {loss0.item():.6f} (want 0.5), z0 {st0['z0']}")

    # T3a: all-good batch must run
    kto_loss(torch.randn(5, device=device), torch.randn(5, device=device),
             torch.ones(5, dtype=torch.bool, device=device), beta=0.2)

    # T2: train, odd batch size 7
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    i = 0
    for _ in range(60):
        batch = [train_s[(i + k) % len(train_s)] for k in range(7)]
        i += 7
        convs = [(c, resp, resp) for c, resp, _ in batch]
        inputs, targets = batch_pairs(tok, convs, device)
        pi = sequence_logprob(policy, inputs, targets)[:7]
        with torch.no_grad():
            rf = sequence_logprob(ref, inputs, targets)[:7]
        labels = torch.tensor([g for _, _, g in batch], device=device)
        loss, stats = kto_loss(pi, rf, labels, beta=0.2, lam_d=1.5)  # 40/60 data: lam_D*n_D = lam_U*n_U
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    lp_end = logps_of(policy, tok, eval_s, device)
    d_good = (lp_end - lp_init)[labels_eval].mean().item()
    d_bad = (lp_end - lp_init)[~labels_eval].mean().item()
    print(f"T2: eval seq-logp delta  good {d_good:+.2f}  bad {d_bad:+.2f}  (last z0 {stats['z0']:.3f})")
    if not (d_good > 0.5 and d_bad < -0.5):
        failures.append(f"T2: want good>+0.5 and bad<-0.5, got {d_good:+.2f}/{d_bad:+.2f}")

    # T4
    if abs(sum(p.float().sum().item() for p in ref.parameters()) - fp0) > 0:
        failures.append("T4: reference changed")

    if failures:
        print("FAIL"); [print(" ", f) for f in failures]; return 1
    print("PASS: T1-T4 green")
    return 0

if __name__ == "__main__":
    sys.exit(main())
