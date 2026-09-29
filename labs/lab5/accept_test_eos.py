"""Lab 5 · Stage 3 · challenge (3): the stop token gets gradient.

Koven's spec (2026-09-29): keep <|assistant_end|> in the rollout row (mask 1) instead of
dropping it. Decisions: Q1(b) <|bos|> still ends the row and is NOT kept; Q2(a) rows cut at
max_new_tokens are untouched (no overlong penalty); Q3(a) flag GRPOConfig.train_eos,
default False == reference. CC caveat: the verifier sees the text without the EOS.

Checks (policy = sft50, kl_beta = 0):
  T1 structure, step-0 rollouts of examples 0..7, same seeds with the flag off and on:
     - row_on == row_off + [EOS] when the row stopped on EOS, else row_on == row_off
       (bos-terminated and truncated rows are unchanged)
     - flag on: the last counted target of every EOS row is EOS; counted tokens =
       off count + number of EOS rows; flag off: EOS never a counted target
     - rewards identical (the verifier never sees the EOS)
  T2 mechanism, one optimizer step on one example that has a stopped, positive-advantage
     row: log P(EOS) at that row's stop position rises with the flag on, and rises more
     than with the flag off (the direct term is positive).
     Criterion revised 2026-09-29 after the first run (Koven chose (a)); the original
     "> 3x |flag-off change|" failed: +0.122 (on) vs -0.065 (off). Finding: with the flag
     off EOS is still moved INDIRECTLY (softmax shares the normalizer with the content
     tokens being pushed), and that indirect effect is the same order as the direct one.
  T3 result on record (first run): both modes drive truncation to ~0 within 40 steps
     (last-10 mean off 0.023, on 0.005); on this task the indirect path already teaches
     stopping, so the flag's effect over 40 steps is small.
  T3 dynamics, 40 steps x 2 seeds per mode: mean truncated fraction over the last 10
     steps is lower with the flag on (direction only; E2 showed single runs are noisy)

Run:  source runs/env_4xh200.sh && .venv/bin/python labs/lab5/accept_test_eos.py
"""

import sys

sys.path.insert(0, "labs/mini")

import torch  # noqa: E402
from lab0_tokenizer import get_tokenizer  # noqa: E402
from lab2_sft import setup_optimizer  # noqa: E402
from lab5_grpo import (GRPOConfig, ArithmeticTask, load_policy_and_maybe_ref,  # noqa: E402
                       rollout_example, grpo_step, train)

DEV = "cuda"
tok = get_tokenizer()
EOS = tok.encode_special("<|assistant_end|>")
OFF, ON = GRPOConfig(train_eos=False), GRPOConfig(train_eos=True)
fails = []

def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)

# ---------------------------------------------------------------- T1
print("T1 structure")
policy, _ = load_policy_and_maybe_ref(OFF, DEV)
task = ArithmeticTask(64, seed=0)
n_eos_rows, candidate = 0, None
for idx in range(8):
    s_off, _, t_off, r_off, _ = rollout_example(policy, tok, task, idx, 0, OFF, DEV)
    s_on, _, t_on, r_on, a_on = rollout_example(policy, tok, task, idx, 0, ON, DEV)
    rows_ok, last_ok = [], []
    for i, (a, b) in enumerate(zip(s_off, s_on)):
        stopped = b[-1] == EOS
        rows_ok.append(b == (a + [EOS] if stopped else a))
        if stopped:
            n_eos_rows += 1
            last = int((t_on[i] >= 0).nonzero()[-1])
            last_ok.append(int(t_on[i, last]) == EOS)
            if candidate is None and a_on[i] > 0:
                candidate = (idx, b)
    check(f"ex{idx} every row: on == off + [EOS] if stopped else identical", all(rows_ok))
    check(f"ex{idx} every EOS row ends its counted targets with EOS", all(last_ok), f"({len(last_ok)} EOS rows)")
    check(f"ex{idx} counted tokens on = off + EOS rows",
          int((t_on >= 0).sum()) == int((t_off >= 0).sum()) + sum(r[-1] == EOS for r in s_on))
    check(f"ex{idx} flag off: EOS never counted", not bool(((t_off == EOS) & (t_off >= 0)).any()))
    check(f"ex{idx} rewards identical", torch.equal(r_off, r_on))
print(f"  rows that stopped on EOS across 8 examples: {n_eos_rows}/128")
assert n_eos_rows > 0, "no row stopped on EOS: T1 did not exercise the change"
del policy
torch.cuda.empty_cache()

# ---------------------------------------------------------------- T2
print("T2 mechanism")
assert candidate is not None, "no stopped row with positive advantage in examples 0..7"
idx, row = candidate
ctx = torch.tensor([row[:-1]], dtype=torch.long, device=DEV)     # (1, L-1): up to the stop position

def logp_eos(model):
    with torch.no_grad():
        return torch.log_softmax(model(ctx)[0, -1], -1)[EOS].item()

deltas = {}
for name, cfg in (("off", OFF), ("on", ON)):
    policy, _ = load_policy_and_maybe_ref(cfg, DEV)
    before = logp_eos(policy)
    opt = setup_optimizer(policy)
    for g in opt.param_groups:
        g["lr"] = g["lr"] * cfg.init_lr_frac                      # step 0: lrm = 1
    grpo_step(policy, None, tok, task, [idx], 0, cfg, DEV)         # same seeds -> same rollout
    opt.step()
    deltas[name] = logp_eos(policy) - before
    print(f"  flag {name}: log P(EOS) at the stop position {before:+.3f} -> {before + deltas[name]:+.3f} (delta {deltas[name]:+.4f})")
    del policy, opt
    torch.cuda.empty_cache()
check("flag on raises log P(EOS)", deltas["on"] > 0)
check("flag on change > flag off change (direct term positive)", deltas["on"] > deltas["off"],
      f"({deltas['on']:+.4f} vs {deltas['off']:+.4f})")

# ---------------------------------------------------------------- T3
print("T3 dynamics (40 steps x 2 seeds per mode)")
summary = {}
for name, cfg in (("off", OFF), ("on", ON)):
    tails = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        policy, _ = load_policy_and_maybe_ref(cfg, DEV)
        hist = train(policy, None, tok, ArithmeticTask(64, seed=seed), cfg, 40, DEV, log=lambda s: None)
        tail = hist[-10:]
        mean = lambda k, h: sum(x[k] for x in h) / len(h)
        tails.append({k: mean(k, tail) for k in ("truncated", "seq_len", "reward")})
        print(f"  flag {name} seed {seed}: first10 truncated {mean('truncated', hist[:10]):.2f} len {mean('seq_len', hist[:10]):.1f} "
              f"| last10 truncated {tails[-1]['truncated']:.2f} len {tails[-1]['seq_len']:.1f} reward {tails[-1]['reward']:.2f}")
        del policy
        torch.cuda.empty_cache()
    summary[name] = {k: sum(t[k] for t in tails) / 2 for k in tails[0]}
print(f"  last-10 mean over seeds: off {summary['off']} | on {summary['on']}")
check("flag on truncates less (last 10 steps, 2 seeds)", summary["on"]["truncated"] < summary["off"]["truncated"])

print("PASS" if not fails else f"FAIL: {len(fails)} checks: {fails[:5]}")
sys.exit(1 if fails else 0)
