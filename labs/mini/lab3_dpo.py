"""
Lab 3 mini: DPO scaffolding — everything EXCEPT the loss formula.

There is no DPO in the reference implementation; this file composes the lab0/1/2 minis
into the standard offline-preference setup. The loss body is deliberately a hole:
challenge (1) fills it from Koven's spec.

The DPO idea (concept only; the code is Stage 3's):
  Each (prompt, chosen, rejected) pair defines an implicit reward
      r(x, y) = beta * [ log pi(y|x) - log pi_ref(y|x) ]
  and DPO trains the policy so the chosen response out-rewards the rejected one:
      loss = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) )
  pi_ref is a FROZEN copy of the starting policy: the anchor that keeps pi from
  drifting arbitrarily far (beta scales how expensive drift is).

What lives here:
  - render_pair: (prompt conv, response text) -> ids + mask, reusing lab0's
    render_conversation so "which tokens count" is exactly the SFT supervision rule
  - batch_pairs: pad chosen+rejected into ONE (2B, T) batch -> ONE forward per model
  - sequence_logprob: the per-token gather everyone gets wrong once (Stage 2 material)
  - frozen reference handling, policy loading from the lab-2 SFT product
  - a training-step skeleton that calls dpo_loss(...) -- NotImplementedError for now
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from lab0_tokenizer import get_tokenizer                     # noqa: E402
from lab1_gpt import GPT, GPTConfig                          # noqa: E402

SFT50_DIR = "/svl/u/koven/nanochat_data/labs/sft50_env/chatsft_checkpoints/d24"
SFT50_STEP = 233

def load_policy_and_ref(device):
    """Lab-2 product (50%-budget SFT d24) -> policy (trainable) + frozen reference."""
    with open(os.path.join(SFT50_DIR, f"meta_{SFT50_STEP:06d}.json")) as f:
        meta = json.load(f)
    sd = torch.load(os.path.join(SFT50_DIR, f"model_{SFT50_STEP:06d}.pt"),
                    map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    def build(state):
        m = GPT(GPTConfig(**meta["model_config"]))
        m.load_state_dict(state, strict=True, assign=True)   # assign ADOPTS the tensors
        return m.to(device)
    policy = build(sd)
    ref = build({k: v.clone() for k, v in sd.items()})  # ref gets its OWN storage: assign
                                                        # with a shared sd would alias the
                                                        # two models (optimizer steps would
                                                        # silently update the "frozen" ref)
    for pp, rp in zip(policy.parameters(), ref.parameters()):
        assert pp.data_ptr() != rp.data_ptr(), "policy/ref alias the same storage"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)                     # frozen anchor; forward under no_grad
    return policy, ref

# -----------------------------------------------------------------------------
# (prompt, response) -> tokens + "which positions count" mask

def render_pair(tok, prompt_conv, response_text):
    """prompt_conv: {"messages":[..., ends with a user turn]}. Returns (ids, mask):
    ids includes the full chat rendering with the response as the final assistant turn;
    mask marks the response tokens (content + <|assistant_end|>) — the SAME rule SFT
    supervises, so DPO's logp sums over exactly what SFT would have trained."""
    conv = {"messages": prompt_conv["messages"] + [{"role": "assistant", "content": response_text}]}
    return tok.render_conversation(conv)            # (list[int], list[int]) same length

def pad_rendered(tok, rendered, device):
    """rendered: list of (ids, mask) from render_pair, length N.
    Returns inputs (N, T) int64, targets (N, T) int64 with -1 outside counted tokens.
    The ONE padding rule for every preference batch (DPO pairs, KTO samples).

    Pad to this batch's longest rendering. Unlike lab2's SFT, T is NOT a fixed
    constant here: T = L-1 varies per batch (fine without torch.compile; rotary
    cache covers 10x sequence_len). No packing: DPO/KTO measure per-sequence logps
    and nanochat-style packing has no block-diagonal mask, so row-mates would
    contaminate the measured margins (training tolerates that noise; measurement
    doesn't). The principled packed form is flash-attn varlen (cu_seqlens).
    """
    L = max(len(ids) for ids, _ in rendered)
    bos = tok.get_bos_token_id()
    rows, masks = [], []
    for ids, mask in rendered:
        pad = L - len(ids)
        rows.append(ids + [bos] * pad)              # value irrelevant: masked out below
        masks.append(mask + [0] * pad)
    batch = torch.tensor(rows, dtype=torch.long)                          # (N, L)
    inputs = batch[:, :-1].to(device)                                     # (N, T) T=L-1
    targets = batch[:, 1:].clone().to(device)                             # (N, T)
    mask_t = torch.tensor(masks, dtype=torch.int8)[:, 1:].to(device)      # (N, T) shifted
    targets[mask_t == 0] = -1                                             # only response tokens count
    return inputs, targets

def batch_pairs(tok, pairs, device):
    """pairs: list of (prompt_conv, chosen_text, rejected_text), length B.
    Returns inputs (2B, T), targets (2B, T) via pad_rendered.
    Row layout: [chosen_0..chosen_{B-1}, rejected_0..rejected_{B-1}] — one tensor,
    so policy and reference each run ONE forward for the whole batch ("双前向" total).
    """
    rendered = ([render_pair(tok, conv, chosen) for conv, chosen, _ in pairs] +
                [render_pair(tok, conv, rejected) for conv, _, rejected in pairs])
    return pad_rendered(tok, rendered, device)                            # (2B, T) each

def sequence_logprob(model, inputs, targets):
    """Sum of per-token log-probs over the counted (non -1) positions.

    inputs  (N, T) int64 -> logits (N, T, V) fp32 (lab1: crop + .float() + softcap)
    log_softmax over V; gather the target token's logp at each position; zero out
    ignored positions; sum over T -> (N,) fp32.
    """
    logits = model(inputs)                                # (N, T, V) fp32
    logp = F.log_softmax(logits, dim=-1)                  # (N, T, V) normalize over vocab
    safe = targets.clamp_min(0)                           # (N, T) -1 -> 0 for gather
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # (N, T) logp of target token
    tok_logp = tok_logp * (targets >= 0)                  # ignored positions contribute 0
    return tok_logp.sum(dim=-1)                           # (N,) sequence log-prob (sum, not mean)

# -----------------------------------------------------------------------------

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """Koven's spec (2026-09-19): margin = beta * [(pi_c - ref_c) - (pi_r - ref_r)],
    Bradley-Terry NLL on it. Caveats applied: -logsigmoid (not sigma, and the stable
    fused form, never log(sigmoid)); beta inside the sigmoid; batch mean."""
    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))   # (B,)
    loss = -F.logsigmoid(margin).mean()                                         # scalar fp32
    stats = {"margin": margin.mean().item(),
             "accuracy": (margin > 0).float().mean().item()}   # fraction correctly ordered
    return loss, stats

def kto_loss(pi_logp, ref_logp, is_good, beta, lam_d=1.0, lam_u=1.0):
    """KTO (unpaired). Koven's decisions (2026-09-20): z0 = batch-mean of r, detached
    and clamped >= 0 (simplified reference point); lam_d/lam_u manual knobs, default 1.

    pi_logp/ref_logp: (N,) fp32 sequence logps; is_good: (N,) bool.
    Value function, NOT an NLL: bounded 1 - sigma(...), gradients saturate for
    hopeless samples (label-noise robustness, the dual of DPO's unbounded -logsig)."""
    r = beta * (pi_logp - ref_logp)                       # (N,) implicit rewards
    z0 = r.detach().mean().clamp_min(0.0)                 # scalar reference point, NO grad:
                                                          # the yardstick must not chase itself
    val = torch.where(is_good, torch.sigmoid(r - z0),     # good: want r above the batch drift
                      torch.sigmoid(z0 - r))              # bad:  want r below it
    lam = torch.where(is_good,
                      torch.full_like(r, lam_d), torch.full_like(r, lam_u))
    loss = (lam * (1.0 - val)).mean()                     # scalar fp32
    stats = {"z0": z0.item(),
             "r_good": r[is_good].mean().item() if is_good.any() else float("nan"),
             "r_bad": r[~is_good].mean().item() if (~is_good).any() else float("nan")}
    return loss, stats


def dpo_step(policy, ref, tok, pairs, beta, device):
    """One DPO training step, minus optimizer.step() (the caller owns the loop)."""
    inputs, targets = batch_pairs(tok, pairs, device)     # (2B, T) each
    B = len(pairs)
    pi_logp = sequence_logprob(policy, inputs, targets)   # (2B,) WITH grad
    with torch.no_grad():
        ref_logp = sequence_logprob(ref, inputs, targets) # (2B,) anchor, no graph
    loss, stats = dpo_loss(pi_logp[:B], pi_logp[B:], ref_logp[:B], ref_logp[B:], beta)
    return loss

if __name__ == "__main__":
    # smoke: tiny fresh policy/ref, one synthetic pair, logp plumbing end-to-end.
    torch.manual_seed(0)
    tok = get_tokenizer()
    cfg = GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
    policy = GPT(cfg); policy.init_weights()
    ref = GPT(cfg); ref.load_state_dict(policy.state_dict())
    ref.eval()
    [p.requires_grad_(False) for p in ref.parameters()]
    prompt = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    pairs = [(prompt, "4", "5 or maybe 6")]
    inputs, targets = batch_pairs(tok, pairs, "cpu")
    assert inputs.shape == targets.shape and inputs.shape[0] == 2
    lp = sequence_logprob(policy, inputs, targets)
    assert lp.shape == (2,) and lp.dtype == torch.float32 and (lp < 0).all()
    lr = sequence_logprob(ref, inputs, targets)
    assert torch.allclose(lp, lr)   # identical weights => identical logps
    loss = dpo_step(policy, ref, tok, pairs, beta=0.1, device="cpu")
    import math
    assert abs(loss.item() - math.log(2)) < 1e-5, "policy==ref must give ln2"
    loss.backward()
    assert policy.lm_head.weight.grad is not None
    print("lab3 mini: ok (dpo_loss live, init loss = ln2)")
