"""
Lab 5 mini: GRPO (online RLVR) — distilled from scripts/chat_rl.py.

The reference is "GRPO" with four simplifications it states in its own docstring:
no KL/trust region, no PPO ratio+clip (on policy), token-level normalization (DAPO),
advantage = r - mean without dividing by sigma. This file keeps all four as the
default and, per Koven's 2026-09-27 decision (b), adds ONE flag on top:
--kl-beta > 0 loads lab3's frozen reference and adds a k3 KL penalty per token.

One optimizer step, single rank (the loop in chat_rl.py:248-300):
    for each of examples_per_step examples:
        rollout: sample num_samples completions of the SAME prompt      (get_batch, L85-146)
                 reward each one with the task's verifier -> (S,)
                 advantage = r - mean(r)                                 (L143-144)
        train:   in micro-batches of device_batch_size, logp (Bd, T) of the sampled
                 tokens under the CURRENT policy, pg_obj = sum(logp * A) / normalizer,
                 backward (accumulate)                                   (L264-271)
    optimizer.step()                                                     (L300)

Distillation decisions (each deliberate; SPEC "蒸馏规则"):
  1. No Engine. generate_batch re-forwards the whole (S, T) for every new token (no KV
     cache; lab1's generate, batched). It keeps Engine.generate_batch's CONTRACT exactly:
     a row is prompt + sampled tokens; mask 0 on the prompt, 1 on sampled tokens; the
     terminal token (<|assistant_end|> or <|bos|>) ends the row and is NOT kept in it.
     Consequence shared with the reference: the policy is never trained on emitting
     <|assistant_end|> (Stage 2 material). No tool use here, so no forced tokens: the
     only mask-0 positions are prompt and padding.
  2. No DDP. examples_per_rank == examples_per_step. The reference averages gradients
     across ranks inside optimizer.step (nanochat/optim.py:284,291, ReduceOp.AVG); here
     the per-rank normalizer already is the whole-step mean.
  3. Task = ArithmeticTask, the analog of GSM8K for a model SFT'd on lab2's data: the
     gold answer lives in the conversation's final assistant turn (which
     render_for_completion pops), the reward parses the SAMPLED text and compares.
     GSM8K's "#### <number>" marker and calculator parts are dataset format, not
     mechanism, and sft50 never saw them; the mini compares the last integer instead.
     Stage 3 (2) adds RovenTask: reward from the lab2 synthetic verifier schema.
  4. No wandb, no checkpoint, no pass@k eval loop; accept tests measure reward directly.
  5. model.eval()/model.train() toggles are kept where the reference has them (L100,
     L252). On this GPT they change nothing (no dropout anywhere; see Stage 1 Q1c).

Stage 3 (3), Koven 2026-09-29: --train-eos keeps <|assistant_end|> in the row with mask 1
so the stop decision gets gradient. Decisions: <|bos|> still ends the row and is NOT kept
(no gradient either way); rows cut at max_new_tokens are untouched (no overlong penalty);
default False == reference behaviour. The reward always sees the text WITHOUT the EOS.

Policy init = lab3's sft50 (d24, step 233). Optimizer = lab2's setup_optimizer
(Muon + AdamW, the reference's model.setup_optimizer), lr scaled by init_lr_frac and
ramped linearly to zero, exactly chat_rl.py:206-213.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(__file__))
from lab0_tokenizer import get_tokenizer                      # noqa: E402
from lab2_sft import setup_optimizer                          # noqa: E402
from lab3_dpo import load_sft50_state, build_gpt              # noqa: E402


@dataclass
class GRPOConfig:                                            # chat_rl.py:34-60 argparse, single rank
    num_samples: int = 16          # completions per prompt == the "group"
    device_batch_size: int = 8     # rows per forward (sampling AND training)
    examples_per_step: int = 2     # reference: 16 across 8 ranks == 2 per rank
    max_new_tokens: int = 64       # reference 256; sft50 (SmolTalk-style) explains, so 64 not 32
    temperature: float = 1.0
    top_k: int = 50
    init_lr_frac: float = 0.05     # lr = base lr * this, then linear rampdown to 0
    kl_beta: float = 0.0           # 0.0 == reference (no ref model loaded at all)
    train_eos: bool = False        # False == reference (terminal token dropped from the row)

    def __post_init__(self):
        assert self.num_samples % self.device_batch_size == 0, "samples must split into whole passes"


# -----------------------------------------------------------------------------
# Task: conversations with a gold final assistant turn + a verifier (tasks/gsm8k.py:36-116)

LAST_INT_RE = re.compile(r"-?\d+")

class ArithmeticTask:
    """N conversations "What is a+b?" -> gold "a+b=c." (the lab2 synthetic template).
    reward(conv, text) == 1.0 iff the last integer in the sampled text equals the gold sum
    (GSM8K.evaluate: parse both, exact match, float). Operands 2..9: sft50 (SmolTalk SFT)
    gets one-digit sums right in a minority of samples and two-digit ones almost never;
    the group needs mixed rewards to carry any gradient (Stage 1 Q3b). A sample cut off
    at max_new_tokens is scored on whatever integer it last wrote: truncation is a
    reward-0 event here as in the reference (no answer marker -> no reward)."""

    def __init__(self, n, seed=0, lo=2, hi=9):
        import random
        rng = random.Random(seed)
        self.conversations = []
        for _ in range(n):
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            self.conversations.append({"messages": [
                {"role": "user", "content": f"What is {a}+{b}?"},
                {"role": "assistant", "content": f"{a}+{b}={a + b}."}]})   # gold; popped for sampling

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

    @staticmethod
    def gold(conversation):
        return int(LAST_INT_RE.findall(conversation["messages"][-1]["content"])[-1])

    def reward(self, conversation, assistant_response):
        nums = LAST_INT_RE.findall(assistant_response)
        return float(bool(nums) and int(nums[-1]) == self.gold(conversation))   # 0.0 / 1.0


# -----------------------------------------------------------------------------
# Sampler with Engine.generate_batch's contract (nanochat/engine.py:277-299)

@torch.no_grad()
def generate_batch(model, tok, tokens, num_samples, max_tokens, temperature, top_k, seed, train_eos=False):
    """tokens: prompt ids (list[int], ends with <|assistant_start|>), length P.
    Returns (results, masks): num_samples lists each; results[i] = prompt + sampled tokens,
    masks[i] = [0]*P + [1]*len(sampled). Rows differ in length. Terminal tokens:
    train_eos=False (reference): neither <|assistant_end|> nor <|bos|> is kept.
    train_eos=True: <|assistant_end|> is kept with mask 1; <|bos|> is still not kept.
    No KV cache: step t re-forwards all (S, P+t) ids and reads the last position."""
    assistant_end = tok.encode_special("<|assistant_end|>")
    bos = tok.get_bos_token_id()
    device = model.transformer.wte.weight.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    ids = torch.tensor([tokens] * num_samples, dtype=torch.long, device=device)   # (S, P)
    results = [tokens.copy() for _ in range(num_samples)]
    masks = [[0] * len(tokens) for _ in range(num_samples)]   # prompt positions: 0
    completed = [False] * num_samples
    for _ in range(max_tokens):
        logits = model(ids)[:, -1, :]                         # (S, V) fp32; whole prefix recomputed
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))      # (S, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")       # keep only the top-k per row
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)          # (S, V)
            next_ids = torch.multinomial(probs, 1, generator=rng)       # (S, 1)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)       # (S, 1) greedy (eval only)
        for i, t in enumerate(next_ids[:, 0].tolist()):
            if completed[i]:
                continue                                      # row keeps sampling; output ignored
            if t == assistant_end and train_eos:
                results[i].append(t)
                masks[i].append(1)                            # stop decision: trained on
                completed[i] = True
            elif t == assistant_end or t == bos:
                completed[i] = True                           # terminal token NOT kept
            else:
                results[i].append(t)
                masks[i].append(1)                            # sampled token: trained on
        ids = torch.cat((ids, next_ids), dim=1)               # (S, P+t+1)
        if all(completed):
            break
    return results, masks


# -----------------------------------------------------------------------------
# One example's rollout -> training tensors (body of chat_rl.py get_batch, L90-146)

def rollout_example(policy, tok, task, example_idx, step, cfg, device):
    """Returns (sequences, inputs, targets, rewards, advantages) for ONE prompt:
    sequences  list[list[int]] length S = num_samples (prompt + sampled, unpadded)
    inputs     (S, T) int64      T = longest row - 1
    targets    (S, T) int64      -1 on prompt and padding positions
    rewards    (S,)   fp32       verifier output per sample
    advantages (S,)   fp32       rewards - rewards.mean()   (sums to 0 by construction)"""
    conversation = task[example_idx]
    tokens = tok.render_for_completion(conversation)           # prompt ids, last = <|assistant_start|>
    prefix_length = len(tokens)
    policy.eval()                                              # L100 (no-op here: no dropout)
    sequences, masks = [], []
    for sampling_step in range(cfg.num_samples // cfg.device_batch_size):
        seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF   # L106: distinct per pass
        seqs_b, masks_b = generate_batch(policy, tok, tokens, cfg.device_batch_size,
                                         cfg.max_new_tokens, cfg.temperature, cfg.top_k, seed,
                                         train_eos=cfg.train_eos)
        sequences.extend(seqs_b)
        masks.extend(masks_b)
    assistant_end = tok.encode_special("<|assistant_end|>")    # pad value (mask 0) and, with
                                                               # train_eos, the kept EOS (mask 1)
    def response_ids(seq):                                     # what the verifier sees: no EOS
        gen = seq[prefix_length:]
        return gen[:-1] if gen and gen[-1] == assistant_end else gen
    rewards = [task.reward(conversation, tok.decode(response_ids(seq))) for seq in sequences]

    max_length = max(len(seq) for seq in sequences)
    ids = torch.tensor([seq + [assistant_end] * (max_length - len(seq)) for seq in sequences],
                       dtype=torch.long, device=device)        # (S, L)
    mask_ids = torch.tensor([m + [0] * (max_length - len(m)) for m in masks],
                            dtype=torch.long, device=device)   # (S, L)
    inputs = ids[:, :-1]                                       # (S, T)
    targets = ids[:, 1:].clone()                               # (S, T) clone: in-place edit below
    targets[mask_ids[:, 1:] == 0] = -1                         # target j is ids[j+1], masked by mask[j+1]
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)   # (S,)
    advantages = rewards - rewards.mean()                      # (S,) baseline = group mean, no /sigma
    return sequences, inputs, targets, rewards, advantages


# -----------------------------------------------------------------------------
# Loss for one micro-batch (chat_rl.py:264-271, plus the optional KL term)

def grpo_microbatch_loss(policy, ref, inputs, targets, advantages, normalizer, kl_beta):
    """inputs/targets (Bd, T) int64; advantages (Bd,) fp32; normalizer = python int
    num_valid * num_passes * examples_per_step, so that the accumulated gradient over the
    whole step is a mean over valid tokens (token-level, DAPO style).
    Returns (loss, pg_obj, kl): loss is what .backward() runs on."""
    # cross_entropy(reduction='none') with ignore_index=-1 is 0 at ignored positions,
    # so logp is exactly 0 wherever targets == -1: prompt and padding drop out of the sums.
    logp = -policy(inputs, targets, loss_reduction="none").view_as(inputs)   # (Bd, T) fp32, grad
    pg_obj = (logp * advantages.unsqueeze(-1)).sum() / normalizer            # scalar
    loss = -pg_obj                                                           # maximize -> minimize
    kl = torch.zeros((), device=inputs.device)
    if kl_beta > 0:
        with torch.no_grad():
            ref_logp = -ref(inputs, targets, loss_reduction="none").view_as(inputs)   # (Bd, T), no graph
        d = ref_logp - logp                                    # (Bd, T) log(pi_ref/pi); 0 at ignored
        kl = (torch.exp(d) - d - 1).sum() / normalizer         # k3 estimator of KL(pi || pi_ref), >= 0
        loss = loss + kl_beta * kl
    return loss, pg_obj.detach(), kl.detach()


def grpo_step(policy, ref, tok, task, example_indices, step, cfg, device):
    """Gradient accumulation for one optimizer step (chat_rl.py:248-277); the caller owns
    optimizer.step() / zero_grad / lr. Returns per-step stats (python floats)."""
    rewards_all, lengths_all, pg_all, kl_all, truncated_all = [], [], [], [], []
    assistant_end = tok.encode_special("<|assistant_end|>")
    for example_idx in example_indices:
        sequences, inputs, targets, rewards, advantages = rollout_example(
            policy, tok, task, example_idx, step, cfg, device)
        policy.train()                                         # L252 (no-op here: no dropout)
        num_passes = inputs.size(0) // cfg.device_batch_size   # S / Bd
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * cfg.device_batch_size, (pass_idx + 1) * cfg.device_batch_size
            t_b = targets[b0:b1]                               # (Bd, T)
            num_valid = int((t_b >= 0).sum().clamp(min=1))     # sampled tokens in THIS micro-batch
            normalizer = num_valid * num_passes * len(example_indices)
            loss, pg_obj, kl = grpo_microbatch_loss(policy, ref, inputs[b0:b1], t_b,
                                                    advantages[b0:b1], normalizer, cfg.kl_beta)
            loss.backward()                                    # grads accumulate across passes/examples
            pg_all.append(pg_obj.item()); kl_all.append(kl.item())
        rewards_all.append(rewards.mean().item())
        lengths_all.extend(len(seq) for seq in sequences)
        P = len(tok.render_for_completion(task[example_idx]))
        truncated_all.extend(len(seq) - P == cfg.max_new_tokens and seq[-1] != assistant_end
                             for seq in sequences)             # hit the cap without stopping
    return {"reward": sum(rewards_all) / len(rewards_all),
            "seq_len": sum(lengths_all) / len(lengths_all),
            "truncated": sum(truncated_all) / len(truncated_all),
            "pg_obj": sum(pg_all), "kl": sum(kl_all)}


def load_policy_and_maybe_ref(cfg, device):
    """kl_beta == 0: policy only (the reference's memory footprint). > 0: plus lab3's
    frozen copy with its own storage (lab3_dpo.load_policy_and_ref)."""
    meta, sd = load_sft50_state(device)
    policy = build_gpt(meta, sd, device)
    if cfg.kl_beta == 0:
        return policy, None
    ref = build_gpt(meta, {k: v.clone() for k, v in sd.items()}, device)
    for pp, rp in zip(policy.parameters(), ref.parameters()):
        assert pp.data_ptr() != rp.data_ptr(), "policy/ref alias the same storage"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return policy, ref


def train(policy, ref, tok, task, cfg, num_steps, device, log=print):
    """chat_rl.py:206-300 for one rank: lr = base * init_lr_frac, linear rampdown to 0,
    examples cycle through the task in order (rank 0 of world size 1)."""
    optimizer = setup_optimizer(policy)
    for g in optimizer.param_groups:
        g["lr"] = g["lr"] * cfg.init_lr_frac
        g["initial_lr"] = g["lr"]
    order = list(range(len(task)))
    history = []
    for step in range(num_steps):
        idx = [order[(step * cfg.examples_per_step + j) % len(task)] for j in range(cfg.examples_per_step)]
        stats = grpo_step(policy, ref, tok, task, idx, step, cfg, device)
        lrm = 1.0 - step / num_steps                            # L210-212
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        optimizer.step()
        policy.zero_grad(set_to_none=True)
        history.append(stats)
        log(f"step {step}/{num_steps} | reward {stats['reward']:.3f} | seq_len {stats['seq_len']:.1f} "
            f"| truncated {stats['truncated']:.2f} | pg_obj {stats['pg_obj']:+.4f} | kl {stats['kl']:.5f} | lrm {lrm:.3f}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--examples", type=int, default=64, help="ArithmeticTask size")
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--train-eos", action="store_true", help="keep <|assistant_end|> in the row, mask 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = GRPOConfig(kl_beta=args.kl_beta, train_eos=args.train_eos)
    tok = get_tokenizer()
    task = ArithmeticTask(args.examples, seed=args.seed)
    policy, ref = load_policy_and_maybe_ref(cfg, device)
    train(policy, ref, tok, task, cfg, args.steps, device)


if __name__ == "__main__" and len(sys.argv) == 1:
    # eyeball smoke: tiny fresh policy, CPU, the whole plumbing once (no sft50 needed).
    torch.manual_seed(0)
    from lab1_gpt import GPT, GPTConfig
    tok = get_tokenizer()
    cfg = GRPOConfig(num_samples=4, device_batch_size=2, examples_per_step=1, max_new_tokens=5, kl_beta=0.1)
    gcfg = GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
    policy = GPT(gcfg); policy.init_weights()
    ref = GPT(gcfg); ref.load_state_dict(policy.state_dict()); ref.eval()
    [p.requires_grad_(False) for p in ref.parameters()]
    task = ArithmeticTask(3, seed=0)
    conv = task[0]
    assert task.reward(conv, f"blah {task.gold(conv)}") == 1.0 and task.reward(conv, "nothing") == 0.0
    prompt = tok.render_for_completion(conv)
    P = len(prompt)
    seqs, masks = generate_batch(policy, tok, prompt, 4, 5, 1.0, 50, seed=1)
    assert len(seqs) == 4 and all(s[:P] == prompt for s in seqs)
    assert all(m[:P] == [0] * P and set(m[P:]) <= {1} and len(m) == len(s) for s, m in zip(seqs, masks))
    assert all(len(s) <= P + 5 for s in seqs)
    _, inputs, targets, rewards, adv = rollout_example(policy, tok, task, 0, step=0, cfg=cfg, device="cpu")
    assert inputs.shape == targets.shape and inputs.shape[0] == 4 and inputs.dtype == torch.long
    assert (targets[:, :P - 1] == -1).all(), "prompt positions must be ignored"
    assert rewards.shape == (4,) and abs(adv.sum().item()) < 1e-5, "advantages sum to 0"
    loss, pg, kl = grpo_microbatch_loss(policy, ref, inputs[:2], targets[:2], adv[:2], 10, kl_beta=0.1)
    assert kl.item() == 0.0, "policy == ref must give zero KL"
    stats = grpo_step(policy, ref, tok, task, [0], step=0, cfg=cfg, device="cpu")
    assert policy.lm_head.weight.grad is not None or adv.abs().sum() == 0
    print(f"lab5 mini: ok (rollout {inputs.shape}, reward {stats['reward']:.2f}, kl {stats['kl']:.1e})")
elif __name__ == "__main__":
    main()
