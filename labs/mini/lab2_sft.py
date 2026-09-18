"""
Lab 2 mini: SFT = a training loop + masking.

Distilled from scripts/chat_sft.py (tag speedrun-4xh200-baseline). Kept, semantically identical:
  - the SFT dataloader: bestfit-PAD packing of whole conversations (never crop), BOS padding
  - the heart: shift-by-one + loss mask -> targets with ignore_index=-1
  - the training loop: grad accumulation, progress-based LR schedule, Muon momentum ramp
  - optimizer setup (distilled from nanochat/gpt.py:422 setup_optimizer, which lab1's mini cut)
  - val bpb evaluation (distilled from nanochat/loss_eval.py evaluate_bpb)
  - loading the frozen d24 base checkpoint into the lab1 mini GPT
Cut, per SPEC: DDP/distributed, wandb, ChatCORE eval, fp16 GradScaler, torch.compile, gc
management, checkpoint robustness, the real task mixture (SmolTalk/MMLU/GSM8K -> synthetic
conversations or a jsonl slice).

Distillation decision: the optimizer CLASS (MuonAdamW) is imported from nanochat.optim —
its internals (ZeRO-2 sharding, fused kernels) are Lab 6 territory; what Lab 2 studies is
the param GROUPING, which is distilled here in setup_optimizer().

Two configs (SPEC):
  eyeball : fresh tiny GPT + synthetic conversations, single GPU, seconds.
            python labs/mini/lab2_sft.py --source tiny
  shadow  : frozen d24 base checkpoint + real data jsonl, milestone only.
            python labs/mini/lab2_sft.py --source d24 --data <path.jsonl>
"""

import argparse
import json
import math
import os
import time

import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lab0_tokenizer import get_tokenizer, get_token_bytes            # noqa: E402
from lab1_gpt import GPT, GPTConfig, COMPUTE_DTYPE                   # noqa: E402
from nanochat.optim import MuonAdamW                                 # noqa: E402

D24_DIR = "/svl/u/koven/nanochat_data/base_checkpoints/d24-baseline-26413dc"
D24_STEP = 5568
OUT_ROOT = "/svl/u/koven/nanochat_data/labs"   # SPEC: all training outputs live under here

# -----------------------------------------------------------------------------
# Model loading

def load_d24(device):
    """Frozen d24 base checkpoint -> lab1 mini GPT. nanochat/checkpoint_manager.py:69-114."""
    with open(os.path.join(D24_DIR, f"meta_{D24_STEP:06d}.json")) as f:
        meta = json.load(f)
    model_data = torch.load(os.path.join(D24_DIR, f"model_{D24_STEP:06d}.pt"),
                            map_location=device, weights_only=True)
    # torch.compile saved the model as _orig_mod.*; strip the prefix (checkpoint_manager.py:93)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model = GPT(GPTConfig(**meta["model_config"]))
    model.load_state_dict(model_data, strict=True, assign=True)   # assign: adopt checkpoint tensors
    return model.to(device), meta

def build_tiny(device):
    """SPEC eyeball config, fresh init."""
    model = GPT(GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4, n_kv_head=4, n_embd=64))
    model.init_weights()
    return model.to(device)

# -----------------------------------------------------------------------------
# Data: conversations -> packed (inputs, targets) batches

def synthetic_conversations(seed=0):
    """Endless stream of small multi-turn conversations (eyeball stand-in for SmolTalk)."""
    import random
    rng = random.Random(seed)
    while True:
        n_turns = rng.choice([1, 1, 2, 3])
        messages = []
        for _ in range(n_turns):
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            messages.append({"role": "user", "content": f"What is {a}+{b}? " + "Explain. " * rng.randint(0, 5)})
            messages.append({"role": "assistant", "content": f"{a}+{b}={a+b}. " + "Easy. " * rng.randint(0, 3)})
        yield {"messages": messages}

def jsonl_conversations(path):
    """One conversation per line: {"messages": [...]}. Loops forever (epochs)."""
    while True:
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

def sft_data_generator(conversations, tokenizer, B, T, device, buffer_size=100, mask_history=False):
    """scripts/chat_sft.py:180-298, single rank. Yields (inputs, targets), both (B, T).

    Packing invariants (the heart of Lab 2):
      - row capacity is T+1 raw tokens; a row yields T inputs and T targets after the shift
      - conversations are packed WHOLE (bestfit); when nothing in the buffer fits, the row is
        PADDED with BOS (never cropped) — cf. pretraining, which crops and discards ~35%
      - render_conversation caps a conversation at 2048 tokens, which must be <= T+1,
        or packing deadlocks (lab0's truncation discussion — the two constants meet here)
    """
    assert T + 1 >= 2048 or True  # tiny configs use T+1 < 2048; synthetic convs are short. See predictions.
    row_capacity = T + 1                                    # +1: last token is target-only
    bos = tokenizer.get_bos_token_id()
    conv_buffer = []                                        # list of (ids, mask), each len <= row_capacity

    if mask_history:
        assistant_start = tokenizer.encode_special("<|assistant_start|>")

    def refill():
        while len(conv_buffer) < buffer_size:
            ids, mask = tokenizer.render_conversation(next(conversations))   # lists, same length
            assert len(ids) <= row_capacity, f"conversation ({len(ids)}) cannot fit a row ({row_capacity})"
            if mask_history:
                # View B: supervise only the LAST assistant turn. Per conversation, BEFORE
                # packing; judged on the token sequence (truncation-safe); narrowing only,
                # so python_output zeros inside the last turn are never resurrected.
                last_as = max((i for i, t in enumerate(ids) if t == assistant_start), default=None)
                mask = [m if last_as is not None and i > last_as else 0 for i, m in enumerate(mask)]
            conv_buffer.append((ids, mask))

    while True:
        rows, mask_rows, row_lengths = [], [], []
        for _ in range(B):
            row, mask_row = [], []                          # grow to exactly row_capacity
            padded_at = row_capacity                        # content length if padding happens
            while len(row) < row_capacity:
                refill()
                remaining = row_capacity - len(row)
                # best fit: LARGEST buffered conversation that fits entirely
                best_idx, best_len = -1, 0
                for i, (c, _) in enumerate(conv_buffer):
                    if best_len < len(c) <= remaining:
                        best_idx, best_len = i, len(c)
                if best_idx >= 0:
                    c, m = conv_buffer.pop(best_idx)
                    row.extend(c)                           # whole conversation, starts with BOS
                    mask_row.extend(m)
                else:
                    padded_at = len(row)                    # nothing fits -> pad the rest with BOS
                    row.extend([bos] * remaining)           # mask 0: padding is never supervised
                    mask_row.extend([0] * remaining)
            rows.append(row)                                # len row_capacity = T+1
            mask_rows.append(mask_row)
            row_lengths.append(padded_at)

        batch = torch.tensor(rows, dtype=torch.long)                          # (B, T+1) cpu
        inputs = batch[:, :-1].to(device=device, dtype=torch.int32)           # (B, T) what the model reads
        targets = batch[:, 1:].to(device=device, dtype=torch.int64)          # (B, T) what it must predict
        # the shift: mask[t] marks token t itself; targets[t] == row[t+1], so align with mask[1:]
        mask_t = torch.tensor(mask_rows, dtype=torch.int8)[:, 1:].to(device)  # (B, T)
        targets[mask_t == 0] = -1                                             # ignore_index
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:                  # mask the content->padding boundary too
                targets[i, content_len - 1:] = -1
        yield inputs, targets

# -----------------------------------------------------------------------------
# Optimizer setup (distilled from nanochat/gpt.py:422-460; SFT passes weight_decay=0)

def setup_optimizer(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, scalar_lr=0.5):
    matrix_params = list(model.transformer.h.parameters())          # Muon: all block matrices
    value_embeds_params = list(model.value_embeds.parameters())     # AdamW from here down
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params, x0_params = [model.resid_lambdas], [model.x0_lambdas]
    smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]
    dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5           # muP-style lr transfer
    param_groups = [
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):          # Muon groups stacked by shape
        group = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(kind='muon', params=group, lr=matrix_lr,
                                 momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0))
    optimizer = MuonAdamW(param_groups)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    return optimizer

# -----------------------------------------------------------------------------
# Val bpb (distilled from nanochat/loss_eval.py, single rank)

@torch.no_grad()
def evaluate_bpb(model, val_loader, steps, token_bytes):
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y = next(val_loader)
        loss2d = model(x, y, loss_reduction='none').view(-1)        # (B*T,) fp32 nats
        y = y.view(-1)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        nb = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
        total_nats += (loss2d * (nb > 0)).sum().item()              # only real-text targets count
        total_bytes += nb.sum().item()
    return float('inf') if total_bytes == 0 else total_nats / (math.log(2) * total_bytes)

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["tiny", "d24"], default="tiny")
    ap.add_argument("--data", type=str, default=None, help="jsonl of conversations (default: synthetic)")
    ap.add_argument("--device-batch-size", type=int, default=4)
    ap.add_argument("--total-batch-size", type=int, default=None, help="in tokens; default = one micro-batch")
    ap.add_argument("--num-iterations", type=int, default=20)
    ap.add_argument("--max-seq-len", type=int, default=None, help="default: tiny 128 / d24 2048")
    ap.add_argument("--init-lr-frac", type=float, default=0.8)      # chat_sft.py default
    ap.add_argument("--warmdown-ratio", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    if args.source == "d24":
        model, _ = load_d24(device)
        T = args.max_seq_len or 2048
    else:
        model = build_tiny(device)
        T = args.max_seq_len or 128
    B = args.device_batch_size
    total_batch = args.total_batch_size or (B * T)
    assert total_batch % (B * T) == 0, "total_batch_size must be a multiple of device_batch * seq_len"
    grad_accum_steps = total_batch // (B * T)

    conv_stream = jsonl_conversations(args.data) if args.data else synthetic_conversations()
    val_stream = jsonl_conversations(args.data) if args.data else synthetic_conversations(seed=1)
    train_loader = sft_data_generator(conv_stream, tokenizer, B, T, device)
    val_loader = sft_data_generator(val_stream, tokenizer, B, T, device)

    optimizer = setup_optimizer(model)
    for g in optimizer.param_groups:                       # chat_sft.py:330 init_lr_frac
        g["lr"] = g["initial_lr"] = g["initial_lr"] * args.init_lr_frac

    def lr_multiplier(progress):                           # warmup=0, constant, linear warmdown
        if progress <= 1.0 - args.warmdown_ratio:
            return 1.0
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return 1.0 - decay                                 # final_lr_frac = 0

    def muon_momentum(it):                                 # ramp 0.85 -> 0.95 over 300 steps
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    run_name = args.run_name or f"{args.source}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"source={args.source} B={B} T={T} grad_accum={grad_accum_steps} steps={args.num_iterations} out={out_dir}")

    x, y = next(train_loader)                              # prefetch (chat_sft.py:324)
    smooth, ema_beta = 0.0, 0.9
    for step in range(args.num_iterations):
        if args.eval_every > 0 and step % args.eval_every == 0:
            model.eval()
            bpb = evaluate_bpb(model, val_loader, args.eval_steps, token_bytes)
            print(f"step {step:04d} | val bpb {bpb:.4f}")
            model.train()
        t0 = time.time()
        for micro in range(grad_accum_steps):
            loss = model(x, y)                             # () fp32, mean over non-ignored targets
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()           # backward SUMS grads => pre-divide
            x, y = next(train_loader)                      # prefetch while GPU is busy
        progress = (step + 1) / args.num_iterations
        lrm = lr_multiplier(progress)
        mom = muon_momentum(step)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
            if g["kind"] == "muon":
                g["momentum"] = mom
        optimizer.step()
        model.zero_grad(set_to_none=True)
        smooth = ema_beta * smooth + (1 - ema_beta) * train_loss.item()
        debiased = smooth / (1 - ema_beta ** (step + 1))
        print(f"step {step+1:04d} | loss {debiased:.4f} | lrm {lrm:.2f} | dt {(time.time()-t0)*1000:.0f}ms")

    if device == "cuda":
        print(f"peak memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    print(f"saved {out_dir}/model_final.pt")

if __name__ == "__main__":
    main()
