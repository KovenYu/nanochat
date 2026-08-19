"""
Lab 1 mini: the nanochat GPT model — forward pass only.

Distilled from nanochat/gpt.py (tag speedrun-4xh200-baseline). Kept, semantically identical
(challenge (2) loads the real d24 checkpoint into this file and must match logits):
  - architecture: rotary (no positional embeddings), QK norm, untied wte/lm_head, relu^2 MLP,
    GQA support, sliding-window pattern, and the modded-nanogpt-style residual-stream extras:
    value embeddings + per-head gates, smear, backout, resid/x0 lambdas, logit softcap
  - init_weights, exactly
  - the naive autoregressive generate()
Cut, per SPEC: KV-cache inference path (that is challenge (3)'s exercise), optimizer setup
(Lab 2), FLOPs/MFU estimators, fp8, DDP/meta-device init, checkpoint robustness.

The ONE functional substitution: the reference calls a custom flash_attn wrapper (FA3 with
SDPA fallback, nanochat/flash_attention.py); this file uses torch SDPA with an explicit
banded causal mask. Same math, different kernel.

Dtype scheme (predict it before tracing!): master weights fp32 EXCEPT embeddings (stored
bf16); Linear casts its weight to the activation dtype at matmul time, so the whole stream
runs bf16 from wte onward; logits are cropped, cast to fp32, then softcapped.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

COMPUTE_DTYPE = torch.bfloat16  # nanochat/common.py autodetects; bf16 on any modern GPU


@dataclass
class GPTConfig:                                            # nanochat/gpt.py:29
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6      # query heads
    n_kv_head: int = 6   # key/value heads; < n_head => GQA
    n_embd: int = 768
    window_pattern: str = "SSSL"  # tiled over layers; S=quarter context, L=full. Last layer always L.


def norm(x):                                                # nanochat/gpt.py:42
    # RMSNorm with NO learnable scale. Normalizes the last dim to unit RMS; runs in bf16.
    return F.rms_norm(x, (x.size(-1),))                     # same shape as x


class Linear(nn.Linear):                                    # nanochat/gpt.py:45
    """Weight stays fp32 (optimizer precision); cast to the activation dtype per matmul.
    This replaces autocast in the reference."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))   # (..., in) -> (..., out)


def has_ve(layer_idx, n_layer):                             # nanochat/gpt.py:53
    # Value embeddings on alternating layers, parity chosen so the LAST layer always has one.
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):                          # nanochat/gpt.py:57
    # x: (B, T, H, D). cos/sin: (1, T, 1, D/2), broadcast over batch and heads.
    # Rotates channel pairs (x1[i], x2[i]) by position-dependent angles; rotation is by
    # -theta vs the textbook convention (transpose) — functionally equivalent, kept for
    # checkpoint compatibility.
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]                         # 2 x (B, T, H, D/2)
    y1 = x1 * cos + x2 * sin                                # (B, T, H, D/2)
    y2 = x1 * (-sin) + x2 * cos                             # (B, T, H, D/2)
    return torch.cat([y1, y2], 3)                           # (B, T, H, D); a rotation => norm-preserving


def sdpa_banded(q, k, v, window_left):
    """Replaces flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window_left, 0)).
    q: (B, T, Hq, D), k/v: (B, T, Hkv, D)  [FA3's native layout] -> returns (B, T, Hq, D).
    Attend to positions j with  i - window_left <= j <= i  (causal band)."""
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    q = q.transpose(1, 2)                                   # (B, Hq, T, D)  SDPA layout
    k = k.transpose(1, 2)                                   # (B, Hkv, T, D)
    v = v.transpose(1, 2)                                   # (B, Hkv, T, D)
    if Hkv != Hq:                                           # GQA: every kv head serves Hq/Hkv query heads
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)                 # (B, Hq, T, D)
        v = v.repeat_interleave(rep, dim=1)                 # (B, Hq, T, D)
    i = torch.arange(T, device=q.device)[:, None]           # (T, 1) query pos
    j = torch.arange(T, device=q.device)[None, :]           # (1, T) key pos
    mask = (j <= i) & (i - j <= window_left)                # (T, T) bool, True = may attend
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, Hq, T, D); scale = 1/sqrt(D)
    return y.transpose(1, 2)                                # (B, T, Hq, D)


class CausalSelfAttention(nn.Module):                       # nanochat/gpt.py:67
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)     # (C -> Hq*D)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # (C -> Hkv*D)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # (C -> Hkv*D)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)                  # (C -> C)
        # Per-head gate for the value embedding, computed from the first 12 channels of the input.
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()                                  # x is already norm()ed by Block

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)      # (B, T, Hq, D)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, Hkv, D)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, Hkv, D)

        # Value residual (ResFormer): blend a per-token value EMBEDDING into v, gated per kv head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)       # (B, T, Hkv, D)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, Hkv) in (0,3)
            v = v + gate.unsqueeze(-1) * ve                         # (B, T, Hkv, D)

        cos, sin = cos_sin                                  # each (1, T, 1, D/2)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # shapes unchanged
        q, k = norm(q), norm(k)                             # QK norm: unit RMS per head vector
        q = q * 1.2                                         # sharper attention; the effective
        k = k * 1.2                                         # softmax temp scales by 1.2^2 = 1.44

        left = window_size[0]                               # (left, 0); left == config.sequence_len for L
        y = sdpa_banded(q, k, v, left)                      # (B, T, Hq, D)

        y = y.contiguous().view(B, T, -1)                   # (B, T, C) reassemble heads
        y = self.c_proj(y)                                  # (B, T, C)
        return y


class MLP(nn.Module):                                       # nanochat/gpt.py:131
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)    # (C -> 4C)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)  # (4C -> C)

    def forward(self, x):
        x = self.c_fc(x)                                    # (B, T, 4C)
        x = F.relu(x).square()                              # relu^2: zero for x<=0, x^2 for x>0
        x = self.c_proj(x)                                  # (B, T, C)
        return x


class Block(nn.Module):                                     # nanochat/gpt.py:144
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # Pre-norm residual: norm feeds the sublayer, the residual stream itself is NOT normed.
        x = x + self.attn(norm(x), ve, cos_sin, window_size)  # (B, T, C)
        x = x + self.mlp(norm(x))                             # (B, T, C)
        return x


class GPT(nn.Module):                                       # nanochat/gpt.py:156
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)  # list of (left, 0), len n_layer
        # Pad vocab so the wte/lm_head matmul dims are multiples of 64 (tensor cores, DDP).
        # Pure optimization: forward() crops logits back to vocab_size.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),          # (Vp, C), stored bf16
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)  # (C -> Vp), fp32
        # Per-layer scalars on the residual stream:
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # scales the stream, init ~1
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # blends x0 back in
        # Smear: mix the previous token's embedding into the current one (cheap bigram info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract the mid-layer residual before the final norm
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings: full (Vp, Hkv*D) embedding tables, only on has_ve layers
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim)
                                           for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # Rotary tables, over-computed 10x past sequence_len. persistent=False => NOT in the
        # checkpoint (recomputed at load). Remember this for the checkpoint-anatomy predictions.
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # (1, 10T, 1, D/2) bf16
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):                                 # nanochat/gpt.py:204
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5   # Uniform(-s, s) has the same std as Normal(0, 1/sqrt(n_embd))
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)   # uniform avoids outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)         # residual writes start at 0
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))  # 1.15 -> 1.05
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))     # 0.20 -> 0.05
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=self.transformer.wte.weight.device)
        self.cos, self.sin = cos, sin
        # ONLY the embedding tables are stored in bf16; all other params stay fp32 masters.
        self.transformer.wte.to(dtype=COMPUTE_DTYPE)
        for ve in self.value_embeds.values():
            ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):  # nanochat/gpt.py:270
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)  # (D/2,)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))                             # (D/2,) 1 .. 1/base
        t = torch.arange(seq_len, dtype=torch.float32, device=device)                     # (10T,)
        freqs = torch.outer(t, inv_freq)                                                  # (10T, D/2)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]                               # (1, 10T, 1, D/2)

    def _compute_window_sizes(self, config):                # nanochat/gpt.py:287
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128    # quarter context, ceil to FA3 tile (2048 -> 512)
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)                 # final layer always full context
        return window_sizes

    def forward(self, idx, targets=None, loss_reduction='mean'):    # nanochat/gpt.py:462
        B, T = idx.size()                                   # idx: (B, T) int64 token ids
        assert T <= self.cos.size(1), f"T={T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]          # (1, T, 1, D/2) each

        x = self.transformer.wte(idx)                       # (B, T, C) bf16 (table is bf16)
        x = x.to(COMPUTE_DTYPE)                             # no-op for bf16
        x = norm(x)                                         # unit-RMS embeddings

        # Smear: x[t] += sigmoid-gated fraction of x[t-1]; position 0 unchanged
        assert T > 1, "Training forward pass should have T > 1"
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))  # (B, T-1, 1)
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)                          # (B, T, C)

        x0 = x                                              # (B, T, C) saved for x0 blending
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0        # (B, T, C)
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None  # (B, T, Hkv*D) or None
            x = block(x, ve, cos_sin, self.window_sizes[i])                # (B, T, C)
            if i == backout_layer:
                x_backout = x                               # snapshot mid-depth stream
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout            # (B, T, C)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)                            # (B, T, Vp)  the big tensor
        logits = logits[..., :self.config.vocab_size]       # (B, T, V) crop vocab padding
        logits = logits.float()                             # fp32 for softcap + loss
        logits = softcap * torch.tanh(logits / softcap)     # smooth clamp to [-15, 15]

        if targets is not None:                             # targets: (B, T) int64, -1 = ignore
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        return logits                                       # (B, T, V) fp32

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):  # nanochat/gpt.py:530
        """Naive: re-forwards the WHOLE sequence for every new token (no KV cache — that is
        challenge (3)). batch=1, plain python list in, ints out."""
        assert isinstance(tokens, list)
        device = self.transformer.wte.weight.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)   # (1, T)
        for _ in range(max_tokens):
            logits = self.forward(ids)                      # (1, T, V) — recomputed from scratch!
            logits = logits[:, -1, :]                       # (1, V) only the last position matters
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (1, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)         # (1, V)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)  # (1, 1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)   # (1, 1)
            ids = torch.cat((ids, next_ids), dim=1)         # (1, T+1) grows every step
            yield next_ids.item()


if __name__ == "__main__":
    # eyeball smoke test: tiny config from SPEC, CPU, forward + backward + 3 greedy tokens.
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=16, vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    model = GPT(cfg)
    model.init_weights()
    B, T = 2, 16
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))
    loss = model(idx, targets)
    assert loss.dtype == torch.float32 and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None and model.transformer.wte.weight.grad is not None
    logits = model(idx)
    assert logits.shape == (B, T, 256) and logits.dtype == torch.float32
    assert logits.abs().max() <= 15.0
    toks = list(model.generate(list(range(5)), max_tokens=3, temperature=0.0))
    assert len(toks) == 3 and all(0 <= t < 256 for t in toks)
    print("lab1 mini: ok")
