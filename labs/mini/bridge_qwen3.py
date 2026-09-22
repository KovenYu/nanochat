"""
Bridge mini: Qwen3-0.6B forward pass, lab1 style, loading the HF safetensors directly.

Why this exists (SPEC, Lab 1 deferred item): Phase 1 trains the Qwen3 family in
LLaMA-Factory. This file is the smallest member of that family written the way lab1_gpt.py
was written, so every architectural difference from nanochat is visible in one diff
(the wrap-up ceremony is bridge_qwen3.py vs lab1_gpt.py).

Reference = HF transformers 5.8.0 (the Phase A env), file
  transformers/models/qwen3/modeling_qwen3.py   (cited below as mq3:<line>)
Run in that env:  source /svl/u/koven/sft-phase-a/env.sh

Kept, semantically identical to the reference: RMSNorm with learnable scale (fp32
internal), GQA, per-head QK-norm applied BEFORE RoPE, HF rotate_half RoPE layout,
SwiGLU MLP, tied embedding/lm_head, no softcap, no input-embedding norm, no residual
extras. Attention kernel: torch SDPA, which is also HF's default (attn_implementation
"sdpa"), so the kernel is the same, not just the math.
Cut: KV cache / generate, sliding-window branch (config says none; asserted), tools in
the chat template, reasoning_content passthrough.

Dtype scheme: the checkpoint is bf16 EVERYWHERE (norm scales included); the whole model
runs bf16. RMSNorm and RoPE tables compute in fp32 internally and cast back. Logits stay
bf16 (mq3:493: "do not upcast them to float if we are not computing the loss"); the loss
path upcasts.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SNAPSHOT = ("/svl/u/koven/sft-phase-a/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/"
            "c1899de289a04d12100db370d81485cdf75e47ca")
COMPUTE_DTYPE = torch.bfloat16
IGNORE_INDEX = -100                                         # HF convention (nanochat uses -1)


@dataclass
class Qwen3Config:                                          # config.json keys, verbatim names
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128         # NOT hidden_size // num_attention_heads (= 64) — set explicitly
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 40960

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            c = json.load(f)
        # Branches this file does not implement must be OFF in the config; fail, don't fall back.
        assert c["model_type"] == "qwen3", c["model_type"]
        assert c["hidden_act"] == "silu", c["hidden_act"]
        assert c["attention_bias"] is False
        assert c["rope_scaling"] is None and c["sliding_window"] is None and not c["use_sliding_window"]
        return cls(**{k: c[k] for k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):                                   # mq3:50
    """RMSNorm WITH a learnable per-channel scale (nanochat's norm() has none)."""
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))         # (dim,)
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype                               # bf16 in
        xf = x.to(torch.float32)                            # (..., dim) fp32 for the statistics
        var = xf.pow(2).mean(-1, keepdim=True)              # (..., 1)
        xf = xf * torch.rsqrt(var + self.eps)               # (..., dim) unit RMS
        return self.weight * xf.to(input_dtype)             # (..., dim) bf16 * bf16 -> bf16


def rope_cos_sin(config, T, device, dtype):                 # mq3:124-133 (inv_freq), mq3:137-148 (tables)
    """HF layout: angle for pair i is DUPLICATED at channels i and i + D/2 -> cos/sin are (1, T, D)."""
    D = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32, device=device) / D))  # (D/2,)
    t = torch.arange(T, dtype=torch.float32, device=device)                # (T,)
    freqs = torch.outer(t, inv_freq)                                       # (T, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                # (T, D)
    return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]           # (1, T, D) each


def rotate_half(x):                                         # mq3:151
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]   # 2 x (..., D/2)
    return torch.cat((-x2, x1), dim=-1)                     # (..., D)


def apply_rotary_pos_emb(q, k, cos, sin):                   # mq3:159-181, unsqueeze_dim=1
    """q: (B, Hq, T, D), k: (B, Hkv, T, D), cos/sin: (1, T, D) -> unsqueeze to (1, 1, T, D).
    y = x*cos + rotate_half(x)*sin: for pair (x1, x2): y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin.
    Compare nanochat's y1 = x1 cos + x2 sin, y2 = -x1 sin + x2 cos: the OPPOSITE rotation
    direction. A trained checkpoint is bound to its own convention."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)           # (1, 1, T, D)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):                                 # mq3:222
    def __init__(self, config):
        super().__init__()
        C, D = config.hidden_size, config.head_dim
        self.n_head, self.n_kv_head, self.head_dim = config.num_attention_heads, config.num_key_value_heads, D
        self.q_proj = nn.Linear(C, self.n_head * D, bias=False)      # (C -> Hq*D)  = (1024 -> 2048)
        self.k_proj = nn.Linear(C, self.n_kv_head * D, bias=False)   # (C -> Hkv*D) = (1024 -> 1024)
        self.v_proj = nn.Linear(C, self.n_kv_head * D, bias=False)   # (C -> Hkv*D)
        self.o_proj = nn.Linear(self.n_head * D, C, bias=False)      # (Hq*D -> C) = (2048 -> 1024)
        self.q_norm = RMSNorm(D, config.rms_norm_eps)       # per-head, over D only (mq3:248)
        self.k_norm = RMSNorm(D, config.rms_norm_eps)
        self.scaling = D ** -0.5                            # mq3:232; no 1.2 sharpening factor

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim)).transpose(1, 2)     # (B, Hq, T, D)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)               # (B, Hkv, T, D)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)         # norm BEFORE rope (nanochat: after); shapes unchanged
        rep = self.n_head // self.n_kv_head                 # mq3:184 repeat_kv
        k = k.repeat_interleave(rep, dim=1)                 # (B, Hq, T, D)
        v = v.repeat_interleave(rep, dim=1)                 # (B, Hq, T, D)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)  # (B, Hq, T, D)
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)                 # (B, T, Hq*D)
        return self.o_proj(y)                               # (B, T, C)


class MLP(nn.Module):                                       # mq3:70
    def __init__(self, config):
        super().__init__()
        C, I = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(C, I, bias=False)        # (C -> I) = (1024 -> 3072)
        self.up_proj = nn.Linear(C, I, bias=False)          # (C -> I)
        self.down_proj = nn.Linear(I, C, bias=False)        # (I -> C)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))   # SwiGLU; hidden (B, T, I)


class DecoderLayer(nn.Module):                              # mq3:294
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)   # (B, T, C) pre-norm residual
        x = x + self.mlp(self.post_attention_layernorm(x))          # (B, T, C)
        return x


class Qwen3Model(nn.Module):                                # mq3:357
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)   # (V, C) V already padded
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen3ForCausalLM(nn.Module):                          # mq3:442; attribute names = checkpoint keys
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)   # (C -> V)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight   # SAME Parameter object (aliasing wanted here)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.max_position_embeddings
        x = self.model.embed_tokens(idx)                    # (B, T, C) bf16; NOT normed (nanochat norms here)
        cos, sin = rope_cos_sin(self.config, T, x.device, x.dtype)   # (1, T, D) each, bf16
        for layer in self.model.layers:
            x = layer(x, cos, sin)                          # (B, T, C)
        x = self.model.norm(x)                              # (B, T, C)
        logits = self.lm_head(x)                            # (B, T, V) bf16, no softcap, no crop
        if targets is not None:                             # HF ForCausalLMLoss: upcast, ignore -100
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=IGNORE_INDEX)
        return logits


def load_qwen3(device):
    """safetensors -> mini. The file carries lm_head.weight even though the config ties it
    (a redundant copy, 311 MB of the 1.5 GB). We check it IS the embedding, then copy (not
    assign) so the tie survives: a tied module's state_dict lists both keys for ONE tensor,
    so strict loading wants both and writes the same values twice."""
    from safetensors.torch import load_file
    config = Qwen3Config.from_json(os.path.join(SNAPSHOT, "config.json"))
    sd = load_file(os.path.join(SNAPSHOT, "model.safetensors"))       # all bf16
    if config.tie_word_embeddings:
        assert torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]), "file lm_head != embed"
    model = Qwen3ForCausalLM(config).to(COMPUTE_DTYPE)
    model.load_state_dict(sd, strict=True)                  # copy into existing (tied) params
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "tie broken"
    return model.to(device).eval(), config


# -----------------------------------------------------------------------------
# Tokenizer + chat template (text-only subset of tokenizer_config.json's Jinja template)

def get_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(SNAPSHOT, "tokenizer.json"))

def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids

def render_chat(tok, messages, enable_thinking=True):
    """messages: [{"role": system|user|assistant, "content": str}] -> (ids, mask), lab0 style.
    Mirrors LLaMA-Factory's `qwen3` template (ReasoningTemplate, the tool Phase 1 uses):
      - no BOS anywhere; every turn is <|im_start|>{role}\n{content}<|im_end|>\n
      - EVERY assistant turn gets the empty reasoning slot '<think>\n\n</think>\n\n'
        (template.py:483-517). enable_thinking=True (LF default): the slot is part of the
        response -> supervised. False: appended to the prompt -> not supervised.
      - mask: assistant header not supervised; content, <|im_end|> and the trailing \n are
        (Phase A, sft-phase-a/DELIVERABLES.md §4).
    HF's own chat_template differs: it inserts the slot ONLY on the final assistant turn
    (loop.last), so LF and HF agree on single-turn samples and disagree on multi-turn.
    Segments are encoded between special tokens exactly as the full-string tokenizer
    splits them (added tokens are split off first), so ids match either tool's ids."""
    im_start, im_end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    think, think_end = tok.token_to_id("<think>"), tok.token_to_id("</think>")
    ids, mask = [], []
    def put(seg_ids, m):
        ids.extend(seg_ids); mask.extend([m] * len(seg_ids))
    for m in messages:
        role, content = m["role"], m["content"]
        if role in ("system", "user"):
            put([im_start] + encode(tok, f"{role}\n{content}") + [im_end] + encode(tok, "\n"), 0)
        elif role == "assistant":
            assert "<think>" not in content and "</think>" not in content, "plain content only"
            put([im_start] + encode(tok, "assistant\n"), 0)              # header: never supervised
            put([think] + encode(tok, "\n\n") + [think_end] + encode(tok, "\n\n"), int(enable_thinking))
            put(encode(tok, content) + [im_end] + encode(tok, "\n"), 1)
        else:
            raise ValueError(role)
    return ids, mask


if __name__ == "__main__":
    # eyeball smoke: load the real 0.6B, one forward, the top-1 continuation must be sane.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_qwen3(device)
    tok = get_tokenizer()
    ids = encode(tok, "The capital of France is")
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device))   # (1, T, V) bf16
    assert logits.shape == (1, len(ids), cfg.vocab_size) and logits.dtype == COMPUTE_DTYPE
    top = tok.decode([logits[0, -1].argmax().item()])
    print("bridge mini: ok; top-1 after 'The capital of France is' ->", repr(top))
    r_ids, r_mask = render_chat(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert len(r_ids) == len(r_mask) and sum(r_mask) > 0
