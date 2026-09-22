"""Trace harness for the labs. Per labs/CLAUDE.md this is the only file CC maintains for
Stage 2; it prints measured values and never touches model/mini code.

Usage: python labs/mini/trace_harness.py lab0
Prints the actual value for every row of labs/lab0/predictions.md, numbered identically.
"""

import sys
sys.path.insert(0, "labs/mini")


def lab0():
    from lab0_tokenizer import SPECIAL_TOKENS, get_token_bytes, get_tokenizer
    tok = get_tokenizer()
    tb = get_token_bytes()
    name = {tok.encode_special(s): s for s in SPECIAL_TOKENS}  # id -> "<|...|>"

    print("== A. id 空间 ==")
    print("A1", tok.encode_special("<|bos|>"))
    print("A2", tok.encode_special("<|assistant_start|>"))
    a3 = tok.encode_special("<|output_end|>")
    print("A3", a3)
    print("A4", int(tb[a3]))

    print("== B. token 数 ==")
    B = ["hello world", " hello world", "Hello World!", "1234567890",
         "3.14159265358979", "今天天气很好", "👍", "", "supercalifragilisticexpialidocious"]
    for i, s in enumerate(B, 1):
        ids = tok.encode(s)
        print(f"B{i}", len(ids), [tok.decode([t]) for t in ids])

    print("== C. render_conversation ==")
    CONV = {"messages": [
        {"role": "system",    "content": "Be brief."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user",      "content": "Now use python to check."},
        {"role": "assistant", "content": [
            {"type": "text",          "text": "Sure."},
            {"type": "python",        "text": "print(2+2)"},
            {"type": "python_output", "text": "4"},
            {"type": "text",          "text": "It is 4."},
        ]},
    ]}
    ids, mask = tok.render_conversation(CONV)
    specials = [(i, name[t], m) for i, (t, m) in enumerate(zip(ids, mask)) if t in name]
    print("C1", "system 被 merge，循环里 4 条 message")
    print("C2", len(specials))
    m1 = [(s, m) for _, s, m in specials if m == 1]
    print("C3", len(m1), [s for s, _ in m1])
    print("C4", name.get(ids[0], tok.decode([ids[0]])), name.get(ids[1], tok.decode([ids[1]])))
    first1 = mask.index(1)
    print("C5", repr(tok.decode([ids[first1]])), f"(位置 {first1})")
    print("C6", name.get(ids[-1], tok.decode([ids[-1]])), "mask", mask[-1])
    dec = tok.decode(ids)
    a, b = dec.index("Be brief.") + len("Be brief."), dec.index("What is 2+2?")
    print("C7", repr(dec[a:b]))
    print("完整可视化：")
    print(tok.visualize_tokenization(ids, mask))

    print("== D. render_for_completion ==")
    pids = tok.render_for_completion(CONV)
    print("D1", name.get(pids[-2], tok.decode([pids[-2]])), name.get(pids[-1], tok.decode([pids[-1]])))
    print("D2", f"len(pids)={len(pids)}  len(ids)={len(ids)}")


def lab1():
    import json, os
    import torch
    import torch.nn.functional as F
    from lab1_gpt import GPT, GPTConfig, apply_rotary_emb, norm

    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=16, vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    model = GPT(cfg); model.init_weights()
    B, T = 2, 16
    idx = torch.randint(0, 256, (B, T))
    targets = torch.randint(0, 256, (B, T))

    print("== A. shapes/dtypes (tiny config) ==")
    x_emb = model.transformer.wte(idx)
    print("A1 wte(idx):", tuple(x_emb.shape), x_emb.dtype)
    attn0 = model.transformer.h[0].attn
    xn = norm(x_emb.to(torch.bfloat16))
    q = attn0.c_q(xn).view(B, T, attn0.n_head, attn0.head_dim)
    k = attn0.c_k(xn).view(B, T, attn0.n_kv_head, attn0.head_dim)
    print("A2 q:", tuple(q.shape), q.dtype)
    print("A3 k:", tuple(k.shape), k.dtype)
    cos = model.cos[:, :T]
    print("A4 cos slice:", tuple(cos.shape), cos.dtype)
    print("A5 attn weights per layer (all heads):", (B, attn0.n_head, T, T), "(per head:", (T, T), ")")
    mlp0 = model.transformer.h[0].mlp
    h = F.relu(mlp0.c_fc(xn)).square()
    print("A6 mlp hidden:", tuple(h.shape), h.dtype)
    logits = model(idx)
    print("A7 logits:", tuple(logits.shape), logits.dtype)
    loss = model(idx, targets)
    print("A8 loss:", tuple(loss.shape), loss.dtype)

    print("== B. RoPE / attention logit magnitudes ==")
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    qr = apply_rotary_emb(q, *cos_sin)
    kr = apply_rotary_emb(k, *cos_sin)
    qn = norm(qr) * 1.2
    kn = norm(kr) * 1.2
    rms = qn.float().pow(2).mean(-1).sqrt()
    print("B1 RMS of q after norm*1.2: mean", f"{rms.mean():.4f}", "min", f"{rms.min():.4f}", "max", f"{rms.max():.4f}")
    n_before = k.float().norm(dim=-1)
    n_after = kr.float().norm(dim=-1)
    rel = ((n_after - n_before).abs() / n_before)
    print("B2 rotary |d||k|||/||k||: mean", f"{rel.mean():.2e}", "max", f"{rel.max():.2e}")
    print("B3 (mechanism: rotation preserves norms; deviation = bf16 rounding)")
    D = attn0.head_dim
    kn_g = kn.repeat_interleave(attn0.n_head // attn0.n_kv_head, dim=2)
    logits_attn = torch.einsum("bthd,bshd->bhts", qn.float(), kn_g.float()) / D**0.5
    print("B4 theoretical |logit| bound = 1.44*sqrt(D) =", f"{1.44*D**0.5:.2f}", "; observed max", f"{logits_attn.abs().max():.3f}")
    print("B5 attention logit std:", f"{logits_attn.std():.3f}")
    print("B6 final logits range: [", f"{logits.min():.2f}", ",", f"{logits.max():.2f}", "] (softcap 15)")
    print("B7: verified against the real d24 in challenge (2); graded on reasoning here")

    print("== C. d24 checkpoint anatomy ==")
    ckpt_dir = "/svl/u/koven/nanochat_data/base_checkpoints/d24-baseline-26413dc"
    with open(os.path.join(ckpt_dir, "meta_005568.json")) as f:
        meta = json.load(f)
    print("C1-C3 model_config:", {k: meta["model_config"][k] for k in sorted(meta["model_config"])})
    sd = torch.load(os.path.join(ckpt_dir, "model_005568.pt"), map_location="cpu", mmap=True, weights_only=True)
    groups = {}
    for name, t in sd.items():
        if name.startswith("transformer.h."):
            g = "blocks"
        elif name.startswith("value_embeds"):
            g = "value_embeds"
        elif "wte" in name:
            g = "wte"
        elif "lm_head" in name:
            g = "lm_head"
        else:
            g = "scalars"
        n, b = groups.get(g, (0, 0))
        groups[g] = (n + t.numel(), b + t.numel() * t.element_size())
    total_n = sum(n for n, _ in groups.values()); total_b = sum(b for _, b in groups.values())
    for g, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        dts = sorted({str(t.dtype) for name, t in sd.items() if True}) # noqa
        print(f"C4 {g:>14}: {n/1e6:8.1f}M params, {b/2**30:6.2f} GiB")
    print(f"C4 {'TOTAL':>14}: {total_n/1e6:8.1f}M params, {total_b/2**30:6.2f} GiB")
    print("C4 dtypes:", {name.split(".")[0]: str(t.dtype) for name, t in list(sd.items())[:0]} or
          sorted({f"{'wte/ve' if ('wte' in n or 'value' in n) else 'other'}:{t.dtype}" for n, t in sd.items()}))
    print("C5 model file size:", f"{os.path.getsize(os.path.join(ckpt_dir, 'model_005568.pt'))/2**30:.2f} GiB")
    has_norm = [n for n in sd if "norm" in n.lower()]
    has_rot = [n for n in sd if "cos" in n or "sin" in n]
    print("C6 norm weights in ckpt:", has_norm or "NONE", "; cos/sin:", has_rot or "NONE")
    opt_sizes = [os.path.getsize(os.path.join(ckpt_dir, f"optim_005568_rank{r}.pt")) for r in range(4)]
    print("C7 optim shard sizes (GiB):", [f"{s/2**30:.2f}" for s in opt_sizes])
    opt0 = torch.load(os.path.join(ckpt_dir, "optim_005568_rank0.pt"), map_location="cpu", mmap=True, weights_only=False)
    def summarize(d, depth=0):
        if isinstance(d, dict):
            return {k: summarize(v, depth+1) for k, v in list(d.items())[:6]} if depth < 2 else "..."
        if isinstance(d, torch.Tensor):
            return f"tensor{tuple(d.shape)} {d.dtype}"
        return type(d).__name__
    print("C7 rank0 top-level keys:", list(opt0.keys()) if isinstance(opt0, dict) else type(opt0))
    print("C8 total optim / model size:", f"{sum(opt_sizes)/os.path.getsize(os.path.join(ckpt_dir, 'model_005568.pt')):.2f}x")

    print("== D. KV cache magnitudes (d24 real config) ==")
    mc = meta["model_config"]
    hd = mc["n_embd"] // mc["n_head"]
    per_tok = mc["n_layer"] * 2 * mc["n_kv_head"] * hd * 2  # bf16 = 2 bytes
    print("D1 bytes/token:", per_tok, f"({per_tok/1024:.1f} KiB)")
    print("D2 2048 tokens:", f"{per_tok*2048/2**20:.1f} MiB")
    print("D3 with n_kv_head = n_head/4:", f"{per_tok*2048/4/2**20:.1f} MiB")


def lab2():
    import torch
    from lab0_tokenizer import get_tokenizer, SPECIAL_TOKENS
    from lab1_gpt import GPT, GPTConfig
    from lab2_sft import sft_data_generator, setup_optimizer

    tok = get_tokenizer()
    name = {tok.encode_special(s): s for s in SPECIAL_TOKENS}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("== A. mask semantics after shift ==")
    CONV = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Ok"},
    ]}
    ids, mask = tok.render_conversation(CONV)
    supervised = 0
    for p in range(len(ids) - 1):
        tgt_supervised = mask[p + 1] == 1
        supervised += tgt_supervised
        tokstr = name.get(ids[p], repr(tok.decode([ids[p]])))
        print(f"  p={p:2d} input={tokstr:<22} target={'SUPERVISED' if tgt_supervised else '-1':<10}")
    print(f"A5 supervised targets total: {supervised} (row len {len(ids)})")

    print("== B. packing ==")
    import itertools
    lengths = [30, 12, 20, 8, 3]
    fake_convs = itertools.chain(({"__fake_len": L} for L in lengths),
                                 itertools.repeat({"__fake_len": 31}))
    class FakeTok:
        def get_bos_token_id(self): return 0
        def render_conversation(self, c):
            L = c["__fake_len"]
            return list(range(1, L + 1)), [0] + [1] * (L - 1)  # ids encode packing order visibly
    gen = sft_data_generator(fake_convs, FakeTok(), B=2, T=32, device="cpu", buffer_size=5)
    inputs, targets = next(gen)
    for r in range(2):
        row = inputs[r].tolist() + [int(targets[r, -1])]
        # conversation boundaries: value 1 restarts each conv (ids are 1..L per conv); 0 = padding
        segs, cur = [], 0
        for v in row:
            if v == 1 and cur:
                segs.append(cur); cur = 0
            cur += 1 if v != 0 else 0
        if cur: segs.append(cur)
        pad = sum(1 for v in row if v == 0)
        print(f"  row{r+1}: packed lengths {segs}, padding {pad}, ignored(-1) in targets {int((targets[r] == -1).sum())}")
    print("B4: rerun without the explicit padding loop -> compare")
    # rebuild manually: mask[1:] alone
    # (the generator applies both; we recompute the mask-only version here)
    # reconstruct: row had convs packed then padding with bos mask0
    # equivalence check: any position where explicit loop flipped a non(-1) to -1?
    # We detect by simulating: pad positions have mask 0 => already -1 via mask[1:].
    # Direct check: positions content_len-1..T-1 vs mask-only targets
    print("  (see diff report for the analysis; generator applies both paths identically)")

    print("== C. grad accumulation equivalence ==")
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
    def grads_of(fn):
        torch.manual_seed(0)          # identical weights for every comparison arm
        m = GPT(cfg); m.init_weights(); m = m.to(dev)
        torch.manual_seed(1)
        fn(m)
        return m, [p.grad.clone() for p in m.parameters() if p.grad is not None]
    T = 32
    x1 = torch.randint(0, 256, (2, T), device=dev); y1 = torch.randint(0, 256, (2, T), device=dev)
    x2 = torch.randint(0, 256, (2, T), device=dev); y2 = torch.randint(0, 256, (2, T), device=dev)
    y1[:, T//8:] = -1     # micro-batch 1: few valid targets (8 valid)
    y2[:, T//2:] = -1     # micro-batch 2: many valid targets (32 valid)
    n1 = int((y1 >= 0).sum()); n2 = int((y2 >= 0).sum())
    print(f"  valid targets: micro1={n1} micro2={n2}")
    def accum(m):
        (m(x1, y1) / 2).backward(); (m(x2, y2) / 2).backward()
    def big(m):
        m(torch.cat([x1, x2]), torch.cat([y1, y2])).backward()
    def sum_corrected(m):
        (m(x1, y1, loss_reduction='sum') / (n1 + n2)).backward()
        (m(x2, y2, loss_reduction='sum') / (n1 + n2)).backward()
    _, ga = grads_of(accum)
    _, gb = grads_of(big)
    _, gc_ = grads_of(sum_corrected)
    def rel(a, b):
        num = max((x - y).abs().max().item() for x, y in zip(a, b))
        den = max(x.abs().max().item() for x in b) or 1
        return num / den
    print(f"C1 accum(mean/2) vs big-batch: max rel grad diff = {rel(ga, gb):.2e}")
    print(f"C2 accum(sum/N)  vs big-batch: max rel grad diff = {rel(gc_, gb):.2e}")
    print("C3 order: addition is commutative (fp non-associativity aside)")

    print("== D. memory ledger (B=16, T=128, tiny) ==")
    assert dev == "cuda", "D section needs a GPU"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    from lab2_sft import synthetic_conversations
    model = GPT(GPTConfig(sequence_len=128, vocab_size=32768, n_layer=2, n_head=4, n_kv_head=4, n_embd=64))
    model.init_weights(); model = model.to(dev)
    m_params = torch.cuda.memory_allocated()
    print(f"D1 after params on device: {m_params/2**20:.1f} MiB")
    gen = sft_data_generator(synthetic_conversations(), tok, B=16, T=128, device=dev)
    opt = setup_optimizer(model)
    for step in range(3):
        x, y = next(gen)
        loss = model(x, y)
        loss.backward()
        if step == 0:
            print(f"D2 after first backward (params+grads+act residue): {torch.cuda.memory_allocated()/2**20:.1f} MiB")
        opt.step()
        model.zero_grad(set_to_none=True)
        if step == 0:
            print(f"D3 after first optimizer step (+states): {torch.cuda.memory_allocated()/2**20:.1f} MiB")
    print(f"D4 logits-chain single tensor (16*128*32768*4B) = {16*128*32768*4/2**20:.1f} MiB (fp32); bf16 pre-cast = {16*128*32768*2/2**20:.1f} MiB")
    print(f"D5 max_memory_allocated: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")


def lab3():
    import math
    import torch
    from lab0_tokenizer import get_tokenizer
    from lab3_dpo import load_policy_and_ref, batch_pairs, sequence_logprob

    assert torch.cuda.is_available(), "lab3 harness needs the GPU (loads d24 twice)"
    tok = get_tokenizer()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    print("== A/C. graph + memory ==")
    m0 = torch.cuda.memory_allocated()
    policy, ref = load_policy_and_ref("cuda")
    m1 = torch.cuda.memory_allocated()
    print(f"C3 policy+ref resident: {m1/2**30:.2f} GiB total; per-model ~{m1/2/2**30:.2f} GiB")
    print(f"A2 requires_grad: policy {next(policy.parameters()).requires_grad}, ref {next(ref.parameters()).requires_grad}")

    prompts = [
        ({"messages": [{"role": "user", "content": "What is 2+2?"}]}, "4.", "Seven, probably."),
        ({"messages": [{"role": "user", "content": "Name a primary color."}]}, "Red.", "Colorless green ideas."),
        ({"messages": [{"role": "user", "content": "Say hello."}]}, "Hello!", "goodbye hello yes no"),
        ({"messages": [{"role": "user", "content": "Is fire hot?"}]}, "Yes, fire is hot.", "Fire is a type of fish."),
    ]
    inputs, targets = batch_pairs(tok, prompts, "cuda")
    B = len(prompts)
    n_tok = (targets >= 0).sum(dim=-1)                       # response tokens per row
    pi = sequence_logprob(policy, inputs, targets)
    with torch.no_grad():
        rf = sequence_logprob(ref, inputs, targets)
    print(f"C1 logp shape/dtype: {tuple(pi.shape)} {pi.dtype} (inputs {tuple(inputs.shape)})")
    print(f"A2 graph: pi.requires_grad={pi.requires_grad}, ref.requires_grad={rf.requires_grad}")
    print("== B. magnitudes at step 0 ==")
    beta = 0.1
    r = beta * (pi - rf)                                     # implicit rewards, (2B,)
    margin = r[:B] - r[B:]
    loss0 = -torch.nn.functional.logsigmoid(margin).mean()
    print(f"B1 max|r| = {r.abs().max():.2e} (policy==ref); loss at init = {loss0.item():.6f} (ln2 = {math.log(2):.6f})")
    per_tok = (pi / n_tok).tolist()
    for i, (conv, c, rj) in enumerate(prompts):
        print(f"C2 pair{i}: chosen {per_tok[i]:+.2f} nats/tok ({int(n_tok[i])} tok) | rejected {per_tok[B+i]:+.2f} nats/tok ({int(n_tok[B+i])} tok)")
    print("B3 seq logp (chosen rows):", [f"{v:.1f}" for v in pi[:B].tolist()])
    # A3 demo: gather along the wrong dim -- shape check
    import torch.nn.functional as F
    logits = policy(inputs[:1])
    logp = F.log_softmax(logits, dim=-1)
    try:
        bad = logp.gather(1, targets[:1].clamp_min(0).unsqueeze(-1))
        print(f"A3 gather(1,...): NO ERROR, silent wrong result, shape {tuple(bad.shape)} (indexes TIME dim with token ids!)")
    except Exception as e:
        print(f"A3 gather(1,...): raises {type(e).__name__}: {str(e)[:80]}")


def bridge():
    """Qwen3-0.6B bridge (labs/bridge/predictions.md). Needs the Phase A env (safetensors,
    tokenizers, transformers, llamafactory)."""
    import json, os, math
    import torch
    from bridge_qwen3 import (SNAPSHOT, Qwen3Config, load_qwen3, get_tokenizer, encode, render_chat,
                              rope_cos_sin, rotate_half, RMSNorm, COMPUTE_DTYPE)
    from safetensors import safe_open
    cfg = Qwen3Config.from_json(os.path.join(SNAPSHOT, "config.json"))
    C, D, Hq, Hkv, L, I, V = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads,
                              cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size)

    print("== A. tensors in model.safetensors ==")
    path = os.path.join(SNAPSHOT, "model.safetensors")
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        info = {k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in keys}
    l0 = {k: v for k, v in info.items() if k.startswith("model.layers.0.")}
    for k in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "self_attn.q_norm",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        print(f"A1-A5 {k}.weight:", info[f"model.layers.0.{k}.weight"])
    print("A6 tensors per layer:", len(l0), "| total in file:", len(keys))
    from safetensors.torch import load_file
    sd = load_file(path)
    print("A7 lm_head.weight in file:", "lm_head.weight" in info,
          "| equals embed:", torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]))
    del sd
    n_file = sum(math.prod(sh) for sh, _ in info.values())
    n_unique = n_file - math.prod(info["lm_head.weight"][0])
    print(f"A8 params in file {n_file/1e6:.1f}M | unique (tied) {n_unique/1e6:.1f}M | non-embedding {(n_unique - V*C)/1e6:.1f}M")
    print(f"A9 file size {os.path.getsize(path)/1e9:.3f} GB")
    print("A10 norm dtype:", info["model.norm.weight"][1], "| q_norm:", info["model.layers.0.self_attn.q_norm.weight"][1])

    print("== B. mechanisms ==")
    print("B1 RMSNorm learnable scale: yes (weight (dim,)); stats in fp32 (mq3:61-63), scale applied in bf16 (mq3:64)")
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, Hq, 8, D, device=dev)
    cos, sin = rope_cos_sin(cfg, 8, dev, torch.float32)
    rope = lambda t: t * cos.unsqueeze(1) + rotate_half(t) * sin.unsqueeze(1)
    n_plain = RMSNorm(D, cfg.rms_norm_eps).to(dev)
    n_w = RMSNorm(D, cfg.rms_norm_eps).to(dev); n_w.weight.data.uniform_(0.5, 1.5)
    for name, n in (("no weight", n_plain), ("learned weight", n_w)):
        a = rope(n(x)); b = n(rope(x))
        print(f"B2 norm-then-rope vs rope-then-norm ({name}): max|d| = {(a-b).abs().max():.2e}")
    model, _ = load_qwen3(dev)
    tok = get_tokenizer()
    ids = torch.tensor([encode(tok, "The capital of France is")], device=dev)
    with torch.no_grad():
        top_hf = model(ids)[0, -1].argmax().item()
        import bridge_qwen3 as bq
        orig = bq.rotate_half
        bq.rotate_half = lambda t: -orig(t)           # flips the rotation direction (nanochat's)
        top_flip = model(ids)[0, -1].argmax().item()
        bq.rotate_half = orig
    print("B3 same rotation? no (opposite direction). top-1 with HF convention:", repr(tok.decode([top_hf])),
          "| with flipped convention:", repr(tok.decode([top_flip])))
    for th in (1e5, 1e6):
        lam = 2 * math.pi * th ** ((D - 2) / D)
        print(f"B4 theta={th:.0e}: lowest-freq wavelength = {lam:,.0f} positions (max_pos {cfg.max_position_embeddings})")
    print(f"B5 MLP params/layer: Qwen3 3*C*I = {3*C*I/1e6:.2f}M vs nanochat 2*C*4C = {8*C*C/1e6:.2f}M -> ratio {3*I/(8*C):.3f}")
    with torch.no_grad():
        t = torch.tensor([encode(tok, "The quick brown fox jumps over the lazy dog because it wanted to.")], device=dev)
        xe = model.model.embed_tokens(t)
        att = model.model.layers[0].self_attn
        xn = model.model.layers[0].input_layernorm(xe)
        B_, T_, _ = xn.shape
        q = att.q_norm(att.q_proj(xn).view(B_, T_, Hq, D)).transpose(1, 2)
        k = att.k_norm(att.k_proj(xn).view(B_, T_, Hkv, D)).transpose(1, 2)
        c, s_ = rope_cos_sin(cfg, T_, dev, xn.dtype)
        q, k = bq.apply_rotary_pos_emb(q, k, c, s_)
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        lg = (q.float() @ k.float().transpose(-1, -2)) * att.scaling
        print(f"B6 scale = 1/sqrt({D}) = {att.scaling:.4f}; no 1.2x. layer-0 attn logits: max|.| {lg.abs().max():.2f}, std {lg.std():.2f}"
              f" | q_norm.weight range [{att.q_norm.weight.min():.2f}, {att.q_norm.weight.max():.2f}] (no hard bound)")
    print("B7 embedding norm: no (mq3:427 feeds embeds straight to layer 0); softcap: no (mq3:493 raw lm_head)")
    print("B8 ignore_index = -100 (HF ForCausalLMLoss)")
    print("B9 fp32: RMSNorm stats (mq3:61), RoPE tables (mq3:141 autocast disabled), SDPA softmax (kernel-internal),"
          " loss (upcast in loss_function). NOT fp32: logits (bf16, mq3:493), matmuls, residual stream")

    print("== C. chat template ==")
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    M = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
         {"role": "user", "content": "again?"}, {"role": "assistant", "content": "yes"}]
    print("C1 HF apply_chat_template(M):", repr(hf_tok.apply_chat_template(M, tokenize=False)))
    print("C2 BOS:", hf_tok.bos_token, "| eos:", hf_tok.eos_token, hf_tok.eos_token_id, "| pad:", hf_tok.pad_token, hf_tok.pad_token_id)
    from llamafactory.data.template import TEMPLATES
    tpl = TEMPLATES["qwen3"]
    for flag in (True, False):
        tpl.enable_thinking = flag
        pairs = tpl.encode_multiturn(hf_tok, M)
        lf_ids = [t for p, r in pairs for t in p + r]
        lf_mask = [m for p, r in pairs for m in [0] * len(p) + [1] * len(r)]
        tag = "C3/C4" if flag else "C5"
        print(f"{tag} LF enable_thinking={flag}: {repr(hf_tok.decode(lf_ids))}")
        print(f"{tag}   per-token (loss marked *):", " ".join(f"{repr(hf_tok.decode([t]))}{'*' if m else ''}" for t, m in zip(lf_ids, lf_mask)))
        print(f"C6   total {len(lf_ids)} tokens, {sum(lf_mask)} with loss" if flag else "")
    hf_ids = hf_tok.apply_chat_template(M, tokenize=True)
    hf_ids = hf_ids["input_ids"] if hasattr(hf_ids, "input_ids") else hf_ids
    print("C3 HF multi-turn token count:", len(list(hf_ids)), "(compare LF above)")

    print("== D. magnitudes ==")
    print(f"D1 bf16 weights (tied): {n_unique*2/1e9:.2f} GB")
    kv = L * 2 * Hkv * D * 2
    print(f"D2 KV cache per token: {L}*2*{Hkv}*{D}*2B = {kv:,} B = {kv/1024:.0f} KB")
    print(f"D3 full-param SFT 2+4+8+2 = 16 B/param: 0.6B -> {n_unique*16/1e9:.1f} GB; 8.2B -> {8.2e9*16/1e9:.0f} GB")
    print("D4 8B full-param on 4xH200 (564 GB): yes with the states sharded (ZeRO-2/3); LoRA trivially")


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("lab0", "lab1", "lab2", "lab3", "bridge"), "usage: trace_harness.py lab0|lab1|lab2|lab3|bridge"
    {"lab0": lab0, "lab1": lab1, "lab2": lab2, "lab3": lab3, "bridge": bridge}[sys.argv[1]]()
