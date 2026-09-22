"""Bridge acceptance: the mini IS Qwen3-0.6B, and its chat rendering IS what Phase 1 trains on.

Run in the Phase A env:  source /svl/u/koven/sft-phase-a/env.sh && python labs/bridge/accept_test.py

  T1 load: strict state_dict load, lm_head tied to embed_tokens (asserted inside load_qwen3)
  T2 fp32 alignment vs HF Qwen3ForCausalLM (both fp32, both SDPA): max |dlogits| < 1e-3 on a
     ~100-token text; argmax identical at every position
  T3 bf16 alignment (the dtype Phase 1 runs in): argmax agreement >= 98%, per-token NLL
     mean within 1e-2 of HF's; max |dlogits| reported (bf16 kernels differ in summation order)
  T4 template: render_chat(M, True) ids == LLaMA-Factory qwen3 encode_multiturn ids, and its
     mask == LF's prompt/response split; render_chat(M, False) == LF with enable_thinking=False;
     for a single-turn conversation, render_chat ids == HF apply_chat_template ids
"""
import sys
sys.path.insert(0, "labs/mini")
import torch
import torch.nn.functional as F
from bridge_qwen3 import SNAPSHOT, load_qwen3, get_tokenizer, encode, render_chat

TEXT = ("Rotary position embeddings encode absolute positions with a rotation matrix and "
        "naturally incorporate explicit relative position dependency in self-attention. "
        "The bracket mounts a 40mm fan to a 2020 aluminium rail; hole spacing is 32mm, "
        "thickness at least 3mm. def box(w, h, t): return Solid.make_box(w, h, t)")
M = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
     {"role": "user", "content": "again?"}, {"role": "assistant", "content": "yes"}]

def main():
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    fails = []
    def ck(name, cond, detail=""):
        if not cond: fails.append(f"{name}: {detail}")
    dev = "cuda"
    tok = get_tokenizer()
    ids = torch.tensor([encode(tok, TEXT)], device=dev)                 # (1, T)
    T = ids.shape[1]

    mini, _ = load_qwen3(dev)                                            # T1 inside
    hf = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    nll = lambda l: F.cross_entropy(l[0, :-1], ids[0, 1:]).item()
    with torch.no_grad():
        # T3 first, in the dtype from_pretrained gave us. Order matters: HF keeps RoPE's
        # inv_freq as a float32 BUFFER, and Module.to(bfloat16) would round it (fp32 -> bf16
        # -> fp32 is lossy), shifting every rotation angle: NLL moves by ~1e-2 and the test
        # blames the mini. Upcasting with .float() afterwards is lossless.
        lm = mini(ids).float(); lh = hf(ids).logits.float()             # (1, T, V)
        d16 = (lm - lh).abs().max().item()
        agree = (lm.argmax(-1) == lh.argmax(-1)).float().mean().item()
        ck("T3 bf16 argmax agree", agree >= 0.98, f"{agree:.3f}")
        ck("T3 bf16 nll", abs(nll(lm) - nll(lh)) < 1e-2, f"{nll(lm):.4f} vs {nll(lh):.4f}")
        # T2: fp32
        mini.float(); hf.float()
        lm = mini(ids).float(); lh = hf(ids).logits.float()
        d32 = (lm - lh).abs().max().item()
        ck("T2 fp32 max|d|", d32 < 1e-3, f"{d32:.2e}")
        ck("T2 fp32 argmax", torch.equal(lm.argmax(-1), lh.argmax(-1)), "argmax differs")

    # T4: templates
    hf_tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    from llamafactory.data.template import TEMPLATES
    tpl = TEMPLATES["qwen3"]
    for flag in (True, False):
        tpl.enable_thinking = flag
        pairs = tpl.encode_multiturn(hf_tok, M)
        lf_ids = [t for p, r in pairs for t in p + r]
        lf_mask = [m for p, r in pairs for m in [0] * len(p) + [1] * len(r)]
        r_ids, r_mask = render_chat(tok, M, enable_thinking=flag)
        ck(f"T4 LF ids (thinking={flag})", r_ids == lf_ids, f"\n  mini {r_ids}\n  LF   {lf_ids}")
        ck(f"T4 LF mask (thinking={flag})", r_mask == lf_mask, f"\n  mini {r_mask}\n  LF   {lf_mask}")
    single = M[:2]
    hf_ids = hf_tok.apply_chat_template(single, tokenize=True)
    if hasattr(hf_ids, "input_ids"): hf_ids = hf_ids["input_ids"]
    r_ids, _ = render_chat(tok, single, enable_thinking=True)
    ck("T4 HF single-turn ids", r_ids == list(hf_ids), f"\n  mini {r_ids}\n  HF   {list(hf_ids)}")
    multi_hf = hf_tok.apply_chat_template(M, tokenize=True)
    if hasattr(multi_hf, "input_ids"): multi_hf = multi_hf["input_ids"]
    r_multi, _ = render_chat(tok, M, enable_thinking=True)

    if fails:
        print("FAIL"); [print(" ", f) for f in fails]; return 1
    print(f"PASS: T={T} fp32 max|d|={d32:.1e}, bf16 max|d|={d16:.2f} argmax agree={agree:.3f}; "
          f"templates match LF (both flags) and HF (single-turn); "
          f"multi-turn HF differs from LF by {len(r_multi) - len(list(multi_hf))} tokens (by design)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
