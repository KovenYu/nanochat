"""Milestone ceremony: chat with the model before/after the mini SFT run.

Two modes per prompt:
  raw  : bos + text completion (a base model's native habitat)
  chat : render_for_completion template (OOD for base; in-distribution after SFT)
Generation: greedy, stop on <|assistant_end|> or --max-tokens cap (base never stops).

Usage: python labs/lab2/milestone_chat.py [--ckpt /path/model_final.pt] [--max-tokens 200]
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini"))
import torch
from lab0_tokenizer import get_tokenizer
from lab2_sft import load_d24

PROMPTS = [
    "Why is the sky blue?",
    "Write a haiku about mountains.",
    "What is 17 + 25?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="mini-saved state dict; default: frozen d24 base")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    tok = get_tokenizer()
    model, _ = load_d24("cuda")
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    aend = tok.encode_special("<|assistant_end|>")
    bos = tok.get_bos_token_id()

    def generate(ids):
        out, hit_cap = [], True
        for t in model.generate(ids, max_tokens=args.max_tokens, temperature=0.0):
            if t == aend:
                hit_cap = False
                break
            out.append(t)
        return tok.decode(out), hit_cap

    label = args.ckpt or "d24-baseline (frozen)"
    print(f"=== model: {label} ===")
    for p in PROMPTS:
        raw_ids = tok.encode(p, prepend=bos)
        text, cap = generate(raw_ids)
        print(f"\n--- RAW  | {p}\n{text}{'  [HIT CAP]' if cap else '  [stopped]'}")
        chat_ids = tok.render_for_completion(
            {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": ""}]})
        text, cap = generate(chat_ids)
        print(f"--- CHAT | {p}\n{text}{'  [HIT CAP: never emitted assistant_end]' if cap else '  [stopped via assistant_end]'}")

if __name__ == "__main__":
    main()
