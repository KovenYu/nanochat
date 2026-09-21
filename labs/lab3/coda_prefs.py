"""Capstone: synthetic Coda trajectory -> DPO pairs + dual-channel KTO samples.

Koven's decisions (2026-09-21):
  DPO   = shared-context fork pairs (v2 vs v2b at the turn fork) + cross-outer pairs
          (v1-in-its-context vs v2b-in-its-context; Bradley-Terry semantics bent, flagged)
  KTO   = dual channel: 'verifier' (inner-loop verdicts) and 'user' (outer-loop reactions),
          emitted separately; v1 is good@verifier but bad@user BY DESIGN
  views = both: View B contexts (true prior messages, fork granularity) and View A
          (re-based on the bare intent)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab2"))
import torch
from coda_views import TRAJ
from lab3_dpo import render_pair, sequence_logprob

def _intent_conv(traj):
    return {"messages": [{"role": "user", "content": traj["intent"] + "\n" + traj["constraints"]}]}

def _ctx_before_outer(traj, k):
    """View B context: messages before outer turn k (fork granularity)."""
    msgs = list(_intent_conv(traj)["messages"])
    for i in range(k):
        msgs.append({"role": "assistant", "content": traj["outer"][i]["attempts"][-1]["code"]})
        msgs.append({"role": "user", "content": traj["outer"][i]["user_feedback"]})
    return {"messages": msgs}

def dpo_pairs(traj):
    """[(conv_chosen, chosen_text, conv_rejected, rejected_text, kind)]"""
    pairs = []
    for k, outer in enumerate(traj["outer"]):          # shared-context fork pairs
        atts = outer["attempts"]
        if len(atts) >= 2:                              # failed attempt(s) before the pass
            ctx = _ctx_before_outer(traj, k)
            pairs.append((ctx, atts[-1]["code"], ctx, atts[0]["code"], "shared"))
    v1 = traj["outer"][0]["attempts"][-1]["code"]       # cross-outer: accepted vs criticized
    v2b = traj["outer"][-1]["attempts"][-1]["code"]
    pairs.append((_ctx_before_outer(traj, len(traj["outer"]) - 1), v2b,
                  _intent_conv(traj), v1, "cross_outer"))
    return pairs

def kto_samples(traj):
    """[{conv, response, channel labels, view}] for both views, both channels."""
    accepted = traj["outer"][-1]["user_feedback"] == "ACCEPT"
    samples = []
    for k, outer in enumerate(traj["outer"]):
        for j, att in enumerate(outer["attempts"]):
            passed = "PASS" in att["verdict"] and "FAIL" not in att["verdict"]
            shown = j == len(outer["attempts"]) - 1     # only the final attempt reaches the user
            if shown:
                user_lab = accepted if k == len(traj["outer"]) - 1 else False
            else:
                user_lab = None                          # user never saw it
            for view, conv in (("B", _ctx_before_outer(traj, k)), ("A", _intent_conv(traj))):
                samples.append({"conv": conv, "response": att["code"], "view": view,
                                "verifier": passed, "user": user_lab})
    return samples

def kto_batch(samples, tok, channel, view, device):
    """Filter one channel+view -> (inputs, targets, labels). Unpaired by construction."""
    keep = [s for s in samples if s["view"] == view and s[channel] is not None]
    rendered = [render_pair(tok, s["conv"], s["response"]) for s in keep]
    L = max(len(ids) for ids, _ in rendered)
    bos = tok.get_bos_token_id()
    rows = [ids + [bos] * (L - len(ids)) for ids, _ in rendered]
    masks = [m + [0] * (L - len(m)) for _, m in rendered]
    batch = torch.tensor(rows, dtype=torch.long)
    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].clone().to(device)
    mask_t = torch.tensor(masks, dtype=torch.int8)[:, 1:].to(device)
    targets[mask_t == 0] = -1
    labels = torch.tensor([s[channel] for s in keep], dtype=torch.bool, device=device)
    return inputs, targets, labels

def dpo_batch(pairs, tok, device):
    """Per-side contexts (cross-outer pairs have different convs per side)."""
    rendered = ([render_pair(tok, c, t) for c, t, _, _, _ in pairs] +
                [render_pair(tok, cr, tr) for _, _, cr, tr, _ in pairs])
    L = max(len(ids) for ids, _ in rendered)
    bos = tok.get_bos_token_id()
    rows = [ids + [bos] * (L - len(ids)) for ids, _ in rendered]
    masks = [m + [0] * (L - len(m)) for _, m in rendered]
    batch = torch.tensor(rows, dtype=torch.long)
    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].clone().to(device)
    mask_t = torch.tensor(masks, dtype=torch.int8)[:, 1:].to(device)
    targets[mask_t == 0] = -1
    return inputs, targets
