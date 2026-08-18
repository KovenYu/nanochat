"""Plot the base-pretraining loss and LR-multiplier curves from a speedrun log."""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"^step (\d+)/(\d+) \(.*?\) \| loss: ([\d.]+) \| lrm: ([\d.]+)")

parser = argparse.ArgumentParser()
parser.add_argument("--log", default="runs/speedrun.log")
parser.add_argument("--out", default="runs/speedrun_curves.png")
args = parser.parse_args()

steps, losses, lrms = [], [], []
with open(args.log, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = STEP_RE.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))
            lrms.append(float(m.group(4)))

if not steps:
    raise SystemExit(f"no base-training step lines found in {args.log}")

LOSS_COLOR = "#3b6fd4"
LR_COLOR = "#c05a1f"
INK = "#1c1c1c"
MUTED = "#6b6b6b"

fig, (ax_loss, ax_lr) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True, height_ratios=[2.4, 1], dpi=160
)

ax_loss.plot(steps, losses, color=LOSS_COLOR, lw=1.6)
ax_loss.set_ylabel("train loss (smoothed)", color=INK)
ax_loss.set_title(
    f"nanochat speedrun — base pretraining ({steps[-1] + 1} steps, "
    f"final loss {losses[-1]:.3f})",
    color=INK, fontsize=12, pad=10,
)
ax_loss.annotate(
    f"{losses[-1]:.3f}",
    xy=(steps[-1], losses[-1]),
    xytext=(-6, 10), textcoords="offset points",
    ha="right", color=LOSS_COLOR, fontsize=9,
)

ax_lr.plot(steps, lrms, color=LR_COLOR, lw=1.6)
ax_lr.set_ylabel("LR multiplier", color=INK)
ax_lr.set_xlabel("step", color=INK)
ax_lr.set_ylim(0, 1.08)

for ax in (ax_loss, ax_lr):
    ax.grid(True, color="#e2e2e2", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")
    ax.tick_params(colors=MUTED, labelsize=9)

fig.tight_layout()
fig.savefig(args.out, facecolor="white")
print(f"wrote {args.out}  ({len(steps)} points)")
