"""Figure 5 - the same floor, enforced by projection and by reservation.

Same visual system as chart_movie.py. Left: allocation ratio of the critical
slice over training at h = 1.35 under projection (orange) and reservation
(blue), eight seeds each. Right: training-time violations by floor multiplier
for both enforcements, against the no-floor reference.
"""
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SURF="#fcfcfb"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"; BLUE="#2a78d6"; ORANGE="#eb6834"; GOOD="#0ca30c"

plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":SURF,
  "axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
  "axes.labelcolor":SEC,"axes.edgecolor":BASE,"xtick.color":MUT,"ytick.color":SEC})

D = "results/"


def traj(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[(float(r["headroom"]), int(r["seed"]))].append(
            (int(r["timestep"]) / 1e6, float(r["alloc_ratio"]), int(r["crit"])))
    return {k: sorted(v) for k, v in by.items()}


old = traj(D + "plateau_traj_real.csv")      # no floor (0.0) and clip 1.35
clip = traj(D + "clip_traj.csv")             # clip 1.2, 1.38
res = traj(D + "reserve_traj.csv")           # reserve 1.0, 1.2, 1.35, 1.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200,
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------- (a) trajectories at h = 1.35
for src, col, label in ((old, ORANGE, "Projection (clip), h = 1.35"),
                        (res, BLUE, "Reservation, h = 1.35")):
    runs = [src[(1.35, s)] for s in range(8)]
    for q in runs:
        ax1.plot([t for t, _, _ in q], [a for _, a, _ in q], color=col,
                 lw=1.2, alpha=0.40, zorder=2)
    ts = [t for t, _, _ in runs[0]]
    mean = [st.mean(q[i][1] for q in runs) for i in range(len(ts))]
    ax1.plot(ts, mean, color=col, lw=2.6, zorder=4, label=label)

ax1.axhline(1.3765, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=3)
ax1.text(0.02, 1.3765 + 0.012, "$h^*$ = 1.3765", va="bottom", ha="left",
         fontsize=8.6, color=INK)
ax1.text(0.5, 1.5743 + 0.015, "all 8 seeds identical: 1.574", va="bottom",
         ha="center", fontsize=8.6, color=BLUE)
ax1.set_xlim(0, 1.03); ax1.set_ylim(1.30, 1.75)
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticklabels(["0", "250k", "500k", "750k", "1M"], fontsize=9)
ax1.set_xlabel("Training steps", fontsize=9.5)
ax1.set_ylabel("Allocation / demand, critical slice", fontsize=9.5)
ax1.grid(True, color=GRID, lw=0.8, zorder=0); ax1.set_axisbelow(True)
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a) Same floor, h = 1.35, two enforcements", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=12)
leg = ax1.legend(loc="upper left", frameon=False, fontsize=9.0)
for t in leg.get_texts(): t.set_color(SEC)
ax1.text(0, -0.15, "Thin lines: seeds (n = 8); bold: mean. Below the dashed\n"
         "line every evaluation step violates the 5 ms target.",
         fontsize=8.2, color=MUT, transform=ax1.transAxes, va="top")

# ---------------- (b) training-time violations by h
def totals(src, h):
    return [sum(c for _, _, c in src[(h, s)]) for s in range(8)]

H_CLIP = [(1.2, clip), (1.35, old), (1.38, clip)]
H_RES = [1.0, 1.2, 1.35, 1.38]
xc = [h for h, _ in H_CLIP]
mc = [st.mean(totals(src, h)) for h, src in H_CLIP]
ax2.plot(xc, mc, color=ORANGE, lw=2.2, marker="^", ms=9, zorder=4,
         label="Projection (clip)")
mr = [st.mean(totals(res, h)) for h in H_RES]
ax2.plot(H_RES, mr, color=BLUE, lw=2.2, marker="o", ms=6.5, zorder=5,
         markerfacecolor=SURF, markeredgewidth=1.8, label="Reservation")
nf = st.mean(totals(old, 0.0))
ax2.axhline(nf, color=GOOD, ls="--", lw=1.6, zorder=2)
ax2.text(1.0, nf + 60, f"no floor ({nf:.0f})", fontsize=8.6, color=GOOD,
         va="bottom")
ax2.axvline(1.3765, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=1)
ax2.text(1.3765 + 0.005, 2950, "$h^*$", ha="left", va="center",
         fontsize=9, color=INK)
ax2.set_xlabel("Floor multiplier h", fontsize=9.5)
ax2.set_ylabel("Critical SLA violations during training", fontsize=9.5)
ax2.set_xlim(0.97, 1.42); ax2.set_ylim(-120, 3100)
ax2.grid(True, color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b) Violations by floor multiplier", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=12)
leg = ax2.legend(loc="center left", frameon=False, fontsize=9.0)
for t in leg.get_texts(): t.set_color(SEC)
ax2.text(0, -0.15, "Mean of 8 seeds per point. At h = 1.38 both\n"
         "enforcements score 0 on every seed.",
         fontsize=8.2, color=MUT, transform=ax2.transAxes, va="top")

fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig("fig_reserve.png", bbox_inches="tight")
fig.savefig("fig_reserve.pdf", bbox_inches="tight")
print("ok")
