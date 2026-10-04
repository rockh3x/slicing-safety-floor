"""Figure 5 - the same floor, enforced by projection and by reservation.

Same visual system as chart_movie.py. Left: allocation ratio of the critical
slice over training at h = 1.35 under projection (orange) and reservation
(blue), eight seeds each. Right: training-time violations by floor multiplier
for both enforcements, against the no-floor reference.
Drawn at final print size (174 mm) with panel letters only; the explanatory
notes live in the LaTeX caption, as JNSM requires.
"""
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SURF="#ffffff"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"; BLUE="#2a78d6"; ORANGE="#eb6834"; GOOD="#0ca30c"

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
  "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
  "text.color":INK,"axes.labelcolor":SEC,"axes.edgecolor":BASE,
  "xtick.color":SEC,"ytick.color":SEC,"axes.labelsize":8.5,
  "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8})

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 2.8), dpi=300,
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------- (a) trajectories at h = 1.35
for src, col, label in ((old, ORANGE, "Projection (clip), $h$ = 1.35"),
                        (res, BLUE, "Reservation, $h$ = 1.35")):
    runs = [src[(1.35, s)] for s in range(8)]
    for q in runs:
        ax1.plot([t for t, _, _ in q], [a for _, a, _ in q], color=col,
                 lw=0.8, alpha=0.40, zorder=2)
    ts = [t for t, _, _ in runs[0]]
    mean = [st.mean(q[i][1] for q in runs) for i in range(len(ts))]
    ax1.plot(ts, mean, color=col, lw=1.8, zorder=4, label=label)

ax1.axhline(1.3765, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=3)
ax1.text(0.02, 1.3765 + 0.012, "$h^*$ = 1.3765", va="bottom", ha="left",
         fontsize=8, color=INK)
ax1.text(0.5, 1.5743 + 0.015, "all 8 seeds identical: 1.574", va="bottom",
         ha="center", fontsize=8, color=BLUE)
ax1.set_xlim(0, 1.03); ax1.set_ylim(1.30, 1.75)
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticklabels(["0", "250k", "500k", "750k", "1M"])
ax1.set_xlabel("Training steps")
ax1.set_ylabel("Allocation / demand, critical slice")
ax1.grid(True, color=GRID, lw=0.6, zorder=0); ax1.set_axisbelow(True)
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
leg = ax1.legend(loc="upper left", frameon=False)
for t in leg.get_texts(): t.set_color(SEC)

# ---------------- (b) training-time violations by h
def totals(src, h):
    return [sum(c for _, _, c in src[(h, s)]) for s in range(8)]

H_CLIP = [(1.2, clip), (1.35, old), (1.38, clip)]
H_RES = [1.0, 1.2, 1.35, 1.38]
xc = [h for h, _ in H_CLIP]
mc = [st.mean(totals(src, h)) for h, src in H_CLIP]
ax2.plot(xc, mc, color=ORANGE, lw=1.5, marker="^", ms=6, zorder=4,
         label="Projection (clip)")
mr = [st.mean(totals(res, h)) for h in H_RES]
ax2.plot(H_RES, mr, color=BLUE, lw=1.5, marker="o", ms=4.5, zorder=5,
         markerfacecolor=SURF, markeredgewidth=1.2, label="Reservation")
nf = st.mean(totals(old, 0.0))
ax2.axhline(nf, color=GOOD, ls="--", lw=1.1, zorder=2)
ax2.text(1.0, nf + 60, f"no floor ({nf:.0f})", fontsize=8, color=GOOD,
         va="bottom")
ax2.axvline(1.3765, color=INK, lw=0.8, ls=(0, (5, 3)), zorder=1)
ax2.text(1.3765 + 0.005, 2950, "$h^*$", ha="left", va="center",
         fontsize=8, color=INK)
ax2.set_xlabel("Floor multiplier $h$")
ax2.set_ylabel("Training-time critical violations")
ax2.set_xlim(0.97, 1.42); ax2.set_ylim(-120, 3100)
ax2.grid(True, color=GRID, lw=0.6, zorder=0); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
leg = ax2.legend(loc="center left", frameon=False)
for t in leg.get_texts(): t.set_color(SEC)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig_reserve.png", bbox_inches="tight")
fig.savefig("fig_reserve.pdf", bbox_inches="tight")
print("ok")
