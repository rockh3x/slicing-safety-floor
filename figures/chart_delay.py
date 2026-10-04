"""Figure 4 - delaying the floor, and the censoring fraction that explains it.

Same visual system as chart_movie.py / chart_entropy.py so the paper's figures
read as one set. Colour does two different jobs here and is assigned accordingly:
ORANGE is a categorical contrast (floor from step 0 - the condition every other
bar is measured against), and the three switch times are ORDINAL, so they take
the same monotonic blue ramp the entropy figure uses. Every bar carries a direct
label and every scatter series a distinct marker, so identity survives greyscale
printing and CVD.

Drawn at final print size (174 mm) with panel letters only; the explanatory
notes live in the LaTeX caption, as JNSM requires.
"""
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"]=42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt

SURF="#ffffff"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"
ORANGE="#eb6834"                                   # floor from step 0
RAMP={150000:"#86b6ef", 300000:"#2a78d6", 500000:"#104281"}   # ordinal, validated
MARK={150000:"o", 300000:"s", 500000:"D"}          # secondary encoding for print
GOOD="#0ca30c"

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
  "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
  "text.color":INK,"axes.labelcolor":SEC,"axes.edgecolor":BASE,
  "xtick.color":SEC,"ytick.color":SEC,"axes.labelsize":8.5,
  "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8})

D = "results/"
ds = list(csv.DictReader(open(D+"delay_sweep_real.csv")))
dt = list(csv.DictReader(open(D+"delay_traj_real.csv")))
fs = [r for r in csv.DictReader(open(D+"fix_sweep_real.csv"))
      if float(r["override"]) == 0 and float(r["ent_coef"]) == 0]
ft = [r for r in csv.DictReader(open(D+"fix_traj_real.csv"))
      if r["condition"] == "floor_only"]
always = sorted(fs, key=lambda r: int(r["seed"]))
# fix_traj.csv rows arrive in worker-completion order, not seed order, because the
# sweep runs under a multiprocessing pool. Chunking by position therefore yields
# the eight runs in an ARBITRARY seed order. Key each chunk by the seed it
# actually contains and look it up, rather than zipping position against the
# seed-sorted summary rows -- doing the latter pairs each run's plateau occupancy
# with another run's violation count.
_chunks = [ft[i * 20:(i + 1) * 20] for i in range(8)]
by_seed = {int(c[0]["seed"]): c for c in _chunks}
runs_on = [by_seed[int(r["seed"])] for r in always]

SW = [150000, 300000, 500000]
by = defaultdict(list)
for r in ds:
    by[int(r["switch_at"])].append(r)

viol = {0: [int(r["total_viol"]) for r in always]}
for s in SW:
    viol[s] = [int(r["total_viol"]) for r in by[s]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 2.8), dpi=300)

# ---------------- panel A: violations by when the floor is imposed
order = [0] + SW
means = [st.mean(viol[k]) for k in order]
sds   = [st.stdev(viol[k]) for k in order]
cols  = [ORANGE] + [RAMP[s] for s in SW]
xs    = range(len(order))
ax1.bar(xs, means, width=0.62, color=cols, zorder=3,
        yerr=sds, ecolor=BASE, capsize=3, error_kw={"lw": 0.9, "zorder": 4})
for x, m, sd in zip(xs, means, sds):
    ax1.text(x, m + sd + 90, f"{m:,.0f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color=INK, zorder=5)

ax1.axhline(268, color=GOOD, ls="--", lw=1.1, zorder=2, label="no floor (268)")

ax1.set_xticks(list(xs))
ax1.set_xticklabels(["0\n(always on)", "150k", "300k", "500k"], linespacing=1.3)
ax1.set_xlabel("Step at which the floor is imposed")
ax1.set_ylabel("Training-time critical violations")
ax1.set_ylim(0, 3450)
ax1.grid(True, axis="y", color=GRID, lw=0.6, zorder=0); ax1.set_axisbelow(True)
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
leg = ax1.legend(loc="upper right", frameon=False, handlelength=1.8)
for t in leg.get_texts(): t.set_color(SEC)

# ---------------- panel B: the mechanism - peak censoring vs damage
def peak_plateau(rows):
    return max(float(r["frac_plateau"]) for r in rows)

ax2.scatter([peak_plateau(r) for r in runs_on],
            [int(r["total_viol"]) for r in always],
            s=26, marker="^", color=ORANGE, edgecolor=SURF, linewidth=0.8,
            zorder=3, label="floor from step 0")

tby = defaultdict(list)
for r in dt:
    tby[(int(r["switch_at"]), int(r["seed"]))].append(r)
for s in SW:
    X, Y = [], []
    for r in by[s]:
        on = [x for x in tby[(s, int(r["seed"]))] if int(x["floor_on"]) == 1]
        X.append(peak_plateau(on)); Y.append(int(r["total_viol"]))
    ax2.scatter(X, Y, s=26, marker=MARK[s], color=RAMP[s], edgecolor=SURF,
                linewidth=0.8, zorder=3, label=f"switch at {s//1000}k")

ax2.set_xlabel("Peak share of states with zero action gradient")
ax2.set_ylabel("Training-time critical violations")
ax2.set_xlim(-0.06, 1.10); ax2.set_ylim(-110, 3450)
ax2.grid(True, color=GRID, lw=0.6, zorder=0); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
ax2.text(0.035, 0.955, "Spearman $r=+0.93$ ($n=32$, $p<10^{-14}$)",
         transform=ax2.transAxes, fontsize=8, color=INK, va="top")
leg = ax2.legend(loc="center left", frameon=False,
                 handletextpad=0.4, borderpad=0.2, labelspacing=0.4)
for t in leg.get_texts(): t.set_color(SEC)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig_delay.png", bbox_inches="tight")
fig.savefig("fig_delay.pdf", bbox_inches="tight")
print("ok")
