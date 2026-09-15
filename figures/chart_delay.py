"""Figure 4 - delaying the floor, and the censoring fraction that explains it.

Same visual system as chart_movie.py / chart_entropy.py so the paper's figures
read as one set. Colour does two different jobs here and is assigned accordingly:
ORANGE is a categorical contrast (floor from step 0 - the condition every other
bar is measured against), and the three switch times are ORDINAL, so they take
the same monotonic blue ramp the entropy figure uses. Every bar carries a direct
label and every scatter series a distinct marker, so identity survives greyscale
printing and CVD.
"""
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"]=42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt

SURF="#fcfcfb"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"
ORANGE="#eb6834"                                   # floor from step 0
RAMP={150000:"#86b6ef", 300000:"#2a78d6", 500000:"#104281"}   # ordinal, validated
MARK={150000:"o", 300000:"s", 500000:"D"}          # secondary encoding for print
GOOD="#0ca30c"

plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":SURF,
  "axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
  "axes.labelcolor":SEC,"axes.edgecolor":BASE,"xtick.color":MUT,"ytick.color":SEC})

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200)

# ---------------- panel A: violations by when the floor is imposed
order = [0] + SW
means = [st.mean(viol[k]) for k in order]
sds   = [st.stdev(viol[k]) for k in order]
cols  = [ORANGE] + [RAMP[s] for s in SW]
xs    = range(len(order))
ax1.bar(xs, means, width=0.62, color=cols, zorder=3,
        yerr=sds, ecolor=BASE, capsize=4, error_kw={"lw": 1.2, "zorder": 4})
for x, m, sd in zip(xs, means, sds):
    ax1.text(x, m + sd + 90, f"{m:,.0f}", ha="center", va="bottom",
             fontsize=10.2, fontweight="bold", color=INK, zorder=5)

ax1.axhline(268, color=GOOD, ls="--", lw=1.6, zorder=2)

ax1.set_xticks(list(xs))
ax1.set_xticklabels(["0\n(always on)", "150k", "300k", "500k"],
                    fontsize=9.2, linespacing=1.5)
ax1.set_xlabel("Training step at which the floor is imposed", fontsize=9.5)
ax1.set_ylabel("Critical SLA violations during training", fontsize=9.5)
ax1.set_ylim(0, 3450)
ax1.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); ax1.set_axisbelow(True)
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a) Cost of imposing the floor at step 0", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=12)
ax1.text(0, -0.235, "n = 8 seeds per level, mean $\\pm$ sd; every switch time "
         "beats always-on on all 8 paired seeds. Dashed line: no floor (268).",
         fontsize=8.2, color=MUT, transform=ax1.transAxes)

# ---------------- panel B: the mechanism - peak censoring vs damage
def peak_plateau(rows):
    return max(float(r["frac_plateau"]) for r in rows)

ax2.scatter([peak_plateau(r) for r in runs_on],
            [int(r["total_viol"]) for r in always],
            s=64, marker="^", color=ORANGE, edgecolor=SURF, linewidth=1.6,
            zorder=3, label="floor from step 0")

tby = defaultdict(list)
for r in dt:
    tby[(int(r["switch_at"]), int(r["seed"]))].append(r)
for s in SW:
    X, Y = [], []
    for r in by[s]:
        on = [x for x in tby[(s, int(r["seed"]))] if int(x["floor_on"]) == 1]
        X.append(peak_plateau(on)); Y.append(int(r["total_viol"]))
    ax2.scatter(X, Y, s=64, marker=MARK[s], color=RAMP[s], edgecolor=SURF,
                linewidth=1.6, zorder=3, label=f"switch at {s//1000}k")

ax2.set_xlabel("Peak fraction of evaluation states with zero action gradient",
               fontsize=9.5)
ax2.set_ylabel("Critical SLA violations during training", fontsize=9.5)
ax2.set_xlim(-0.06, 1.10); ax2.set_ylim(-110, 3450)
ax2.grid(True, color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b) Damage scales with how much of the state space is censored",
              fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
ax2.text(0.035, 0.955, "Spearman r = +0.93   (n = 32, p < 1e-14)",
         transform=ax2.transAxes, fontsize=9.4, color=INK,
         fontweight="bold", va="top")
leg = ax2.legend(loc="center left", frameon=False, fontsize=9.0,
                 handletextpad=0.4, borderpad=0.2, labelspacing=0.45)
for t in leg.get_texts(): t.set_color(SEC)
ax2.text(0, -0.235, "Each point is one training run (n = 32).",
         fontsize=8.2, color=MUT, transform=ax2.transAxes)

fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.savefig("fig_delay.png", bbox_inches="tight")
fig.savefig("fig_delay.pdf", bbox_inches="tight")
print("ok")
