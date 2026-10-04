"""Figure 3 - entropy: the dose-response, and the dispersion mechanism.

The two panels deliberately come from DIFFERENT conditions, because on the real
Milan trace neither condition can carry both jobs:

  (a) needs a condition where the violation count is not saturated. The
      floor-only baseline sits at 97.3% of the 2880 maximum with five of eight
      seeds pinned exactly at it, so a dose-response cannot be resolved there.
      With the override cost applied the metric has room, and it responds.
  (b) needs a condition where terminal log-std actually varies. That is the
      floor-only condition: once the override cost is present the policy
      converges tightly at every coefficient (log-std -2.39 to -1.85) and the
      relationship vanishes, which is itself the point being made.

Same visual system as the other figures. Ordinal blue ramp for the entropy
levels, direct labels on every bar, distinct markers per level in (b).
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
BAR="#2a78d6"; GOOD="#0ca30c"
RAMP={0.001:"#86b6ef", 0.003:"#2a78d6", 0.01:"#1c5aa3", 0.03:"#104281"}
MARK={0.001:"o", 0.003:"s", 0.01:"^", 0.03:"D"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
  "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
  "text.color":INK,"axes.labelcolor":SEC,"axes.edgecolor":BASE,
  "xtick.color":SEC,"ytick.color":SEC,"axes.labelsize":8.5,
  "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8})

D="results/"
# ---- panel A source: entropy on top of the override cost (has headroom)
ef=defaultdict(dict)
for x in csv.DictReader(open(D+"ent_fix.csv")):
    ef[float(x["ent_coef"])][int(x["seed"])]=x
LEV=[0.0,0.001,0.003,0.01,0.03]
mean=[st.mean([int(ef[e][s]["total_viol"]) for s in range(8)]) for e in LEV]
sd  =[st.stdev([int(ef[e][s]["total_viol"]) for s in range(8)]) for e in LEV]

# ---- panel B source: floor-only sweep (log-std varies)
es=defaultdict(dict)
for x in csv.DictReader(open(D+"ent_sweep.csv")):
    es[float(x["ent_coef"])][int(x["seed"])]=x

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.85,2.8),dpi=300)

# ---------------- (a) dose-response with the override cost applied
x=range(len(LEV))
ax1.bar(x, mean, width=0.62, color=BAR, zorder=3, yerr=sd, ecolor=BASE,
        capsize=3, error_kw={"lw":0.9,"zorder":4})
for xi,m,s in zip(x,mean,sd):
    ax1.text(xi, m+s+8, f"{m:.0f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color=INK, zorder=5)
ax1.axhline(268, color=GOOD, ls="--", lw=1.1, zorder=2, label="no floor (268)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(["0\n(override\nonly)","0.001","0.003","0.01","0.03"],
                    linespacing=1.2)
ax1.set_xlabel("PPO entropy coefficient")
ax1.set_ylabel("Training-time critical violations")
ax1.set_ylim(0,700)
ax1.grid(True, axis="y", color=GRID, lw=0.6, zorder=0); ax1.set_axisbelow(True)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
leg = ax1.legend(loc="upper left", frameon=False, handlelength=1.8)
for t in leg.get_texts(): t.set_color(SEC)

# ---------------- (b) what entropy does to the policy, in both conditions
ORANGE="#eb6834"
for lbl, src, col, mk in (("floor only", es, ORANGE, "^"),
                          ("floor + override", ef, BAR, "o")):
    xs, ys = [], []
    for i, e in enumerate(LEV):
        for sd_ in range(8):
            xs.append(i + (0.10 if col == BAR else -0.10))
            ys.append(float(src[e][sd_]["logstd_last"]))
    ax2.scatter(xs, ys, s=20, marker=mk, color=col, edgecolor=SURF,
                linewidth=0.7, zorder=3, alpha=0.85, label=lbl)
    m = [st.mean([float(src[e][sd_]["logstd_last"]) for sd_ in range(8)])
         for e in LEV]
    ax2.plot(range(len(LEV)), m, color=col, lw=1.5, zorder=4)

ax2.set_xticks(range(len(LEV)))
ax2.set_xticklabels(["0", "0.001", "0.003", "0.01", "0.03"])
ax2.set_xlabel("PPO entropy coefficient")
ax2.set_ylabel("Policy log-std at end of training")
ax2.set_xlim(-0.5, 4.5); ax2.set_ylim(-3.4, 1.7)
ax2.grid(True, color=GRID, lw=0.6, zorder=0); ax2.set_axisbelow(True)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b)", fontsize=9, fontweight="bold", color=INK, loc="left", pad=4)
leg = ax2.legend(loc="upper left", frameon=False,
                 handletextpad=0.4, borderpad=0.2, labelspacing=0.4)
for t in leg.get_texts(): t.set_color(SEC)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig_entropy.png", bbox_inches="tight")
fig.savefig("fig_entropy.pdf", bbox_inches="tight")
print("ok")
