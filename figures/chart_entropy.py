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
BAR="#2a78d6"; GOOD="#0ca30c"
RAMP={0.001:"#86b6ef", 0.003:"#2a78d6", 0.01:"#1c5aa3", 0.03:"#104281"}
MARK={0.001:"o", 0.003:"s", 0.01:"^", 0.03:"D"}

plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":SURF,
  "axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
  "axes.labelcolor":SEC,"axes.edgecolor":BASE,"xtick.color":MUT,"ytick.color":SEC})

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

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.6,4.5),dpi=200)

# ---------------- (a) dose-response with the override cost applied
x=range(len(LEV))
ax1.bar(x, mean, width=0.62, color=BAR, zorder=3, yerr=sd, ecolor=BASE,
        capsize=4, error_kw={"lw":1.2,"zorder":4})
for xi,m,s in zip(x,mean,sd):
    ax1.text(xi, m+s+8, f"{m:.0f}", ha="center", va="bottom",
             fontsize=10.2, fontweight="bold", color=INK, zorder=5)
ax1.axhline(268, color=GOOD, ls="--", lw=1.6, zorder=2)
ax1.set_xticks(list(x))
ax1.set_xticklabels(["0\n(override only)","0.001","0.003","0.01","0.03"],
                    fontsize=9.2, linespacing=1.5)
ax1.set_xlabel("PPO entropy coefficient", fontsize=9.5)
ax1.set_ylabel("Critical SLA violations during training", fontsize=9.5)
ax1.set_ylim(0,700)
ax1.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); ax1.set_axisbelow(True)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
ax1.spines["left"].set_color(BASE); ax1.spines["bottom"].set_color(BASE)
ax1.tick_params(length=0)
ax1.set_title("(a) Entropy still costs, with the override cost applied",
              fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
ax1.text(0,-0.235,"n = 8 seeds per level, mean $\\pm$ sd; worse on 8/8 paired "
         "seeds at $\\beta \\geq 0.01$. Dashed line: no safety floor (268).",
         fontsize=8.2, color=MUT, transform=ax1.transAxes)

# ---------------- (b) what entropy does to the policy, in both conditions
ORANGE="#eb6834"
for lbl, src, col, mk in (("floor only", es, ORANGE, "^"),
                          ("floor + override", ef, BAR, "o")):
    xs, ys = [], []
    for i, e in enumerate(LEV):
        for sd_ in range(8):
            xs.append(i + (0.10 if col == BAR else -0.10))
            ys.append(float(src[e][sd_]["logstd_last"]))
    ax2.scatter(xs, ys, s=46, marker=mk, color=col, edgecolor=SURF,
                linewidth=1.3, zorder=3, alpha=0.85, label=lbl)
    m = [st.mean([float(src[e][sd_]["logstd_last"]) for sd_ in range(8)])
         for e in LEV]
    ax2.plot(range(len(LEV)), m, color=col, lw=2.2, zorder=4)

ax2.set_xticks(range(len(LEV)))
ax2.set_xticklabels(["0", "0.001", "0.003", "0.01", "0.03"], fontsize=9.2)
ax2.set_xlabel("PPO entropy coefficient", fontsize=9.5)
ax2.set_ylabel("Policy log-std at end of training", fontsize=9.5)
ax2.set_xlim(-0.5, 4.5); ax2.set_ylim(-3.4, 1.7)
ax2.grid(True, color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_color(BASE); ax2.spines["bottom"].set_color(BASE)
ax2.tick_params(length=0)
ax2.set_title("(b) Entropy widens the policy only where the gradient is censored",
              fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
leg = ax2.legend(loc="center left", frameon=False, fontsize=9.0,
                 handletextpad=0.4, borderpad=0.2, labelspacing=0.45)
for t in leg.get_texts(): t.set_color(SEC)
ax2.text(0, -0.235, "Lines join level means. With the override cost the policy "
         "converges tightly at every level.",
         fontsize=8.2, color=MUT, transform=ax2.transAxes)

fig.tight_layout(rect=[0,0.055,1,1])
fig.savefig("fig_entropy.png", bbox_inches="tight")
fig.savefig("fig_entropy.pdf", bbox_inches="tight")
print("ok")
