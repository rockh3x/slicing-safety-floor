import csv, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"]=42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt

SURF="#fcfcfb"; INK="#0b0b0b"; SEC="#52514e"; MUT="#898781"
GRID="#e1e0d9"; BASE="#c3c2b7"; BLUE="#2a78d6"; ORANGE="#eb6834"

plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":SURF,
  "axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
  "axes.labelcolor":SEC,"axes.edgecolor":BASE,"xtick.color":MUT,"ytick.color":SEC})

R=[]
_rows=list(csv.DictReader(open("results/plateau_traj_real.csv")))
for i in range(16):
    for r in _rows[i*20:(i+1)*20]:
        R.append({"h":float(r["headroom"]),"s":int(r["seed"]),
                  "t":int(r["timestep"])/1e6,"a":float(r["alloc_ratio"])})

fig, ax = plt.subplots(figsize=(8.6,4.4), dpi=200)
for h,c in ((0.0,BLUE),(1.35,ORANGE)):
    for s in range(8):
        q=sorted([r for r in R if r["h"]==h and r["s"]==s], key=lambda x:x["t"])
        ax.plot([r["t"] for r in q],[r["a"] for r in q],color=c,lw=1.3,alpha=0.42,zorder=2)
    # mean line
    ts=sorted({r["t"] for r in R if r["h"]==h})
    mean=[sum(r["a"] for r in R if r["h"]==h and r["t"]==t)/8 for t in ts]
    ax.plot(ts,mean,color=c,lw=2.6,zorder=4,
            label=("No safety floor" if h==0 else "Safety floor at 1.35"))

ax.axhline(1.3765, color=INK, lw=1.4, ls=(0,(5,3)), zorder=3)
ax.text(1.012, 1.3765, "$h^*$ = 1.3765\nSLA feasibility bound", va="center", ha="left",
        fontsize=8.6, color=INK, linespacing=1.4)

ax.annotate("All shielded seeds remain at exactly 1.35 for\n750k–1M steps; below $h^*$, so every\nevaluation step violates the SLA",
            xy=(0.50,1.352), xytext=(0.335,3.30), fontsize=9, color=SEC,
            ha="left", va="bottom", linespacing=1.45,
            arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.1, shrinkA=6, shrinkB=3))

ax.set_xlim(0,1.30); ax.set_ylim(1.02,4.8)
ax.set_xticks([0,0.25,0.5,0.75,1.0])
ax.set_xticklabels(["0","250k","500k","750k","1M"],fontsize=9)
ax.set_xlabel("Training steps", fontsize=9.5)
ax.set_ylabel("Allocation / demand, critical slice", fontsize=9.5)
ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
ax.tick_params(length=0)
ax.set_title("Allocation ratio of the critical slice during training", fontsize=13, fontweight="bold",
             color=INK, loc="left", pad=12)
leg=ax.legend(loc="upper right", frameon=False, fontsize=9.5, handlelength=1.6)
for t in leg.get_texts(): t.set_color(SEC)
ax.text(0,-0.155,"Thin lines: individual seeds (n = 8 per condition). Bold: condition mean. Values below the dashed line violate the 5 ms latency target.",
        fontsize=8.6,color=MUT,transform=ax.transAxes)
fig.tight_layout()
fig.savefig("fig_trap.png", bbox_inches="tight")
fig.savefig("fig_trap.pdf", bbox_inches="tight")
print("ok")
