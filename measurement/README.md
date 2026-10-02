# Delay against offered load on a shaped Linux link

The paper's latency model, `L(ρ) = L₀ + A·max(0, ρ − ρ_knee)^κ`, is **assumed**,
not fitted. Its results need the scope condition `ρ_knee < 1/h` from a real
bearer, not those four constants (the paper's knee bound: `h* ≤ 1/ρ_knee`).
This harness measures a real Linux queue to see which side of that condition it
falls on.

## What this measures, and what it does not

It measures **round-trip delay through a Linux traffic shaper** (an HTB class
feeding a 1000-packet pfifo, or TBF with a comparable byte limit where the
kernel lacks HTB) as a function of offered-load ratio `ρ`, using a light UDP
probe stream alongside a constant-rate UDP bulk flow. The packets, queue and
network stack are real.

It is **not** a radio bearer. There is no HARQ, no scheduler grant loop, no
fading and no contention between UEs, which are the mechanisms that make radio
links queue well before saturation. Describe it as a "shaped virtual link",
naming the qdisc that was used, never as a 6G or radio measurement and never
as a calibration of the paper's constants.

## Result

On this link delay stays flat until saturation and then jumps. The knee sits
near `ρ = 1`, far above the `ρ_knee < 1/1.35 ≈ 0.74` the pathology needs. The
knee bound therefore predicts that a 1.35 floor cannot be sub-threshold on
this bearer. The link is a measured case outside the scope condition, not a
fit.

Fitting the power-law model to these points is not meaningful. The curve is a
step, not a convex rise past a knee, and the fitted curve from
`fit_latency.py` leaves strongly structured residuals. Do not quote its `h*` as
a measured value.

## Data

All three sweeps shaped the link to 100 Mbit/s, used 1200-byte UDP payloads and
a 100 Hz probe, and recorded zero loss at every point. `ρ` is the nominal ratio
set in the script (but see the caveats below).

| File | Where | Points | p95 RTT |
|---|---|---|---|
| `data/run0_sandbox.csv` | cloud sandbox VM | 16 `ρ` × 3 repeats, 0.30–0.98 | noisy; drifts from ~1 ms to 4–5 ms by `ρ = 0.94`, 5.0–6.7 ms at 0.96, 9.4–9.6 ms at 0.98 |
| `data/run1_coarse.csv` | laptop, WSL2, veth + netns | 16 `ρ` × 5, 0.30–0.98 | 0.10–0.21 ms up to `ρ = 0.92`; 0.22–1.74 ms at 0.96; 9.13–9.17 ms at 0.98 |
| `data/run2_knee.csv` | same laptop, dense near the jump | 13 `ρ` × 5, 0.88–0.99 | ≤ 0.24 ms up to 0.92; unstable at 0.94–0.955 (0.13–3.88 ms at 0.955); 8.90–9.20 ms from 0.965 to 0.99 |

`run1` and `run2` are the controlled measurements. The sandbox shares its host
with other tenants and is the least reliable of the three.

## Interpreting the numbers: two caveats

1. **Nominal `ρ` counts payload, not wire bytes.** `iperf3 -b` sets the UDP
   *payload* rate. With `-l 1200` each datagram reaches the qdisc as about
   1242 bytes (UDP 8 + IP 20 + Ethernet 14), and the shaper meters those bytes. A
   nominal `ρ` therefore loads the link at about 1.035 × `ρ`, and nominal
   `ρ = 1200/1242 = 0.966` is 100% of the shaped rate. The jump in `run2`,
   between nominal 0.960 and 0.965, is saturation itself. On the wire scale
   the knee is at ≈ 1.0, which only strengthens the conclusion above.
2. **The 9.1 ms plateau is probably the sender, not the link.** Past saturation
   delay stops rising at about 9.1 ms with zero loss, although a full shaper
   queue (1000 packets) would hold about 100 ms and drop. 9.1 ms at 100 Mbit/s is
   about 92 packets of 1242 bytes, consistent with the bulk sender's default
   socket send buffer (~212 KB at ~2.3 KB of kernel memory per packet) filling
   and blocking `iperf3`, because the shaped queue sits on the sending host.
   This has not been verified. Repeating one point above saturation with a
   larger buffer (`iperf3 -w`) would show whether the plateau moves. Either
   way, the plateau height is not a property of the bearer.

## Running it

One Linux machine (veth pair + network namespace; works under WSL2 if the
kernel has `sch_htb` or `sch_tbf`):

```bash
sudo ./preflight.sh                       # checks tools, netns, veth, shapers
sudo ./run_sweep_local.sh 100mbit         # default 16-point grid
sudo ./run_sweep_local.sh 100mbit --rhos "0.94 0.95 0.96 0.97"   # custom grid
python3 fit_latency.py latency_measurements.csv --sla 5.0
```

Pass the grid with `--rhos`, not `sudo RHOS=...`: `sudo` may silently drop an
environment variable containing spaces. The script prints the grid and where it
came from before it starts.

Two machines on a direct cable (shaping on the sender's egress):

```bash
# receiver
iperf3 -s &
python3 latency_probe.py serve
# sender
sudo ./run_sweep.sh <receiver-ip> <iface> 100mbit
```

`latency_probe.py` reports RTT, not one-way delay, so no clock synchronisation
is needed; the unshaped reverse path adds a near-constant term.
