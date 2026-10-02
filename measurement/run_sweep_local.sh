#!/usr/bin/env bash
# run_sweep_local.sh - the same measurement on ONE Linux machine
# ==============================================================
# Builds a virtual Ethernet pair into a network namespace, shapes one end with
# HTB, and measures across it. No second laptop, no cable, no firewall.
#
#     host                                  namespace "slfns"
#     veth0 10.99.0.1  <--- HTB shaped --->  veth1 10.99.0.2
#                                            iperf3 -s
#                                            latency_probe.py serve
#
# WHAT THIS IS AND IS NOT
#   The queue being measured is a real Linux HTB class feeding a real pfifo,
#   carrying real packets through the real network stack. That is the thing the
#   latency model is meant to describe, and it is genuinely measured.
#   What is NOT present is a physical link: propagation delay is ~0, so the
#   fitted L0 reflects stack and scheduling overhead rather than a real bearer's
#   baseline. Say "HTB-shaped virtual link" in any write-up, never "testbed".
#
# USAGE
#     sudo ./run_sweep_local.sh 100mbit
#     python3 fit_latency.py latency_measurements.csv --sla 5.0
set -euo pipefail

RATE="${1:-100mbit}"
NS=slfns
DWELL="${DWELL:-30}"
SETTLE="${SETTLE:-5}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-latency_measurements.csv}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Occupancy points.
#
# These can be set three ways, in increasing order of reliability:
#   ./run_sweep_local.sh 100mbit --rhos "0.94 0.96 0.98"   <-- always works
#   RHOS="0.94 0.96 0.98" ./run_sweep_local.sh 100mbit     <-- works from a shell
#   sudo RHOS="0.94 0.96 0.98" ./run_sweep_local.sh ...    <-- MAY BE SILENTLY DROPPED
#
# sudo resets the environment by default, and depending on the sudoers policy a
# value containing spaces can be discarded while single-token variables survive.
# The failure is silent: the sweep runs the default grid and nothing warns you.
# Prefer --rhos, or drop sudo if you are already root.
# Capture the environment value BEFORE `read` overwrites RHOS with the grid;
# testing RHOS afterwards would always find it set and mislabel the source.
RHOS_ENV="${RHOS:-}"
read -r -a RHOS <<< "${RHOS_ENV:-0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.84 0.88 0.90 0.92 0.94 0.96 0.98}"
RHOS_SOURCE="default grid"
[ -n "$RHOS_ENV" ] && RHOS_SOURCE="RHOS environment variable"

# Flags, parsed after the positional rate argument.
shift $(( $# > 0 ? 1 : 0 ))
while [ $# -gt 0 ]; do
  case "$1" in
    --rhos)    read -r -a RHOS <<< "$2"; RHOS_SOURCE="--rhos flag"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --dwell)   DWELL="$2";   shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
command -v iperf3 >/dev/null || { echo "install iperf3"; exit 1; }
command -v tc     >/dev/null || { echo "install iproute2"; exit 1; }

cleanup() {
  ip netns pids "$NS" 2>/dev/null | xargs -r kill 2>/dev/null || true
  ip netns del "$NS" 2>/dev/null || true
  ip link del veth0 2>/dev/null || true
}
trap cleanup EXIT
cleanup

echo "grid (${#RHOS[@]} points, from ${RHOS_SOURCE}): ${RHOS[*]}"
echo "repeats ${REPEATS}, dwell ${DWELL}s, output ${OUT}"
echo
echo "building namespace and veth pair"
ip netns add "$NS"
ip link add veth0 type veth peer name veth1
ip link set veth1 netns "$NS"
ip addr add 10.99.0.1/24 dev veth0
ip link set veth0 up
ip netns exec "$NS" ip addr add 10.99.0.2/24 dev veth1
ip netns exec "$NS" ip link set veth1 up
ip netns exec "$NS" ip link set lo up

# Segmentation offload hands the NIC 64 kB super-packets, which makes HTB
# account in lumps and smears the knee. Turn it off on both ends or the
# measurement is of the offload engine, not the shaper.
for d in veth0; do ethtool -K $d tso off gso off gro off 2>/dev/null || true; done
ip netns exec "$NS" ethtool -K veth1 tso off gso off gro off 2>/dev/null || true

QDISC="${QDISC:-htb}"
echo "shaping veth0 egress to $RATE using $QDISC"
if [ "$QDISC" = "htb" ]; then
  tc qdisc add dev veth0 root handle 1: htb default 10
  # quantum pinned to one MTU: the default (rate/r2q) is ~1.25 MB at
  # 100 Mbit, which makes HTB dequeue in large bursts and coarsens the
  # very knee we are trying to locate.
  tc class add dev veth0 parent 1: classid 1:10 htb rate "$RATE" ceil "$RATE" quantum 1514
  # A pfifo of stated depth. The queue discipline is part of what is being
  # measured, so it is set explicitly rather than left to the default.
  tc qdisc add dev veth0 parent 1:10 handle 10: pfifo limit 1000
elif [ "$QDISC" = "tbf" ]; then
  # Fallback for kernels without sch_htb (common on WSL2). TBF is a token
  # bucket with its own FIFO; `limit` is that FIFO in BYTES, so 1000 packets
  # of 1500 B keeps the queue depth comparable to the htb branch above.
  tc qdisc add dev veth0 root handle 1: tbf rate "$RATE" burst 15k limit 1500000
else
  echo "QDISC must be htb or tbf (got '$QDISC')"; exit 1
fi

echo "starting responders inside the namespace"
ip netns exec "$NS" iperf3 -s -D
ip netns exec "$NS" python3 "$HERE/latency_probe.py" serve &
sleep 2

RATE_NUM=$(echo "$RATE" | sed 's/[^0-9]//g')
RATE_UNIT=$(echo "$RATE" | sed 's/[0-9]//g')

echo "rho,target_mbit,repeat,min_ms,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,loss_pct" > "$OUT"

for RHO in "${RHOS[@]}"; do
  LOAD=$(awk -v r="$RATE_NUM" -v p="$RHO" 'BEGIN{printf "%.3f", r*p}')
  for REP in $(seq 1 "$REPEATS"); do
    printf "  rho=%-5s load=%8s%s  repeat %d/%d\n" "$RHO" "$LOAD" "$RATE_UNIT" "$REP" "$REPEATS"
    iperf3 -c 10.99.0.2 -u -b "${LOAD}${RATE_UNIT}" \
           -t $((DWELL + SETTLE + 2)) -l 1200 >/dev/null 2>&1 &
    LOADPID=$!
    sleep "$SETTLE"
    LINE=$(python3 "$HERE/latency_probe.py" probe --host 10.99.0.2 \
             --seconds "$DWELL" --rate-hz 100 --csv)
    echo "$RHO,$LOAD,$REP,$LINE" >> "$OUT"
    wait "$LOADPID" 2>/dev/null || true
    sleep 2
  done
done

echo
echo "wrote $OUT"
echo "next:  python3 fit_latency.py $OUT --sla 5.0"
