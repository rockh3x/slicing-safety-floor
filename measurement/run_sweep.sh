#!/usr/bin/env bash
# run_sweep.sh - measure RTT against offered-load ratio on an HTB-shaped link
# =========================================================================
# Produces the data the latency model is fitted to: for each target occupancy
# rho = offered_rate / shaped_rate, hold a bulk load at that rate and record
# RTT percentiles from a light probe stream.
#
# TOPOLOGY
#   sender (this host, root needed for tc)  --ethernet-->  receiver
#   Shaping is applied to the sender's egress only. Use a direct cable or an
#   otherwise idle switch; a busy Wi-Fi link measures the Wi-Fi, not the shaper.
#
# ON THE RECEIVER, first:
#   iperf3 -s &
#   python3 latency_probe.py serve
#
# ON THE SENDER:
#   sudo ./run_sweep.sh 192.168.1.50 eth0 100mbit
#
# OUTPUT
#   latency_measurements.csv  ->  feed to fit_latency.py
set -euo pipefail

PEER="${1:?usage: run_sweep.sh <peer-ip> <iface> <rate e.g. 100mbit>}"
IFACE="${2:?}"
RATE="${3:?}"
DWELL="${DWELL:-30}"          # seconds of measurement per point
SETTLE="${SETTLE:-5}"         # seconds to let the queue reach steady state
OUT="${OUT:-latency_measurements.csv}"
REPEATS="${REPEATS:-3}"       # independent repeats per rho, for error bars

# Occupancy points. Dense near and above the expected knee, because that is the
# region the whole paper turns on; sparse below it, where the curve is flat.
RHOS=(0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.84 0.88 0.90 0.92 0.94 0.96 0.98)

RATE_NUM=$(echo "$RATE" | sed 's/[^0-9]//g')
RATE_UNIT=$(echo "$RATE" | sed 's/[0-9]//g')

cleanup() { tc qdisc del dev "$IFACE" root 2>/dev/null || true; }
trap cleanup EXIT

command -v iperf3 >/dev/null || { echo "iperf3 not installed"; exit 1; }
command -v tc     >/dev/null || { echo "iproute2 (tc) not installed"; exit 1; }

echo "shaping $IFACE egress to $RATE"
cleanup
tc qdisc add dev "$IFACE" root handle 1: htb default 10
tc class add dev "$IFACE" parent 1: classid 1:10 htb rate "$RATE" ceil "$RATE"
# A pfifo of known depth. The queue discipline is part of what you are
# measuring, so it must be recorded in the write-up, not left to the default.
tc qdisc add dev "$IFACE" parent 1:10 handle 10: pfifo limit 1000

echo "rho,target_mbit,repeat,min_ms,mean_ms,p50_ms,p95_ms,p99_ms,max_ms,loss_pct" > "$OUT"

for RHO in "${RHOS[@]}"; do
  LOAD=$(awk -v r="$RATE_NUM" -v p="$RHO" 'BEGIN{printf "%.3f", r*p}')
  for REP in $(seq 1 "$REPEATS"); do
    echo "  rho=$RHO  load=${LOAD}${RATE_UNIT}  repeat $REP/$REPEATS"

    # Bulk load. UDP at a fixed bitrate so offered load is what we set it to;
    # TCP would back off and the occupancy would no longer be the independent
    # variable.
    iperf3 -c "$PEER" -u -b "${LOAD}${RATE_UNIT}" \
           -t $((DWELL + SETTLE + 2)) -l 1200 >/dev/null 2>&1 &
    LOADPID=$!
    sleep "$SETTLE"

    LINE=$(python3 latency_probe.py probe --host "$PEER" \
             --seconds "$DWELL" --rate-hz 100 --csv)
    echo "$RHO,$LOAD,$REP,$LINE" >> "$OUT"

    wait "$LOADPID" 2>/dev/null || true
    sleep 2                       # let the queue drain before the next point
  done
done

echo
echo "wrote $OUT"
echo "next:  python3 fit_latency.py $OUT --sla 5.0"
