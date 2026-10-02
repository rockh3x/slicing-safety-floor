"""
latency_probe.py - UDP round-trip probe with percentile reporting
=================================================================
One small, evenly-paced probe stream measured while a separate bulk flow
loads the shaped link. The probe is deliberately tiny and low-rate so that it
measures the queue rather than contributing to it.

Run the responder on the receiver:      python latency_probe.py serve
Run the prober on the sender:           python latency_probe.py probe --host <ip> --seconds 30

Reports min / mean / p50 / p95 / p99 / max in milliseconds, plus the loss rate.
The paper's latency model is a function of queue occupancy, so what matters
here is the percentile at a known offered-load ratio - see run_sweep.sh.

WHY RTT AND NOT ONE-WAY DELAY
  One-way delay needs clock synchronisation between the two hosts to be
  meaningful, and an unsynchronised one-way number is worse than useless.
  RTT over a link where only one direction is shaped is dominated by the
  shaped direction, and the reverse path adds a near-constant term that the
  fitted L0 absorbs. State this in any write-up: the quantity fitted is RTT,
  not one-way latency.
"""

import argparse
import socket
import statistics as st
import struct
import sys
import time

PACKET = 64                      # bytes; small enough to be negligible load
MAGIC = b"SLPB"


def serve(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    sys.stderr.write(f"responder listening on udp/{port}\n")
    while True:
        data, addr = s.recvfrom(2048)
        if data[:4] == MAGIC:
            s.sendto(data, addr)          # echo verbatim; sender holds the clock


def probe(host, port, seconds, rate_hz):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    interval = 1.0 / rate_hz
    rtts, sent, lost = [], 0, 0
    deadline = time.perf_counter() + seconds
    seq = 0

    while time.perf_counter() < deadline:
        due = time.perf_counter() + interval
        payload = MAGIC + struct.pack("!Qd", seq, 0.0)
        payload += b"\0" * (PACKET - len(payload))
        t0 = time.perf_counter()
        try:
            s.sendto(payload, (host, port))
            sent += 1
            while True:
                data, _ = s.recvfrom(2048)
                if data[:4] == MAGIC and struct.unpack("!Q", data[4:12])[0] == seq:
                    rtts.append((time.perf_counter() - t0) * 1000.0)
                    break
        except socket.timeout:
            lost += 1
        seq += 1
        slack = due - time.perf_counter()
        if slack > 0:
            time.sleep(slack)

    if not rtts:
        print("ERROR: no responses - check the responder and any firewall",
              file=sys.stderr)
        sys.exit(1)

    rtts.sort()
    pct = lambda p: rtts[min(len(rtts) - 1, int(round(p / 100.0 * len(rtts))))]
    return {
        "n": len(rtts), "sent": sent, "lost": lost,
        "loss_pct": 100.0 * lost / max(1, sent),
        "min": rtts[0], "mean": st.mean(rtts), "p50": pct(50),
        "p95": pct(95), "p99": pct(99), "max": rtts[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sv = sub.add_parser("serve"); sv.add_argument("--port", type=int, default=9999)
    pr = sub.add_parser("probe")
    pr.add_argument("--host", required=True)
    pr.add_argument("--port", type=int, default=9999)
    pr.add_argument("--seconds", type=float, default=30.0)
    pr.add_argument("--rate-hz", type=float, default=100.0)
    pr.add_argument("--csv", action="store_true",
                    help="print one comma-separated line instead of a table")
    a = ap.parse_args()

    if a.mode == "serve":
        serve(a.port)
    else:
        r = probe(a.host, a.port, a.seconds, a.rate_hz)
        if a.csv:
            print(",".join(f"{r[k]:.4f}" for k in
                           ("min", "mean", "p50", "p95", "p99", "max", "loss_pct")))
        else:
            for k in ("n", "sent", "lost", "loss_pct", "min", "mean",
                      "p50", "p95", "p99", "max"):
                print(f"{k:>9}: {r[k]:.4f}")


if __name__ == "__main__":
    main()
