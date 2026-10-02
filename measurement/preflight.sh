#!/usr/bin/env bash
# preflight.sh - can this machine run the sweep?
# ==============================================
# Checks everything the measurement needs, in about ten seconds, and tells you
# which shaper to use. Run it before anything else:
#
#     chmod +x preflight.sh && sudo ./preflight.sh
#
# Written mainly for WSL2, where the Microsoft kernel ships a reduced set of
# queueing disciplines and HTB is not guaranteed to be present.
set -uo pipefail

PASS=0; FAIL=0
ok()   { echo "  [ ok ] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
note() { echo "         $1"; }

echo
echo "kernel:  $(uname -srm)"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "running under WSL2"
fi
echo

echo "privileges"
[ "$(id -u)" -eq 0 ] && ok "running as root" || bad "not root - re-run with sudo"

echo
echo "tools"
for t in tc ip iperf3 python3; do
  command -v "$t" >/dev/null && ok "$t" || bad "$t missing"
done
command -v ethtool >/dev/null && ok "ethtool" || \
  note "[warn] ethtool missing - offload cannot be disabled, knee will smear"

echo
echo "namespace support"
if ip netns add _pf 2>/dev/null; then
  ok "network namespaces"
  ip netns del _pf 2>/dev/null
else
  bad "cannot create a network namespace"
fi

echo
echo "virtual ethernet"
if ip link add _pfa type veth peer name _pfb 2>/dev/null; then
  ok "veth pairs"
  ip link del _pfa 2>/dev/null
else
  bad "veth not supported by this kernel"
fi

echo
echo "shapers"
SHAPER=""
ip link add _pfa type veth peer name _pfb 2>/dev/null
if tc qdisc add dev _pfa root handle 1: htb default 10 2>/dev/null; then
  ok "htb  (preferred)"
  SHAPER="htb"
  tc qdisc del dev _pfa root 2>/dev/null
else
  note "[warn] htb unavailable - the WSL2 kernel often omits sch_htb"
fi
if tc qdisc add dev _pfa root handle 1: tbf rate 10mbit burst 15k limit 100000 2>/dev/null; then
  ok "tbf  (acceptable fallback)"
  [ -z "$SHAPER" ] && SHAPER="tbf"
  tc qdisc del dev _pfa root 2>/dev/null
else
  note "[warn] tbf unavailable"
fi
ip link del _pfa 2>/dev/null

echo
echo "python"
python3 -c "import scipy" 2>/dev/null && ok "scipy (for the fit)" || \
  note "[warn] scipy missing - needed only by fit_latency.py:  pip install scipy"

echo
echo "-----------------------------------------------------------"
if [ "$FAIL" -eq 0 ] && [ -n "$SHAPER" ]; then
  echo "READY.  Run:   sudo QDISC=$SHAPER ./run_sweep_local.sh 100mbit"
  if [ "$SHAPER" = "tbf" ]; then
    echo
    echo "NOTE: falling back to TBF. That is a token-bucket shaper rather than"
    echo "a hierarchical one, which is arguably a cleaner model of a"
    echo "rate-limited bearer - but describe it as TBF, not HTB, in any"
    echo "write-up."
  fi
else
  echo "NOT READY - $FAIL check(s) failed."
  echo
  echo "On WSL2, missing tools are usually one command away:"
  echo "    sudo apt update && sudo apt install -y iproute2 iperf3 ethtool python3-pip"
  echo "    pip3 install scipy matplotlib"
  echo
  echo "If BOTH htb and tbf are unavailable, this kernel cannot shape traffic."
  echo "Use a real Linux machine, or a Linux VM with a stock distro kernel."
fi
echo
