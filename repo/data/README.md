# Obtaining the traffic data

The experiments are driven by one day of the Telecom Italia "Big Data Challenge"
mobile traffic dataset for Milan.

**The dataset is not redistributed here.** It is published by its original
providers and should be obtained from them:

- Harvard Dataverse, DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)
- Dataset: *Telecommunications — SMS, Call, Internet — MI*
- File required: the day file for **1 November 2013**

Place it in this directory as `milan.txt`. The loader
(`slicing/milan_traffic.py`) also accepts `milan.txt.txt` and any
`milan*.txt*`, because Windows appends a second extension when "hide known
file extensions" is enabled — a detail that silently invalidated an earlier
round of this project's results and is now handled explicitly.

The file is roughly 171 MB and 2.46 M lines. On first use the loader parses it
once and writes `milan_cache.npz` beside it, keyed on the source file's size and
modification time; later runs read the cache.

## What the loader does with it

Columns are `square_id, time_interval_ms, country_code, sms_in, sms_out,
call_in, call_out, internet`. Activity is summed across all 6259 grid squares to
give a city-wide series of 144 ten-minute intervals, then:

| Slice   | Source                                   |
|---------|------------------------------------------|
| `video` | internet activity, normalised, peak 96 u |
| `volte` | call activity, normalised, peak 20 u     |
| `urllc` | **synthesised** — see below              |

`urllc` demand is **not** derived from the dataset. No public URLLC trace exists
for 2013, so it is synthesised as a base of 0.45 ± 0.10 with bursts to 1.0 on 6%
of steps, scaled to a peak of 14 units. This is stated in the paper and is a
limitation, not an oversight.

The peak values are chosen so that aggregate demand exceeds the 100-unit pool
over part of the day: on this trace 70 of 144 steps are in overload.

## If the file is missing

`MilanTraffic` raises `RuntimeError`. It does **not** fall back to a synthetic
stand-in. An earlier version did, which is how a full round of experiments came
to be run on synthetic data without anyone noticing. A stand-in now has to be
asked for explicitly (`allow_synthetic=True`, or `SLICING_ALLOW_SYNTHETIC=1`)
and must never be used for a reported result.
