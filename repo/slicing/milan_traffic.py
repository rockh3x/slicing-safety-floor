"""
milan_traffic.py
----------------
Traffic driver in the style of CLARA: demand derived from the Telecom Italia
Milan dataset ("Telecommunications - SMS, Call, Internet - MI", Big Data
Challenge 2014, DOI 10.7910/DVN/EGZHFV on Harvard Dataverse).

WHY THIS FILE WAS REWRITTEN (2026-09-10)
  The previous version looked for exactly `data/milan.txt` and, when it did not
  find it, fell back to a synthetic diurnal curve with only a printed notice.
  The downloaded file on disk was named `milan.txt.txt` - Windows appends the
  extension a second time when "hide known file extensions" is on - so the
  check failed and EVERY experiment in the project silently ran on the
  synthetic stand-in while 171 MB of real data sat unused beside it. The notice
  scrolled past in runs that print hundreds of lines.

  Two changes prevent a recurrence:
    1. The real file is located by SEARCH, not by one exact name.
    2. Falling back to synthetic now RAISES. A stand-in has to be asked for
       explicitly - allow_synthetic=True, or SLICING_ALLOW_SYNTHETIC=1 - so it
       can never again be the silent default in a result that gets published.

  A parsed-curve cache is also written beside the data file. The raw file is
  2.46 M lines; parsing it once per worker made an 8-way parallel sweep pay
  that cost eight times over.

MAPPING ACTIVITY -> SLICE DEMAND (the standard trick in papers using this
dataset, since 2013 Milan obviously carried no URLLC traffic):
  internet activity -> video slice demand
  call activity     -> VoLTE slice demand
  URLLC             -> synthesized: low base + occasional bursts
                       (state this openly in any write-up; CLARA-era works do
                       the same kind of mapping because no public URLLC trace
                       exists)
"""

import os
import glob
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA_DIR, "milan_cache.npz")

# peak demand targets in resource units (capacity is 100): the evening peak
# deliberately exceeds capacity (~130 total) so the overload trade-off appears -
# a benchmark every baseline passes teaches nothing
PEAKS = {"video": 96.0, "volte": 20.0, "urllc": 14.0}


def find_data_file(data_dir=DATA_DIR):
    """Locate the Milan day file, tolerating a doubled or altered extension.

    Returns the path, or None. Ordered from most to least specific so an exact
    `milan.txt` always wins if it is present.
    """
    exact = os.path.join(data_dir, "milan.txt")
    if os.path.exists(exact):
        return exact
    doubled = os.path.join(data_dir, "milan.txt.txt")
    if os.path.exists(doubled):
        return doubled
    hits = sorted(glob.glob(os.path.join(data_dir, "milan*.txt*")))
    hits = [h for h in hits if not h.endswith(".npz")]
    return hits[0] if hits else None


def _load_real(path):
    """Aggregate a Milan day file into 144 ten-minute (video, volte) points.

    Columns are square_id, time_interval_ms, country_code, sms_in, sms_out,
    call_in, call_out, internet. Fields are frequently EMPTY rather than zero,
    which `float(x or 0)` handles. Activity is summed across all grid squares,
    giving a city-wide series.
    """
    sums = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            try:
                t = int(parts[1])
                call = float(parts[5] or 0) + float(parts[6] or 0)
                internet = float(parts[7] or 0)
            except ValueError:
                continue
            s = sums.setdefault(t, [0.0, 0.0])
            s[0] += internet
            s[1] += call
    times = sorted(sums)
    if not times:
        raise RuntimeError(f"{path} parsed to zero usable rows")
    internet = np.array([sums[t][0] for t in times])
    calls = np.array([sums[t][1] for t in times])
    return internet / internet.max(), calls / calls.max(), times


def _load_real_cached(path):
    """_load_real with an npz cache keyed on the source file's size and mtime."""
    stat = os.stat(path)
    key = np.array([stat.st_size, int(stat.st_mtime)], dtype=np.int64)
    if os.path.exists(CACHE):
        try:
            z = np.load(CACHE)
            if np.array_equal(z["key"], key):
                return z["video"], z["volte"], int(z["n"])
        except Exception:
            pass                                   # a bad cache is not fatal
    v, c, times = _load_real(path)
    try:
        np.savez(CACHE, key=key, video=v, volte=c, n=len(times))
    except OSError:
        pass                                       # read-only tree: just skip
    return v, c, len(times)


def _synthetic_day(steps=144, seed=0):
    """Diurnal curves shaped like the Milan data. LABELED SYNTHETIC.

    Retained for smoke tests and for environments with no dataset access. It is
    NOT reachable by accident any more - see MilanTraffic.__init__.
    """
    rng = np.random.default_rng(seed)
    h = np.linspace(0, 24, steps, endpoint=False)
    # internet/video: low at night, climb after 17:00, peak ~21:30
    video = 0.18 + 0.82 * np.exp(-0.5 * ((h - 21.5) / 2.6) ** 2) \
        + 0.25 * np.exp(-0.5 * ((h - 13.0) / 3.5) ** 2)
    # calls/VoLTE: business-hours plateau + small evening bump
    volte = 0.15 + 0.70 * np.exp(-0.5 * ((h - 11.0) / 2.8) ** 2) \
        + 0.45 * np.exp(-0.5 * ((h - 18.5) / 2.2) ** 2)
    video = np.clip(video / video.max(), 0, 1)
    volte = np.clip(volte / volte.max(), 0, 1)
    return video, volte


class MilanTraffic:
    """Drop-in traffic model for SlicingEnv: .demand() and .scenario.

    Raises RuntimeError if the Milan file is absent, unless a synthetic
    stand-in is explicitly requested. That is deliberate: a missing dataset
    should stop an experiment, not quietly change what it measures.
    """

    def __init__(self, slices, noise=0.08, seed=0, allow_synthetic=None,
                 quiet=False):
        self.slice_names = [s.name for s in slices]
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.t = 0

        if allow_synthetic is None:
            allow_synthetic = os.environ.get("SLICING_ALLOW_SYNTHETIC") == "1"

        path = find_data_file()
        if path is not None:
            v, c, n = _load_real_cached(path)
            self.real = True
            self.source = (f"REAL (Telecom Italia Milan, {n} intervals "
                           f"aggregated city-wide, {os.path.basename(path)})")
        elif allow_synthetic:
            v, c = _synthetic_day(seed=seed)
            self.real = False
            self.source = "SYNTHETIC diurnal stand-in - NOT the Milan dataset"
        else:
            raise RuntimeError(
                "Milan traffic file not found in "
                f"{os.path.abspath(DATA_DIR)}. Looked for milan.txt, "
                "milan.txt.txt and milan*.txt*. Download the day file from "
                "Harvard Dataverse (DOI 10.7910/DVN/EGZHFV) and place it "
                "there. To run on the synthetic stand-in instead - which must "
                "never be used for a reported result - pass "
                "allow_synthetic=True or set SLICING_ALLOW_SYNTHETIC=1.")

        self.video_curve, self.volte_curve = v, c
        self.steps = len(v)
        # URLLC: low base with occasional bursts (openly synthesized)
        base = 0.45 + 0.10 * self.rng.standard_normal(self.steps)
        bursts = (self.rng.random(self.steps) < 0.06) * self.rng.uniform(0.5, 1.0, self.steps)
        self.urllc_curve = np.clip(base + bursts, 0.2, 1.0)
        self.scenario = "milan-day"
        if not quiet:
            print(f"[MilanTraffic] source: {self.source}  ({self.steps} steps/day)")

    def demand(self):
        i = self.t % self.steps
        self.t += 1
        n = lambda: max(0.0, 1.0 + self.rng.normal(0, self.noise))
        return {
            "video": round(PEAKS["video"] * self.video_curve[i] * n(), 2),
            "volte": round(PEAKS["volte"] * self.volte_curve[i] * n(), 2),
            "urllc": round(PEAKS["urllc"] * self.urllc_curve[i] * n(), 2),
        }
