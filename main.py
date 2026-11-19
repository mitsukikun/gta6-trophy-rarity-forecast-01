"""gta6-trophy-rarity-forecast-01: forecast every GTA VI PS5 trophy's earn rate
on Exophase at the case's target, as five quantiles a trophy.

Python 3.13, standard library only. Run once per case:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

It reads every <id>.in in the input directory (the grader runs one case at a
time, with only that case's file there) and writes <id>.out to the output
directory: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Model (counts, not rates -- Exophase's page is a snapshot of two moving
quantities, the players tracked N and the players holding each trophy E_i,
and a rate is E_i / N):
  - History is grouped into epochs: consecutive reads showing the identical
    page. Exophase refreshes its counts on its own schedule, hours apart,
    so each epoch is one refresh; the refresh instant is estimated as the
    midpoint between the previous epoch's last read and this epoch's first.
  - N at the target comes from a power-law fit of log N on log(1 + hours
    since release) over the last 8 epochs, slope damped 0.9. The implied
    growth factor over the future window is capped at 8x: extrapolating a
    fitted power law 4.5 days ahead explodes when growth is saturating.
    The future N trajectory runs through the last epoch's point with the
    capped slope, so the dilution and the progression below use one
    consistent N(t).
  - Each refresh jump decomposes, per trophy, as
        dE_i = p_i * dN + g_i * N_old * dt
    p_i is the share of the newly tracked players who already hold the
    trophy, g_i the per-player per-hour earn rate of the players already
    tracked. Both come from a non-negative least-squares fit over the
    recent refreshes: this separates dilution (new players arriving with
    few trophies) from progression (old players earning more over time),
    which a rate-only trend fit cannot do.
  - E_i(target) = E_last + p_i * (N_target - N_last)
                        + g_i * (integral of N(t) dt over the future window)
    and the median rate is 100 * E_i(target) / N_target. As N grows the
    forecast interpolates from the last rate toward the marginal cohort's
    rate; as the window lengthens, the progress term grows with it.
  - Spread (logit sigma) combines a random walk 0.5*sqrt(days), the
    scatter of the decomposition fit, and the uncertainty in N_target,
    in quadrature, floored at 0.5 and capped at 2.0. A page frozen far
    longer than the usual refresh gap widens it further: the displayed
    rates are stale and a refresh is overdue.
  - Flat history (no refresh ever observed): the page has been frozen
    since launch while the game just released, so a refresh is expected
    before the target. The median dilutes the last rate by a factor J=4
    toward the new cohort's rate q = 100*(r/100)^1.5 -- joiners have
    played far less than the launch-night cohort, so they hold each
    trophy at a much lower rate (a trophy everyone gets instantly stays
    near 100% under this form) -- with a wide honest spread (sigma 0.9).

Deterministic: no randomness, no dict-order dependence, fixed float
formatting. Runtime is far under the 8 s cap (milliseconds per case).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (this entry is one file, but keep
# the line for consistency with the kit's starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Stale-page fallback (flat history): expected tracked-player multiplier
# at the next refresh, new-cohort earn-rate exponent, spread.
J_FALLBACK = 4.0
GAMMA_FALLBACK = 1.5
SIG_FALLBACK = 0.9
# Decomposition model.
DAMP = 0.9           # slope damping on the log-N power-law fit
MAX_REFRESHES = 10   # recent refreshes feeding the (p, g) fit
N_FIT_EPOCHS = 8     # recent epochs feeding the log-N fit
N_CAP = 8.0          # max N_target / N_last
SIG_N = 0.35         # log-uncertainty of N_target
SIG_FLOOR = 0.5
SIG_CAP = 2.0
STALE_GAP_MULT = 2.0  # widen spread when the page is frozen this many
STALE_SIG_EXTRA = 0.3  # times longer than the median refresh gap


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def invlogit(l):
    if l > 30:
        return 100.0
    if l < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """x rounded to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def epochs_of(history):
    """Group consecutive identical reads into epochs (one refresh each).

    Each epoch carries t, the estimated refresh instant: the midpoint
    between the previous epoch's last read and this epoch's first read --
    the refresh happened in that gap, and the displayed N is N_true at
    the refresh, not at the first read that shows it.
    """
    eps = []
    for r in history:
        key = (r["players_tracked"], tuple(r["rates"]))
        if eps and eps[-1]["key"] == key:
            eps[-1]["last_at"] = r["at"]
            continue
        eps.append({"key": key, "t": parse_ts(r["at"]), "last_at": r["at"],
                    "N": r["players_tracked"], "rates": list(r["rates"])})
    for k in range(1, len(eps)):
        prev_last = parse_ts(eps[k - 1]["last_at"])
        first = eps[k]["t"]
        eps[k]["t"] = prev_last + (first - prev_last) / 2
    return eps


def linfit(xs, ys):
    """Least squares ys = a + b*xs. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def solve_pg(rows):
    """Non-negative least squares for dE = p*dN + g*(N_old*dt).

    rows are (dN, N_old*dt, dE). Returns (p, g) with p in [0, 1], g >= 0.
    """
    s11 = sum(r[0] ** 2 for r in rows)
    s12 = sum(r[0] * r[1] for r in rows)
    s22 = sum(r[1] ** 2 for r in rows)
    t1 = sum(r[0] * r[2] for r in rows)
    t2 = sum(r[1] * r[2] for r in rows)
    det = s11 * s22 - s12 * s12
    if det <= 0 or s11 <= 0 or s22 <= 0:
        p = max(0.0, t1 / s11) if s11 > 0 else 0.0
        return min(1.0, p), 0.0
    p = (t1 * s22 - t2 * s12) / det
    g = (t2 * s11 - t1 * s12) / det
    if p < 0 and g < 0:
        return 0.0, 0.0
    if p < 0:
        return 0.0, max(0.0, t2 / s22)
    if g < 0:
        return min(1.0, max(0.0, t1 / s11)), 0.0
    return min(1.0, p), g


def last_rate(eps, i):
    """A trophy's rate in the most recent epoch that lists it."""
    for e in reversed(eps):
        r = e["rates"][i]
        if r is not None:
            return r
    return 0.01


def forecast_case(case):
    trophies = case["trophies"]
    ntr = len(trophies)
    hist = case["history"]
    if not hist:
        raise ValueError("case has no history")
    last = hist[-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    release = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    try:
        rd = case.get("game", {}).get("release_date")
        if rd:
            release = datetime.fromisoformat(rd + "T00:00:00+00:00")
    except Exception:
        pass
    checkpoint_t = parse_ts(case["at"])
    target_t = parse_ts(case["target"])
    h_target = max(0.0, (target_t - release).total_seconds() / 3600.0)
    h_last = max(0.0, (checkpoint_t - release).total_seconds() / 3600.0)

    eps = epochs_of(hist)

    def quantiles(median_pct, sig):
        lg = logit(median_pct)
        qs = [round4(invlogit(lg + z * sig)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        return [min(100.0, max(0.0, q)) for q in qs]

    rows = []
    if len(eps) < 2:
        # ---- stale-page fallback: no refresh ever observed ----
        for i, t in enumerate(trophies):
            r = last_rate(eps, i)
            q_new = 100.0 * (r / 100.0) ** GAMMA_FALLBACK if r > 0 else 0.01
            med = (r + (J_FALLBACK - 1.0) * q_new) / J_FALLBACK
            rows.append({"id": t["id"], "q": quantiles(med, SIG_FALLBACK)})
        return {"trophies": rows}

    # ---- N at the target: recent-epoch power-law fit, growth capped ----
    use = eps[-N_FIT_EPOCHS:]
    xs = [math.log(1.0 + max(0.0, (e["t"] - release).total_seconds() / 3600.0))
          for e in use]
    ys = [math.log(max(1.0, e["N"])) for e in use]
    a, b = linfit(xs, ys)
    b_d = b * DAMP
    a_d = (sum(ys) / len(ys)) - b_d * (sum(xs) / len(xs))
    N_last = eps[-1]["N"]
    n_uncapped = max(N_last, math.exp(a_d + b_d * math.log(1.0 + h_target)))
    growth_c = min(n_uncapped / N_last, N_CAP)
    N_target = N_last * growth_c
    # Future trajectory through the last epoch's point, hitting N_target:
    # N(h) = N_last * ((1+h)/(1+h_last))^b_c. Its integral feeds the
    # progress term, so dilution and progression use one consistent N(t).
    if h_target > h_last and growth_c > 1.0:
        b_c = math.log(growth_c) / (math.log(1.0 + h_target) - math.log(1.0 + h_last))
    else:
        b_c = 0.0
    if abs(b_c + 1.0) < 1e-9:
        int_n = N_last * (1.0 + h_last) * math.log((1.0 + h_target) / (1.0 + h_last))
    else:
        int_n = (N_last * (1.0 + h_last) / (b_c + 1.0)
                 * (((1.0 + h_target) / (1.0 + h_last)) ** (b_c + 1.0) - 1.0))
    int_n = max(0.0, int_n)
    dt_future = max(0.0, h_target - h_last)

    E_last = [last_rate(eps, i) * N_last / 100.0 for i in range(ntr)]

    # Refresh jumps for the (p, g) decomposition.
    jumps = []
    for k in range(1, len(eps)):
        d_n = eps[k]["N"] - eps[k - 1]["N"]
        dt = (eps[k]["t"] - eps[k - 1]["t"]).total_seconds() / 3600.0
        if d_n <= 0 or dt <= 0:
            continue
        d_e = []
        for i in range(ntr):
            r0 = eps[k - 1]["rates"][i]
            r1 = eps[k]["rates"][i]
            if r0 is None or r1 is None:
                d_e.append(0.0)
                continue
            d_e.append(r1 * eps[k]["N"] / 100.0 - r0 * eps[k - 1]["N"] / 100.0)
        jumps.append((d_n, eps[k - 1]["N"] * dt, d_e))
    jumps = jumps[-MAX_REFRESHES:]

    # Staleness: how long the displayed page has been frozen, against the
    # usual refresh gap. An overdue refresh widens the spread.
    gaps = [(eps[k]["t"] - eps[k - 1]["t"]).total_seconds() / 3600.0
            for k in range(1, len(eps))]
    gaps.sort()
    med_gap = gaps[len(gaps) // 2] if gaps else 0.0
    stale_h = (checkpoint_t - eps[-1]["t"]).total_seconds() / 3600.0
    sig_stale = STALE_SIG_EXTRA if (med_gap > 0 and stale_h > STALE_GAP_MULT * med_gap) else 0.0

    rw = 0.5 * math.sqrt(horizon_days)
    for i, t in enumerate(trophies):
        rows_i = [(j[0], j[1], j[2][i]) for j in jumps]
        p_i, g_i = solve_pg(rows_i)
        resid = [d_e - (p_i * d_n + g_i * ndt) for d_n, ndt, d_e in rows_i]
        sd_e = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
        e_t = E_last[i] + p_i * (N_target - N_last) + g_i * int_n
        e_t = min(max(e_t, E_last[i]), N_target)
        med = 100.0 * e_t / N_target
        dlogit_de = (100.0 / N_target) * (100.0 / max(1e-6, med * (100.0 - med)))
        sig_fit = sd_e * dlogit_de * math.sqrt(max(1.0, dt_future / 24.0))
        e_hi = E_last[i] + p_i * (N_target * math.exp(SIG_N) - N_last) + g_i * int_n * math.exp(SIG_N)
        e_lo = E_last[i] + p_i * (N_target * math.exp(-SIG_N) - N_last) + g_i * int_n * math.exp(-SIG_N)
        r_hi = 100.0 * min(e_hi, N_target * math.exp(SIG_N)) / (N_target * math.exp(SIG_N))
        r_lo = 100.0 * max(e_lo, E_last[i]) / (N_target * math.exp(-SIG_N))
        sig_n = abs(logit(r_hi) - logit(r_lo)) / 2.0
        sig = math.sqrt(rw * rw + sig_fit * sig_fit + sig_n * sig_n + sig_stale * sig_stale)
        sig = min(SIG_CAP, max(SIG_FLOOR, sig))
        rows.append({"id": t["id"], "q": quantiles(med, sig)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels", LEVELS) != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
