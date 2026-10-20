"""Scoring functions. All pitch metrics are the standard ones from the pitch-tracking literature
(e.g. Rabiner et al. 1976; Salamon et al. 2014, mir_eval), computed frame by frame."""
import numpy as np

GROSS_TOL = 0.20      # a pitch more than 20% off counts as a gross error
RPA_CENTS = 50.0      # "raw pitch accuracy": within 50 cents (half a semitone) of the truth
OCTAVE_TOL = 0.10     # |log2(est/ref)| within 0.1 of +-1  ->  octave error

F0_BINS = [(0, 75), (75, 100), (100, 150), (150, 200), (200, 300), (300, 2000)]


def cents(est, ref):
    return 1200.0 * np.log2(est / ref)


def pitch_counts(ref, est):
    """ref/est: aligned arrays of Hz with 0 (or negative) = unvoiced. Returns raw counts, so results
    from many files can be added up and turned into percentages at the end (frame-weighted)."""
    ref = np.asarray(ref, float)
    est = np.asarray(est, float)
    rv, ev = ref > 0, est > 0
    both = rv & ev
    c = np.zeros(int(both.sum()))
    if both.any():
        c = cents(est[both], ref[both])
    ratio_err = np.abs(est[both] / ref[both] - 1) if both.any() else np.zeros(0)
    gross = ratio_err > GROSS_TOL
    octave = np.abs(np.abs(c / 1200.0) - 1) < OCTAVE_TOL
    return {
        "frames": len(ref), "ref_voiced": int(rv.sum()), "ref_unvoiced": int((~rv).sum()),
        "both_voiced": int(both.sum()), "missed": int((rv & ~ev).sum()), "false_alarm": int((~rv & ev).sum()),
        "gross": int(gross.sum()), "octave": int(octave.sum()), "within_50c": int((np.abs(c) < RPA_CENTS).sum()),
        "fine_abs_cents": np.abs(c[~gross]),
    }


def add_counts(a, b):
    if a is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    out = {}
    for k in a:
        out[k] = np.concatenate([a[k], b[k]]) if isinstance(a[k], np.ndarray) else a[k] + b[k]
    return out


def pitch_summary(c):
    """Percentages from summed counts.
    voicing_recall   % of truly voiced frames where a pitch was reported
    false_alarm      % of truly unvoiced/silent frames where a pitch was reported anyway
    VDE              voicing decision error: % of all frames with the wrong voiced/unvoiced call
    RPA              raw pitch accuracy: % of truly voiced frames with a pitch within 50 cents
    GPE              gross pitch error: % of frames both call voiced where the pitch is >20% off
    octave           % of those frames that are octave jumps (half or double)
    fine_cents       median error in cents on frames that are not gross errors"""
    if c is None or c["frames"] == 0:
        return {}
    pct = lambda n, d: 100.0 * n / d if d else float("nan")  # noqa: E731
    return {
        "frames": c["frames"],
        "voicing_recall": pct(c["ref_voiced"] - c["missed"], c["ref_voiced"]),
        "false_alarm": pct(c["false_alarm"], c["ref_unvoiced"]),
        "VDE": pct(c["missed"] + c["false_alarm"], c["frames"]),
        "RPA": pct(c["within_50c"], c["ref_voiced"]),
        "GPE": pct(c["gross"], c["both_voiced"]),
        "octave": pct(c["octave"], c["both_voiced"]),
        "fine_cents": float(np.median(c["fine_abs_cents"])) if len(c["fine_abs_cents"]) else float("nan"),
    }


def shift(est, lag, n):
    """Returns est re-indexed so out[k] = est[k + lag] for k in range(n) (0 where out of range)."""
    out = np.zeros(n)
    k = np.arange(n)
    j = k + lag
    ok = (j >= 0) & (j < len(est))
    out[ok] = est[j[ok]]
    return out


def best_lag(pairs, lags):
    """pairs: list of (ref, est) arrays on the same 10 ms grid. Picks the lag (in frames) that gives the
    highest overall raw pitch accuracy. Returns (best_lag, {lag: RPA})."""
    scores = {}
    for lag in lags:
        tot_ok = tot_v = 0
        for ref, est in pairs:
            e = shift(est, lag, len(ref))
            rv = ref > 0
            both = rv & (e > 0)
            if both.any():
                tot_ok += int((np.abs(cents(e[both], ref[both])) < RPA_CENTS).sum())
            tot_v += int(rv.sum())
        scores[lag] = 100.0 * tot_ok / max(tot_v, 1)
    return max(scores, key=scores.get), scores


def by_f0_bin(ref, est):
    """Raw pitch accuracy split by the true pitch range (to show where the tracker struggles)."""
    rows = []
    for lo, hi in F0_BINS:
        m = (ref >= lo) & (ref < hi) & (ref > 0)
        if m.sum() == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        e = est[m]
        detected = 100.0 * (e > 0).mean()
        ok = (e > 0)
        acc = 100.0 * (np.abs(cents(e[ok], ref[m][ok])) < RPA_CENTS).sum() / m.sum() if ok.any() else 0.0
        rows.append((lo, hi, int(m.sum()), detected, acc))
    return rows


# ------------------------------------------------------------------------------------------------
# formants
# ------------------------------------------------------------------------------------------------
def formant_summary(ref, est):
    """ref/est: arrays of Hz for one formant (ref 0 = not measurable, est 0 = not reported).
    coverage   % of measurable cases where a value was reported
    median_pct median absolute error in % of the true value (only where reported)
    median_hz  median absolute error in Hz
    within_10  % of measurable cases reported AND within 10% of the truth (missing counts as a miss)"""
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    m = ref > 0
    rep = m & (est > 0)
    if m.sum() == 0:
        return {}
    err = np.abs(est[rep] - ref[rep])
    pct = 100 * err / ref[rep]
    return {
        "n": int(m.sum()),
        "coverage": 100.0 * rep.sum() / m.sum(),
        "median_pct": float(np.median(pct)) if rep.any() else float("nan"),
        "median_hz": float(np.median(err)) if rep.any() else float("nan"),
        "within_10": 100.0 * (pct < 10).sum() / m.sum(),
    }


# ------------------------------------------------------------------------------------------------
# voice-quality features (no ground truth exists, so test whether they behave sensibly)
# ------------------------------------------------------------------------------------------------
def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(y[ok]) == 0:
        return 0.0
    return float(spearmanr(x[ok], y[ok]).correlation)
