"""Formants at a syllable nucleus, with Praat's Burg method (through Parselmouth, same as main10).

The value reported is the MEDIAN of the Burg frames within +-`half` seconds of the nucleus, which
steadies it against single-frame jumps. The analysis window length hardly matters (12-35 ms all within
~1 point on Hillenbrand), but the formant ceiling (Praat's "maximum formant") matters a lot: too low and
F2/F3 merge (your /i/ F2 fell from 2247 to 1740 Hz at 4500). Ceiling choices:
  "speaker"  (default) one ceiling per speaker, re-estimated as you talk: the candidate that makes tokens
             of the same vowel most alike (Escudero et al. 2009). Beat "auto" for all 5 voices tested;
             lands at 6000 Hz for both ARCTIC women and 4500 Hz for both men.
  "auto"     a ceiling per vowel token, the one with the smoothest F1-F3 tracks (Barreda's "Fast Track"
             idea). Best on isolated Hillenbrand words, but picks too-low ceilings on ~20% of running-speech
             tokens, which scatters them.
  a number   fixed (Praat's advice: ~5000 Hz adult male, ~5500 Hz adult female).
"""
from collections import deque

import numpy as np
import parselmouth

from .frontend import SR

AUTO_CEILINGS = (4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 5750.0, 6000.0, 6500.0)
SPEED_OF_SOUND_CM_S = 35000.0


def bark(f):
    """Hz -> Bark (Traunmueller 1990): a perceptual scale, so spreads are compared fairly across F1/F2."""
    f = np.asarray(f, dtype=float)
    return 26.81 * f / (1960.0 + f) - 0.53


def _burg(seg, t0, ceiling):
    snd = parselmouth.Sound(np.asarray(seg, dtype=np.float64), sampling_frequency=SR, start_time=t0)
    return snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=float(ceiling),
                               window_length=0.025, pre_emphasis_from=50.0)


def _tracks(fm, t, half, n=4):
    ts = np.array([x for x in fm.ts() if abs(x - t) <= half + 1e-9])
    F = np.full((len(ts), n), np.nan)
    B = np.full((len(ts), n), np.nan)
    for a, tt in enumerate(ts):
        for i in range(n):
            F[a, i] = fm.get_value_at_time(i + 1, tt)
            B[a, i] = fm.get_bandwidth_at_time(i + 1, tt)
    return ts, F, B


def _median(F):
    """Per-column median ignoring NaN (0 where a formant was never found). Same result as
    np.nanmedian(F, axis=0), which on these few-row arrays went through masked arrays and took half
    of the whole pipeline's time; sorting once is ~20x faster."""
    F = np.asarray(F, dtype=float)
    out = np.zeros(F.shape[1])
    if not len(F):
        return out
    s = np.sort(F, axis=0)                     # NaN sorts to the end of each column
    for i, k in enumerate((~np.isnan(F)).sum(axis=0)):
        if k:
            m = 0.5 * (s[(k - 1) // 2, i] + s[k // 2, i])
            out[i] = m if np.isfinite(m) else 0.0
    return out


def _roughness(ts, F):
    """How badly F1-F3 jump around near the nucleus: mean relative residual from a quadratic fit,
    plus penalties for missing or implausible values (what a wrong ceiling produces)."""
    if len(ts) < 5:
        return np.inf
    err = 0.0
    for i in range(3):
        f = F[:, i]
        ok = np.isfinite(f)
        if ok.sum() < 5:
            return np.inf
        err += (1 - ok.mean()) * 2.0
        c = np.polyfit(ts[ok] - ts.mean(), f[ok], 2)
        res = f[ok] - np.polyval(c, ts[ok] - ts.mean())
        err += float(np.mean(np.abs(res)) / max(np.mean(f[ok]), 1.0))
    med = _median(F)
    if not (150 <= med[0] <= 1300) or not (med[1] - med[0] >= 150) or not (med[2] - med[1] >= 150):
        err += 1.0
    return err


def measure(audio_store_or_array, t, ceiling=5500.0, half=0.02, seg_t0=None):
    """Formants F1..F4 (Hz, 0 = not found) and bandwidths at time t.

    audio_store_or_array: an AudioStore (live) or a whole 16 kHz array (then seg_t0 must be None
    and t is seconds from the array start). ceiling: a number or "auto".
    Returns (F[4], B[4], ceiling_used)."""
    span = 0.07 if ceiling == "auto" else half + 0.04
    if hasattr(audio_store_or_array, "get"):
        seg = audio_store_or_array.get(t - span, t + span)
        t0 = t - span
    else:
        y = audio_store_or_array
        a = max(0, int(round((t - span) * SR)))
        b = min(len(y), int(round((t + span) * SR)))
        seg, t0 = y[a:b], a / SR
    if len(seg) < int(0.05 * SR) or not np.any(seg):
        return [0.0] * 4, [0.0] * 4, ceiling
    if ceiling != "auto":
        ts, F, B = _tracks(_burg(seg, t0, ceiling), t, half)
        return list(_median(F)), list(_median(B)), float(ceiling)
    best = None
    for c in AUTO_CEILINGS:
        fm = _burg(seg, t0, c)
        ts, F, B = _tracks(fm, t, 0.04)
        r = _roughness(ts, F)
        if best is None or r < best[0]:
            best = (r, c, fm)
    _, c, fm = best
    ts, F, B = _tracks(fm, t, half)
    return list(_median(F)), list(_median(B)), float(c)


def measure_all(audio_store_or_array, t, ceilings=AUTO_CEILINGS, half=0.02):
    """Formants at t for every candidate ceiling: {ceiling: (F[4], B[4])}. Same cost as "auto"; lets the
    speaker-level ceiling be changed later without measuring again."""
    span = half + 0.04
    if hasattr(audio_store_or_array, "get"):
        seg, t0 = audio_store_or_array.get(t - span, t + span), t - span
    else:
        y = audio_store_or_array
        a, b = max(0, int(round((t - span) * SR))), min(len(y), int(round((t + span) * SR)))
        seg, t0 = y[a:b], a / SR
    if len(seg) < int(0.05 * SR) or not np.any(seg):
        return {float(c): ([0.0] * 4, [0.0] * 4) for c in ceilings}
    out = {}
    for c in ceilings:
        ts, F, B = _tracks(_burg(seg, t0, c), t, half)
        out[float(c)] = (list(_median(F)), list(_median(B)))
    return out


class SpeakerCeiling:
    """One formant ceiling per speaker (Escudero, Boersma, Rauber & Bion 2009): the candidate that makes
    tokens of the same vowel most alike, i.e. the smallest within-vowel spread of F1/F2 in Bark. A wrong
    ceiling makes the tracker merge or swap formants on some tokens, which scatters them.
    It re-estimates every `every` tokens over the last `window` tokens, so it follows a voice that changes
    during training. With only a few minutes of speech the estimate is noisy (its curve is often flat over
    several ceilings), so it waits for `min_tokens` over `min_vowels` vowels, needs a `margin` improvement,
    and moves ONE step at a time toward the best ceiling: no sudden reshuffling of the chart."""

    def __init__(self, ceilings=AUTO_CEILINGS, default=5500.0, window=400, min_tokens=60, min_vowels=5,
                 every=10, margin=0.03):
        self.ceilings = tuple(float(c) for c in ceilings)
        self.current = float(default)
        self.tokens = deque(maxlen=window)
        self.min_tokens, self.min_vowels, self.every, self.margin = min_tokens, min_vowels, every, margin
        self.n_new = 0
        self.last_spread = {}

    def add(self, vowel, f12):
        """f12: array (len(ceilings), 2) of this token's F1, F2 at each candidate ceiling (same order as
        self.ceilings). Returns True if the speaker ceiling changed."""
        f12 = np.asarray(f12, dtype=float)
        if f12.shape != (len(self.ceilings), 2):
            return False
        self.tokens.append((vowel, f12))
        self.n_new += 1
        if self.n_new < self.every:
            return False
        self.n_new = 0
        return self.update()

    def spread(self):
        """{ceiling: mean within-vowel absolute deviation (Bark, F1 and F2)} over vowels with >= 3 tokens."""
        by = {}
        for v, a in self.tokens:
            by.setdefault(v, []).append(a)
        groups = [np.array(g) for g in by.values() if len(g) >= 3]          # each (tokens, ceilings, 2)
        if sum(len(g) for g in groups) < self.min_tokens or len(groups) < self.min_vowels:
            return {}
        out = {}
        for i, c in enumerate(self.ceilings):
            d = []
            for g in groups:
                a = g[:, i, :]
                a = a[(a[:, 0] > 0) & (a[:, 1] > 0)]
                if len(a) >= 3:
                    b = bark(a)
                    d.append(float(np.mean(np.abs(b - np.median(b, axis=0)))))
            if d:
                out[c] = float(np.mean(d))
        return out

    def update(self):
        s = self.spread()
        self.last_spread = s
        if not s:
            return False
        best = min(s, key=s.get)
        cur = s.get(self.current, np.inf)
        if best != self.current and s[best] < cur * (1 - self.margin):
            i, j = self.ceilings.index(self.current), self.ceilings.index(best)
            self.current = self.ceilings[i + (1 if j > i else -1)]
            return True
        return False


def apparent_vtl_cm(F):
    """Apparent vocal-tract length from formant spacing (Reby & McComb 2003 regression):
    a uniform tube closed at one end has F_i = (2i - 1) / 2 * dF, so dF is fitted through the origin
    and VTL = c / (2 dF). Uses F1..F4; 0 when fewer than 3 are present. Dominated by F3/F4, which
    depend more on the speaker's size than on the vowel."""
    F = np.asarray(F, dtype=float)[:4]
    i = np.arange(1, len(F) + 1)
    ok = F > 0
    if ok.sum() < 3:
        return 0.0
    w = (2 * i[ok] - 1) / 2.0
    dF = float(np.sum(F[ok] * w) / np.sum(w * w))
    return SPEED_OF_SOUND_CM_S / (2 * dF) if dF > 0 else 0.0
