"""Live syllable detector.

Method (a causal version of de Jong & Wempe 2009, "Praat script to detect syllable nuclei and measure
speech rate automatically"):
  1. Smooth the 300-2500 Hz level ("band", see frontend.py). Every vowel makes a hill in it; the
     consonants between vowels make valleys.
  2. Peak picking with hysteresis: a hill counts once the level has fallen `min_dip_db` below its top,
     and the next hill can only start once the level has climbed `min_dip_db` out of the valley.
     This is what makes it live: a syllable is reported ~50-100 ms after its vowel's loudest point.
  3. A hill is a syllable NUCLEUS only if it is (a) within `silence_rel_db` of the loudest speech of
     the last 30 s, (b) above the noise floor, and (c) voiced (periodicity >= `voicing` near the top).
     (c) removes the hills that /s/, /sh/, breaths and noise make.
  4. The syllable's START is where the valley before it is: if there was a pause, the moment sound
     resumes (on the broadband level, so an initial /s/ counts); otherwise the bottom of the valley
     between this vowel and the previous one (`onset_rule` picks which point of a flat valley).
"""
from dataclasses import dataclass, field

import numpy as np

from .frontend import FRAME_S


@dataclass
class DetectorParams:
    """Defaults = the values benchmark/eval_syllables.py chose on CMU ARCTIC speakers clb + rms
    (then checked on slt + bdl, which tuning never saw)."""
    band: tuple = (300.0, 2500.0)  # Hz range of the "vowel energy" level (used by the front end)
    min_dip_db: float = 1.5        # how deep the valley between two vowels must be (both sides)
    silence_rel_db: float = 25.0   # nuclei must be within this of the loudest speech (99th percentile)
    floor_margin_db: float = 3.0   # ...and this far above the noise floor (10th percentile)
    voicing: float = 0.6           # periodicity needed within +-30 ms of the peak
    smooth: int = 5                # frames (10 ms each) in the centred smoothing of the band level
    pause_margin_db: float = 6.0   # broadband level below (floor + this) counts as silence
    pause_min_s: float = 0.05      # this much silence before a syllable = "starts out of a pause"
    min_sep_s: float = 0.06        # two nuclei closer than this are one vowel
    onset_rule: str = "end"        # argmin / start / mid / end of the valley (within dip_tol_db of its bottom),
                                   # or "fall": where the level drops off the previous vowel (fall_frac)
    dip_tol_db: float = 1.0
    fall_frac: float = 0.5
    onset_rel_db: float = 30.0     # after a pause: start = first frame within this many dB of the vowel's
                                   # broadband level (skips breaths/noise before speech); 0 = off
    onset_shift_s: float = 0.0     # constant correction added to starts inside a phrase
    max_back_s: float = 0.6        # how far before the peak the start can be
    stats_window_s: float = 30.0


@dataclass
class Syllable:
    idx: int
    t_peak: float                  # nucleus: loudest point of the vowel
    t_onset: float                 # estimated syllable start
    peak_db: float
    per: float                     # periodicity at the nucleus
    f0: float                      # pitch at the nucleus (Hz, 0 = unknown)
    after_pause: bool
    t_end: float = None            # next syllable's start, or where the following pause begins
    formants: list = None          # F1..F4 in Hz at the nucleus (0 = not found)
    bandwidths: list = None
    ceiling: float = None
    # filled in by the alignment with Whisper's words
    status: str = "pending"        # pending / matched / extra / unscored
    word_i: int = None
    syl_i: int = None
    label: str = None              # IPA of the dictionary syllable it matched, e.g. "ˈtɔ"
    vowel: str = None              # ARPAbet vowel it matched, e.g. "AO"
    extra: dict = field(default_factory=dict)


class FrameHistory:
    """Fixed-capacity frame arrays addressed by absolute frame index."""

    FIELDS = ("t", "band", "broad", "per", "f0", "sband")

    def __init__(self, cap=12000):
        self.cap = cap
        self.a = {f: np.zeros(cap * 2) for f in self.FIELDS}
        self.base = 0   # absolute index of a[...][0]
        self.n = 0      # frames stored

    @property
    def end(self):
        return self.base + self.n

    def append(self, fr):
        m = len(fr["t"])
        if self.n + m > len(self.a["t"]):
            drop = min(self.n, self.n + m - self.cap)
            for f in self.FIELDS:
                self.a[f][: self.n - drop] = self.a[f][drop: self.n]
            self.n -= drop
            self.base += drop
            if self.n + m > len(self.a["t"]):
                for f in self.FIELDS:
                    self.a[f] = np.concatenate([self.a[f][: self.n], np.zeros(m + self.cap)])
        for f in ("t", "band", "broad", "per", "f0"):
            self.a[f][self.n: self.n + m] = fr[f]
        self.a["sband"][self.n: self.n + m] = np.nan
        self.n += m

    def view(self, name, k0, k1):
        k0 = max(k0, self.base)
        k1 = min(k1, self.end)
        return self.a[name][k0 - self.base: max(k0, k1) - self.base]

    def get(self, name, k):
        return self.a[name][k - self.base]

    def set(self, name, k, v):
        self.a[name][k - self.base] = v


class SyllableDetector:
    def __init__(self, params=None):
        self.p = params or DetectorParams()
        self.h = FrameHistory()
        self.n_smoothed = 0
        self.state = "rise"
        self.cand = None           # frame index of the hill top being tracked
        self.dip = None            # frame index of the valley bottom being tracked
        self.last_peak_k = None    # frame index of the last accepted nucleus
        self.syllables = []
        self.ref = None            # loudest speech level (99th percentile of the band level)
        self.floor = None          # band noise floor (10th percentile)
        self.bfloor = None         # broadband noise floor (5th percentile)
        self.silent_run = 0
        self.last_stats_k = -999

    # ----------------------------------------------------------------------------------------------
    def push(self, frames):
        """frames: dict from FrameAnalyzer (or None). Returns the syllables confirmed by these frames."""
        if frames is None or len(frames["t"]) == 0:
            return []
        self.h.append(frames)
        hw = self.p.smooth // 2
        out = []
        while self.n_smoothed + hw < self.h.end:
            self._smooth(self.n_smoothed, hw)
            out += self._step(self.n_smoothed)
            self.n_smoothed += 1
        return out

    def flush(self):
        """End of the stream: finish smoothing and accept a hill that the end of the audio cut off."""
        hw = self.p.smooth // 2
        out = []
        while self.n_smoothed < self.h.end:
            self._smooth(self.n_smoothed, hw)
            out += self._step(self.n_smoothed)
            self.n_smoothed += 1
        if self.state == "rise" and self.cand is not None:
            s = self._peak(self.cand)
            if s:
                out.append(s)
        if self.syllables and self.syllables[-1].t_end is None and self.h.n:
            self.syllables[-1].t_end = float(self.h.get("t", self.h.end - 1))
        return out

    # ----------------------------------------------------------------------------------------------
    def _smooth(self, i, hw):
        p = 10 ** (self.h.view("band", i - hw, i + hw + 1) / 10)
        self.h.set("sband", i, 10 * np.log10(np.mean(p) + 1e-12))

    def _update_stats(self, i):
        w = int(self.p.stats_window_s / FRAME_S)
        s = self.h.view("sband", i - w, i + 1)
        b = self.h.view("broad", i - w, i + 1)
        self.ref = float(np.percentile(s, 99))
        self.floor = float(np.percentile(s, 10))
        self.bfloor = float(np.percentile(b, 5))
        self.last_stats_k = i

    def _step(self, i):
        if i - self.last_stats_k >= (1 if i < 100 else 10):
            self._update_stats(i)
        out = []
        v = self.h.get("sband", i)
        silent = self.h.get("broad", i) < self.bfloor + self.p.pause_margin_db
        self.silent_run = self.silent_run + 1 if silent else 0
        pause_frames = max(1, int(round(self.p.pause_min_s / FRAME_S)))
        if self.silent_run == pause_frames and self.syllables and self.syllables[-1].t_end is None:
            self.syllables[-1].t_end = float(self.h.get("t", i - pause_frames + 1))

        d = self.p.min_dip_db
        if self.state == "rise":
            if self.cand is None or v > self.h.get("sband", self.cand):
                self.cand = i
            elif v < self.h.get("sband", self.cand) - d:
                s = self._peak(self.cand)
                if s:
                    out.append(s)
                self.state, self.dip, self.cand = "fall", i, None
        else:
            if v < self.h.get("sband", self.dip):
                self.dip = i
            elif v > self.h.get("sband", self.dip) + d:
                self.state, self.cand, self.dip = "rise", i, None
        return out

    def _peak(self, j):
        p = self.p
        pk = float(self.h.get("sband", j))
        per_near = self.h.view("per", j - 3, j + 4)
        per = float(per_near.max()) if len(per_near) else 0.0
        if self.ref is None:
            self._update_stats(j)
        if pk < self.ref - p.silence_rel_db or pk < self.floor + p.floor_margin_db or per < p.voicing:
            return None
        t = float(self.h.get("t", j))
        if self.last_peak_k is not None and t - float(self.h.get("t", self.last_peak_k)) < p.min_sep_s:
            return None
        onset, after_pause = self._onset(j)
        f0s = self.h.view("f0", j - 2, j + 3)
        f0s = f0s[f0s > 0]
        syl = Syllable(idx=len(self.syllables), t_peak=t, t_onset=onset, peak_db=pk, per=per,
                       f0=float(np.median(f0s)) if len(f0s) else 0.0, after_pause=after_pause)
        if self.syllables and self.syllables[-1].t_end is None:
            self.syllables[-1].t_end = min(onset, t)
        self.syllables.append(syl)
        self.last_peak_k = j
        return syl

    def _onset(self, j):
        p = self.p
        k0 = j - int(p.max_back_s / FRAME_S)
        if self.last_peak_k is not None:
            k0 = max(k0, self.last_peak_k + 1)
        k0 = max(k0, self.h.base)
        t = self.h.view("t", k0, j + 1)
        s = self.h.view("sband", k0, j + 1)
        b = self.h.view("broad", k0, j + 1)
        if len(t) < 2:
            return float(self.h.get("t", j)), self.last_peak_k is None
        silent = b < self.bfloor + p.pause_margin_db
        pause_frames = max(1, int(round(p.pause_min_s / FRAME_S)))
        # the last run of silence long enough to be a pause -> the syllable starts when sound resumes
        run, last_end = 0, None
        for idx in range(len(silent)):
            run = run + 1 if silent[idx] else 0
            if run >= pause_frames:
                last_end = idx
        if last_end is None and self.last_peak_k is None and k0 == self.h.base and silent[0]:
            last_end = int(np.argmax(~silent)) - 1   # very start of the stream: leading quiet = a pause
        if last_end is not None:
            nxt = last_end + 1
            while nxt < len(silent) and silent[nxt]:
                nxt += 1
            nxt = min(nxt, len(t) - 1)
            if p.onset_rel_db > 0:
                # a breath or room noise before speaking is far quieter than the speech itself
                level = float(b[-4:].max()) - p.onset_rel_db
                loud = np.where(b[nxt:] >= level)[0]
                if len(loud):
                    nxt += int(loud[0])
            return float(t[nxt]), True
        m = int(np.argmin(s))
        shift = p.onset_shift_s
        if p.onset_rule == "argmin":
            return float(t[m]) + shift, False
        if p.onset_rule == "fall" and self.last_peak_k is not None and self.last_peak_k >= self.h.base:
            # where the level falls off the previous vowel: the first point after the previous peak
            # that is below (valley bottom + fall_frac * (previous peak - valley bottom)). That is where
            # the consonant(s) before this vowel begin, which is where a syllable "starts".
            top = float(self.h.get("sband", self.last_peak_k))
            level = s[m] + p.fall_frac * (top - s[m])
            below = np.where(s[: m + 1] <= level)[0]
            return float(t[below[0] if len(below) else m]) + shift, False
        low = s <= s[m] + p.dip_tol_db
        a = m
        while a > 0 and low[a - 1]:
            a -= 1
        z = m
        while z < len(s) - 1 and low[z + 1]:
            z += 1
        k = {"start": a, "end": z, "mid": (a + z) // 2}.get(p.onset_rule, m)
        return float(t[k]) + shift, False
