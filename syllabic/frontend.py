"""Streaming front end: microphone audio at any rate -> 16 kHz -> one analysis frame every 10 ms.

Each frame k is CENTRED on t = k * 10 ms. (main11 stamps a value at the END of its window, which made
its pitch look ~30 ms late in the benchmark; centring avoids that.) A frame can only be computed once
20 ms of audio after its centre has arrived, so the front end runs 20 ms behind the microphone.

Per frame:
  band   level of the 300-2500 Hz band in dB. Vowels put most of their energy here (F1 and F2), so it
         peaks on vowels and dips on consonants: the "syllable envelope".
  broad  level of 100-7800 Hz in dB. Also sees fricatives like /s/, so it is used to tell real pauses
         apart from consonants.
  per    periodicity 0..1: the highest normalised autocorrelation over pitch lags (75-600 Hz).
         Vowels are periodic (>0.6); noise, breath and /s/ are not. Same check that halved the
         voicing errors of get_pitch in the earlier experiment.
  f0     pitch estimate (Hz) from the best autocorrelation lag, 0 when per is low.
"""
import numpy as np
import scipy.signal as sps
import soxr

SR = 16000
HOP = 160                 # 10 ms
FRAME_S = HOP / SR
WIN = 512                 # 32 ms window for the band levels
PER_WIN = 640             # 40 ms window for periodicity
LOOKAHEAD = PER_WIN // 2  # samples needed after a frame centre (20 ms)
F0_MIN, F0_MAX = 75.0, 600.0
LAG_LO, LAG_HI = int(SR / F0_MAX), int(np.ceil(SR / F0_MIN))

BAND = (300.0, 2500.0)
BROAD = (100.0, 7800.0)


class Resampler:
    """Streaming resampler to 16 kHz (soxr keeps the timing exact across blocks)."""

    def __init__(self, in_sr):
        self.in_sr = int(round(in_sr))
        self.stream = None if self.in_sr == SR else soxr.ResampleStream(self.in_sr, SR, 1, dtype="float32",
                                                                         quality="HQ")

    def process(self, x, last=False):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        if self.stream is None:
            return x
        return self.stream.resample_chunk(x, last=last)


class HighPass:
    """60 Hz high-pass so mains hum and rumble do not look like voice or loudness."""

    def __init__(self, cutoff=60.0):
        self.sos = sps.butter(2, cutoff, btype="highpass", fs=SR, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, x):
        if len(x) == 0:
            return np.zeros(0, np.float32)
        y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


def to_16k(y, sr):
    """Whole-file conversion (offline use): mono float32 at 16 kHz, high-passed like the live path."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if int(sr) != SR:
        y = soxr.resample(y, sr, SR).astype(np.float32)
    return HighPass().process(y)


OCTAVE_COST = 0.01   # Praat's default: per octave of lag, so a higher octave wins only if nearly as strong


def periodicity(frames):
    """frames (n, W) -> (per, f0). per = max normalised autocorrelation over lags LAG_LO..LAG_HI.
    f0 = the autocorrelation peak that is strongest after Praat's octave cost, refined by parabolic
    interpolation. (The first version took the shortest lag within 90% of the best peak. In vowels whose
    F1 sits near the 2nd harmonic that picked the octave above: 18% of one session's vowel tokens, vs
    5% now, judged against Praat's pitch.)"""
    x = frames - frames.mean(axis=1, keepdims=True)
    n, w = x.shape
    if n == 0:
        return np.zeros(0), np.zeros(0)
    nfft = 1 << (2 * w - 1).bit_length()
    spec = np.fft.rfft(x, nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :w]
    c = np.cumsum(x * x, axis=1)
    taus = np.arange(LAG_LO, LAG_HI + 1)
    e1 = c[:, w - taus - 1]
    e2 = c[:, -1:] - c[:, taus - 1]
    r = ac[:, taus] / np.sqrt(e1 * e2 + 1e-20)
    r[c[:, -1] < 1e-10] = 0.0
    best = r.max(axis=1)
    peak = np.zeros_like(r, dtype=bool)
    peak[:, 1:-1] = (r[:, 1:-1] > r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
    score = np.where(peak, r - OCTAVE_COST * np.log2(taus / taus[0])[None, :], -np.inf)
    has = peak.any(axis=1)
    j = np.where(has, score.argmax(axis=1), r.argmax(axis=1))
    j = np.clip(j, 1, r.shape[1] - 2)
    rows = np.arange(n)
    a, b, cc = r[rows, j - 1], r[rows, j], r[rows, j + 1]
    den = a - 2 * b + cc
    shift = np.where(np.abs(den) > 1e-9, 0.5 * (a - cc) / den, 0.0)
    lag = taus[j] + np.clip(shift, -1, 1)
    f0 = np.where(best >= 0.45, SR / lag, 0.0)
    return np.clip(best, 0.0, 1.0), f0


class FrameAnalyzer:
    """Push 16 kHz audio in any block size; get back the frames that became computable."""

    def __init__(self, band=BAND, broad=BROAD):
        self.buf = np.zeros(LOOKAHEAD, np.float32)  # silence before t = 0 so frame 0 is centred on 0
        self.buf_t0 = -LOOKAHEAD                     # absolute sample index of buf[0]
        self.k = 0                                   # next frame to compute
        self.n_samples = 0                           # real samples received
        freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
        self.band_bins = (freqs >= band[0]) & (freqs <= band[1])
        self.broad_bins = (freqs >= broad[0]) & (freqs <= broad[1])
        self.win = np.hanning(WIN).astype(np.float32)
        self.norm = 2.0 / (WIN * np.sum(self.win.astype(np.float64) ** 2))
        self.off_level = np.arange(-WIN // 2, WIN // 2)
        self.off_per = np.arange(-PER_WIN // 2, PER_WIN // 2)

    def push(self, x16):
        x16 = np.asarray(x16, dtype=np.float32)
        self.n_samples += len(x16)
        self.buf = np.concatenate([self.buf, x16])
        return self._compute(self.buf_t0 + len(self.buf))

    def flush(self):
        """End of stream: pad with silence so the frames up to the last real sample get computed."""
        last_k = self.n_samples // HOP
        need = last_k * HOP + LOOKAHEAD - (self.buf_t0 + len(self.buf))
        if need > 0:
            self.buf = np.concatenate([self.buf, np.zeros(need, np.float32)])
        return self._compute(self.buf_t0 + len(self.buf), max_k=last_k)

    def _compute(self, end, max_k=None):
        k_last = (end - LOOKAHEAD) // HOP
        if max_k is not None:
            k_last = min(k_last, max_k)
        if k_last < self.k:
            return None
        ks = np.arange(self.k, k_last + 1)
        centres = ks * HOP - self.buf_t0
        fl = self.buf[centres[:, None] + self.off_level[None, :]]
        fp = self.buf[centres[:, None] + self.off_per[None, :]]
        spec = np.abs(np.fft.rfft(fl * self.win, axis=1)) ** 2
        band = 10 * np.log10(spec[:, self.band_bins].sum(axis=1) * self.norm + 1e-12)
        broad = 10 * np.log10(spec[:, self.broad_bins].sum(axis=1) * self.norm + 1e-12)
        per, f0 = periodicity(fp.astype(np.float64))
        self.k = k_last + 1
        keep = self.k * HOP - LOOKAHEAD - self.buf_t0
        if keep > 0:
            self.buf = self.buf[keep:]
            self.buf_t0 += keep
        return {"k": ks, "t": ks * FRAME_S, "band": band, "broad": broad, "per": per, "f0": f0}


class AudioStore:
    """Keeps the last `seconds` of 16 kHz audio, addressed by absolute time since the start."""

    def __init__(self, seconds=180.0):
        self.cap = int(seconds * SR)
        self.data = np.zeros(self.cap * 2, np.float32)
        self.start = 0   # absolute sample index of data[0]
        self.n = 0       # valid samples in data

    @property
    def total(self):
        return self.start + self.n

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.n + len(x) > len(self.data):
            drop = self.n + len(x) - self.cap
            drop = min(max(drop, 0), self.n)
            self.data[: self.n - drop] = self.data[drop: self.n]
            self.n -= drop
            self.start += drop
            if len(x) > len(self.data) - self.n:  # one huge block (offline file): grow
                self.data = np.concatenate([self.data[: self.n], np.zeros(len(x), np.float32)])
        self.data[self.n: self.n + len(x)] = x
        self.n += len(x)

    def get(self, t0, t1):
        """Samples between t0 and t1 seconds (zeros where nothing is stored)."""
        a, b = int(round(t0 * SR)), int(round(t1 * SR))
        out = np.zeros(max(0, b - a), np.float32)
        lo, hi = max(a, self.start), min(b, self.total)
        if hi > lo:
            out[lo - a: hi - a] = self.data[lo - self.start: hi - self.start]
        return out
