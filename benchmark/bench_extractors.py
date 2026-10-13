"""The things being compared: your analysis code (main10's get_audio_data) and reference tools.

`run_yours` reproduces exactly what main11 does in headless file mode: pad the start with BUFFER_SECONDS
of silence, then for each timestamp t hand get_audio_data the BUFFER_SECONDS of audio that END at t.
So a value logged at time t was computed from audio before t (it is "late" by up to the window length).
"""
import os
import sys
import time
import warnings

import numpy as np
import librosa
import parselmouth

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import main10morevocalinformation as m10  # noqa: E402  (your analysis code)

BUFFER_SECONDS = 2.0  # same as main11

# Every continuous value get_audio_data returns that the benchmark tracks
YOUR_FEATURES = ["pitch", "F1", "F2", "F3", "F4", "F5", "vocalweight",
                 "spectral_slope_base", "spectral_slope_derived",
                 "spectral_centroid_base", "spectral_centroid_derived",
                 "harmonic_rolloff", "noise_rolloff"]


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return np.nan_to_num(y).astype(np.float32), sr


def noise_profile_like_main11(y, sr):
    """main11 averages the FFT power of the first 2 s in 100 ms chunks (passed to get_audio_data)."""
    chunk = int(sr * 0.100)
    cal = y[:min(int(sr * 2.0), len(y))]
    if len(cal) < chunk:
        cal = np.pad(cal, (0, chunk - len(cal)))
    window = np.hanning(chunk)
    n = len(cal) // chunk
    prof = np.zeros(chunk // 2 + 1)
    for i in range(n):
        prof += np.abs(np.fft.rfft(cal[i * chunk:(i + 1) * chunk] * window)) ** 2
    return prof / n


def _window_ending_at(padded, sr, t, buf_n):
    head = buf_n + int(round(t * sr))
    seg = padded[max(0, head - buf_n):head]
    if len(seg) < buf_n:  # asked for a time past the end of the file: pad with silence
        seg = np.pad(seg, (0, buf_n - len(seg)))
    return seg


def run_yours(y, sr, end_times, analyse=None, gates=False):
    """Runs get_audio_data with windows ending at each time in end_times.
    Returns {feature: np.array}, plus 'ms_per_frame'. With gates=True it also records the two voicing
    gates get_pitch uses (RMS of the last 50 ms and its zero-crossing rate) so misses can be explained."""
    analyse = analyse or m10.get_audio_data
    buf_n = int(sr * BUFFER_SECONDS)
    padded = np.pad(y, (buf_n, 0))
    prof = noise_profile_like_main11(y, sr)
    out = {k: np.zeros(len(end_times)) for k in YOUR_FEATURES}
    out["ms_per_frame"] = np.zeros(len(end_times))
    if gates:
        out["gate_rms"] = np.zeros(len(end_times))
        out["gate_zcr"] = np.zeros(len(end_times))
    for i, t in enumerate(end_times):
        buf = _window_ending_at(padded, sr, t, buf_n)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = analyse(buf, BUFFER_SECONDS, sr, prof)
        out["ms_per_frame"][i] = 1000 * (time.perf_counter() - t0)
        out["pitch"][i] = d["pitch"]
        fm = list(d["formants"]) + [0.0] * 5
        for j in range(5):
            out[f"F{j + 1}"][i] = fm[j]
        for k in YOUR_FEATURES[6:]:
            out[k][i] = d.get(k, np.nan)
        if gates:
            recent = buf[-int(sr * 0.050):]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out["gate_rms"][i] = np.mean(librosa.feature.rms(y=recent))
                out["gate_zcr"][i] = np.mean(librosa.feature.zero_crossing_rate(y=recent))
    return out


# ------------------------------------------------------------------------------------------------
# Reference tools (whole-file, frames centred on their timestamps)
# ------------------------------------------------------------------------------------------------
def praat_pitch(y, sr, times, floor=75.0, ceiling=600.0):
    """Praat's standard autocorrelation pitch tracker (Boersma 1993), Praat default settings.
    Returns f0 at each time (0 = unvoiced)."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f = p.selected_array["frequency"]
    xs = p.xs()
    idx = np.clip(np.round((np.asarray(times) - xs[0]) / 0.01).astype(int), 0, len(f) - 1)
    res = f[idx].copy()
    res[(np.asarray(times) < xs[0] - 0.01) | (np.asarray(times) > xs[-1] + 0.01)] = 0
    return res


def pyin_pitch(y, sr, times, fmin=60.0, fmax=800.0):
    """librosa's probabilistic YIN (Mauch & Dixon 2014), the slower but more robust cousin of the
    plain YIN your get_pitch uses. Run at 16 kHz with 10 ms hops. Returns f0 (0 = unvoiced)."""
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, vflag, _ = librosa.pyin(y16, fmin=fmin, fmax=fmax, sr=16000, frame_length=1024, hop_length=160)
    f0 = np.where(vflag & np.isfinite(f0), f0, 0.0)
    idx = np.clip(np.round(np.asarray(times) / 0.01).astype(int), 0, len(f0) - 1)
    return f0[idx]


def praat_formants(y, sr, times, ceiling=5500.0, n_formants=4):
    """Praat Burg formants on the whole file. Returns array (len(times), n_formants), 0 = not found."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling)
    res = np.zeros((len(times), n_formants))
    for i, t in enumerate(times):
        for j in range(n_formants):
            v = fm.get_value_at_time(j + 1, float(t))
            res[i, j] = 0.0 if np.isnan(v) else v
    return res


# ------------------------------------------------------------------------------------------------
# Candidate fixes, tested on the same data so the report's suggestions come with evidence
# ------------------------------------------------------------------------------------------------
def candidate_pitch_praat_window(buf, sr, window_s=0.100, floor=60.0, ceiling=1000.0):
    """Real-time-friendly alternative to get_pitch: Praat autocorrelation on only the last `window_s`
    seconds (same audio the formant code already uses). Value taken from the newest analysable frame."""
    seg = buf[-int(sr * window_s):].astype(np.float64)
    if np.max(np.abs(seg)) < 1e-4:
        return -1.0
    snd = parselmouth.Sound(seg, sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f = p.selected_array["frequency"]
    voiced = f[f > 0]
    if len(f) == 0 or f[-1] <= 0:
        return float(np.median(voiced)) if len(voiced) >= max(1, len(f) // 2) else -1.0
    return float(f[-1])


def adaptive_ceiling(f0):
    """Formant ceiling picked from pitch (Praat's advice: ~5000 Hz for adult male voices,
    ~5500 Hz for adult female voices, higher for children, whose vocal tracts are shorter)."""
    if f0 <= 0:
        return 5500.0
    if f0 < 155:
        return 5000.0
    if f0 < 235:
        return 5500.0
    return 6500.0


def candidate_formants_adaptive(buf, sr, f0, window_s=0.100):
    """Same as get_formants_praat (Burg on the last 100 ms, value at the window middle) but with the
    ceiling chosen from the pitch instead of always 5500 Hz."""
    seg = buf[-int(sr * window_s):].astype(np.float64)
    snd = parselmouth.Sound(seg, sampling_frequency=sr)
    fm = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=adaptive_ceiling(f0))
    mid = snd.get_total_duration() / 2
    vals = [fm.get_value_at_time(j, mid) for j in range(1, 5)]
    return [0.0 if np.isnan(v) else v for v in vals]


def run_candidates(y, sr, end_times, pitch_only=False):
    """Runs both candidate fixes on windows ending at end_times. Returns {'pitch', 'F1'..'F4'}."""
    buf_n = int(sr * BUFFER_SECONDS)
    padded = np.pad(y, (buf_n, 0))
    out = {"pitch": np.zeros(len(end_times))}
    for j in range(4):
        out[f"F{j + 1}"] = np.zeros(len(end_times))
    for i, t in enumerate(end_times):
        buf = _window_ending_at(padded, sr, t, buf_n)
        f0 = candidate_pitch_praat_window(buf, sr)
        out["pitch"][i] = f0
        if f0 > 0 and not pitch_only:
            fm = candidate_formants_adaptive(buf, sr, f0)
            for j in range(4):
                out[f"F{j + 1}"][i] = fm[j]
    return out
