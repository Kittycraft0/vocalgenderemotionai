"""Downloading and loading the reference datasets the benchmark compares against.

Datasets
--------
* PTDB-TUG (Graz University of Technology): read speech from 10 male + 10 female speakers with a
  laryngograph (throat sensor) recorded at the same time. The reference pitch comes from the
  laryngograph, so it is close to the "true" vocal-fold rate. Open Database License.
  https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html
  Only a subset is downloaded (N sentences per speaker) because the full set is 3.9 GB.
* Hillenbrand et al. (1995): 12 American English vowels from men, women and children, with hand-checked
  F0 and formant measurements at the vowel's steady state. MIT license (hosted with permission).
  https://github.com/santiagobarreda/hillenbrand_et_al_1995
* Synthetic: vowels generated here with exactly known pitch, formants, spectral tilt and timing.

Downloaded files only ever get read as audio or text; nothing in the data folder is executed.
"""
import io
import os
import random
import re
import zipfile

import numpy as np
import requests
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(HERE), "benchmark_data")

PTDB_BASE = "http://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH%20DATA"
HILLENBRAND_URL = "https://github.com/santiagobarreda/hillenbrand_et_al_1995/raw/main/h95-alldata.zip"


# ----------------------------------------------------------------------------------------------
# small download helpers
# ----------------------------------------------------------------------------------------------
def _download(url, dest, timeout=120):
    """Downloads url to dest (skips if it already exists). Writes to a temp name first so a broken
    download never leaves a half-written file that looks finished."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def _safe_extract(zf, target_dir, only_suffix=None):
    """Extracts a zip without allowing paths that escape target_dir ("zip slip")."""
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for member in zf.infolist():
        if member.is_dir():
            continue
        name = os.path.basename(member.filename)
        if not name or (only_suffix and not name.lower().endswith(only_suffix)):
            continue
        out = os.path.abspath(os.path.join(target_dir, name))
        if not out.startswith(target_dir + os.sep):
            continue
        with zf.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())


def _list_index(url):
    """Returns the file/folder names in an Apache directory listing."""
    html = requests.get(url, timeout=60).text
    names = re.findall(r'href="([^"?/][^"?]*)"', html)
    return [n for n in names if not n.startswith("http")]


# ----------------------------------------------------------------------------------------------
# PTDB-TUG
# ----------------------------------------------------------------------------------------------
def ptdb_dir():
    return os.path.join(DATA_ROOT, "ptdb_tug")


def download_ptdb(per_speaker=8, seed=0, log=print):
    """Downloads `per_speaker` random sentences (microphone wav + reference pitch) for each of the
    20 speakers. The same seed always picks the same sentences, so results are comparable run to run."""
    rng = random.Random(seed)
    for gender, prefix in (("FEMALE", "F"), ("MALE", "M")):
        for i in range(1, 11):
            spk = f"{prefix}{i:02d}"
            mic_url = f"{PTDB_BASE}/{gender}/MIC/{spk}/"
            wavs = sorted(n for n in _list_index(mic_url) if n.endswith(".wav"))
            chosen = rng.sample(wavs, min(per_speaker, len(wavs)))
            for wav in chosen:
                utt = wav[len("mic_"):-len(".wav")]
                _download(mic_url + wav, os.path.join(ptdb_dir(), "MIC", spk, wav))
                _download(f"{PTDB_BASE}/{gender}/REF/{spk}/ref_{utt}.f0",
                          os.path.join(ptdb_dir(), "REF", spk, f"ref_{utt}.f0"))
            log(f"  PTDB-TUG {spk}: {len(chosen)} sentences")


def load_ptdb(max_per_speaker=None):
    """Returns a list of items: {name, path, speaker, group ('male'/'female'), ref_f0, ref_voiced}.
    Reference files have 4 columns per 10 ms frame (RAPT output on the laryngograph signal):
    F0 in Hz, voicing flag, RMS energy, cross-correlation peak."""
    items = []
    mic_root = os.path.join(ptdb_dir(), "MIC")
    if not os.path.isdir(mic_root):
        return items
    for spk in sorted(os.listdir(mic_root)):
        wavs = sorted(f for f in os.listdir(os.path.join(mic_root, spk)) if f.endswith(".wav"))
        if max_per_speaker:
            wavs = wavs[:max_per_speaker]
        for wav in wavs:
            utt = wav[len("mic_"):-len(".wav")]
            ref_path = os.path.join(ptdb_dir(), "REF", spk, f"ref_{utt}.f0")
            if not os.path.exists(ref_path):
                continue
            ref = np.loadtxt(ref_path, ndmin=2)
            items.append({
                "name": utt,
                "path": os.path.join(mic_root, spk, wav),
                "speaker": spk,
                "group": "female" if spk.startswith("F") else "male",
                "ref_f0": ref[:, 0],
                "ref_voiced": (ref[:, 1] > 0.5) & (ref[:, 0] > 0),
                "ref_hop_s": 0.010,
            })
    return items


# ----------------------------------------------------------------------------------------------
# Hillenbrand 1995
# ----------------------------------------------------------------------------------------------
def hillenbrand_dir():
    return os.path.join(DATA_ROOT, "hillenbrand")


def download_hillenbrand(log=print):
    root = hillenbrand_dir()
    zpath = _download(HILLENBRAND_URL, os.path.join(root, "h95-alldata.zip"))
    with zipfile.ZipFile(zpath) as outer:
        names = outer.namelist()
        for n in names:
            base = os.path.basename(n)
            if base.endswith(".dat") or base.endswith(".ds") or base.lower().startswith("readme"):
                with outer.open(n) as src, open(os.path.join(root, base), "wb") as dst:
                    dst.write(src.read())
        for n in names:
            if os.path.basename(n) in ("men.zip", "women.zip", "kids.zip"):
                inner = zipfile.ZipFile(io.BytesIO(outer.read(n)))
                _safe_extract(inner, os.path.join(root, "wav"), only_suffix=".wav")
    log(f"  Hillenbrand: {len(os.listdir(os.path.join(root, 'wav')))} wav files")


HILLENBRAND_GROUPS = {"m": "men", "w": "women", "b": "boys", "g": "girls"}


def load_hillenbrand(limit=None, seed=0):
    """Returns items: {name, path, group, vowel, f0, formants [F1..F4] at steady state (0 = not measurable),
    t_ss (steady-state time, s), t_start, t_end, f_20/f_50/f_80 ([F1..F3] at 20/50/80% of the vowel)}."""
    root = hillenbrand_dir()
    vow_path, time_path = os.path.join(root, "vowdata.dat"), os.path.join(root, "timedata.dat")
    if not (os.path.exists(vow_path) and os.path.exists(time_path)):
        return []
    times = {}
    for line in open(time_path):
        parts = line.split()
        if len(parts) == 5 and re.fullmatch(r"[mwbg]\d\d[a-z]{2}", parts[0]):
            start, end, c1, c2 = map(float, parts[1:])
            times[parts[0]] = (start / 1000, end / 1000, (c1 + c2) / 2000)
    items = []
    for line in open(vow_path):
        parts = line.split()
        if len(parts) != 16 or not re.fullmatch(r"[mwbg]\d\d[a-z]{2}", parts[0]):
            continue
        name = parts[0]
        wav = os.path.join(root, "wav", name + ".wav")
        if name not in times or not os.path.exists(wav):
            continue
        v = [float(x) for x in parts[1:]]
        start, end, t_ss = times[name]
        items.append({
            "name": name, "path": wav, "group": HILLENBRAND_GROUPS[name[0]], "vowel": name[3:5],
            "f0": v[1], "formants": v[2:6], "t_ss": t_ss, "t_start": start, "t_end": end,
            "f_20": v[6:9], "f_50": v[9:12], "f_80": v[12:15],
        })
    if limit and limit < len(items):
        items = sorted(random.Random(seed).sample(items, limit), key=lambda it: it["name"])
    return items


# ----------------------------------------------------------------------------------------------
# Synthetic vowels with exactly known parameters
# ----------------------------------------------------------------------------------------------
# Typical formant targets (Hz) and bandwidths, roughly from Peterson & Barney / Hillenbrand averages.
SYNTH_VOWELS = {
    "male":   {"a": [730, 1090, 2440, 3400], "i": [270, 2290, 3010, 3700], "u": [300, 870, 2240, 3300]},
    "female": {"a": [850, 1220, 2810, 3900], "i": [310, 2790, 3310, 4200], "u": [370, 950, 2670, 3800]},
    "child":  {"a": [1030, 1370, 3170, 4300], "i": [370, 3200, 3730, 4600], "u": [430, 1170, 3260, 4300]},
}
SYNTH_BANDWIDTHS = [80, 100, 150, 200]


def glottal_source(f0, dur, sr, tilt_db_per_oct=-12.0, jitter=0.0, breath=0.0, seed=0):
    """Harmonic glottal source built directly in the frequency domain so the spectral tilt is exact:
    harmonic k has amplitude k ** (tilt / 6.02). f0 may be a number or an array (one value per sample).
    `breath` mixes in high-passed noise (0..1) to make a breathier voice."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    f0_arr = np.full(n, float(f0)) if np.isscalar(f0) else np.asarray(f0, dtype=float)
    if jitter:
        f0_arr = f0_arr * (1 + jitter * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)))
    phase = 2 * np.pi * np.cumsum(f0_arr) / sr
    out = np.zeros(n)
    kmax = int((sr / 2 - 200) / max(np.min(f0_arr), 40))
    exponent = tilt_db_per_oct / 6.0206
    for k in range(1, kmax + 1):
        alive = (k * f0_arr) < (sr / 2 - 200)  # never put a harmonic above Nyquist (no aliasing)
        out += alive * (k ** exponent) * np.sin(k * phase)
    if breath > 0:
        noise = rng.standard_normal(n)
        noise = np.diff(noise, prepend=0.0)  # tilt the noise upward a bit, like aspiration
        out = out / (np.std(out) + 1e-12) + breath * noise / (np.std(noise) + 1e-12)
    return out


def formant_filter(x, sr, formants, bandwidths=SYNTH_BANDWIDTHS):
    """Cascade of 2nd-order resonators (Klatt style), each with unity gain at 0 Hz."""
    from scipy.signal import lfilter
    y = x.copy()
    for f, bw in zip(formants, bandwidths):
        if f <= 0 or f >= sr / 2:
            continue
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        b = [sum(a)]
        y = lfilter(b, a, y)
    return y


def synth_vowel(f0, formants, sr=48000, dur=1.0, tilt=-12.0, breath=0.0, gain=0.3, seed=0, lead_silence=0.0):
    src = glottal_source(f0, dur, sr, tilt_db_per_oct=tilt, breath=breath, seed=seed)
    y = formant_filter(src, sr, formants)
    fade = int(0.01 * sr)
    env = np.ones_like(y)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    y = y * env
    y = gain * y / (np.max(np.abs(y)) + 1e-12)
    if lead_silence:
        y = np.concatenate([np.zeros(int(lead_silence * sr)), y])
    return y.astype(np.float32)


def synthetic_dir():
    return os.path.join(DATA_ROOT, "synthetic")


def write_wav(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)
    return path


# ----------------------------------------------------------------------------------------------
# KEELE Pitch Database
# ----------------------------------------------------------------------------------------------
def keele_dir():
    return os.path.join(DATA_ROOT, "keele", "KEELE")


def load_keele():
    """
    Loads speakers from the KEELE pitch database.
    Each item contains:
      - name: speaker ID (e.g. 'f1nw0000')
      - path: path to speech audio WAV (20 kHz)
      - group: 'female' or 'male'
      - ref_f0: ground-truth pitch from throat sensor (laryngograph) in Hz
      - ref_times: timestamps of the reference frames (s)
      - ref_voiced: boolean array of voicing
      - ref_hop_s: 0.010 s (10ms frame rate)
    """
    root = keele_dir()
    if not os.path.isdir(root):
        return []

    items = []
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        wav_path = os.path.join(spk_dir, "signal.wav")
        pitch_path = os.path.join(spk_dir, "pitch.npy")
        if not (os.path.exists(wav_path) and os.path.exists(pitch_path)):
            continue

        arr = np.load(pitch_path)
        f0 = arr["pitch"].astype(np.float32)
        times = arr["time"].astype(np.float32)
        voiced = f0 > 0.0

        items.append({
            "name": spk,
            "path": wav_path,
            "speaker": spk[:2],
            "group": "female" if spk.lower().startswith("f") else "male",
            "ref_f0": f0,
            "ref_times": times,
            "ref_voiced": voiced,
            "ref_hop_s": 0.010,
        })
    return items


# ----------------------------------------------------------------------------------------------
# Peterson & Barney (1952) Vowel Database
# ----------------------------------------------------------------------------------------------
def peterson_barney_dir():
    return os.path.join(DATA_ROOT, "peterson_barney")


def load_peterson_barney():
    """
    Loads tabular ground-truth vowel formant measurements from Peterson & Barney (1952).
    1520 tokens across 76 speakers (men, women, children).
    Returns list of items with F0, F1, F2, F3 and vowel labels.
    """
    tsv_path = os.path.join(peterson_barney_dir(), "pb52.tsv")
    if not os.path.exists(tsv_path):
        return []

    items = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().replace('"', '').split("\t")
        for line in f:
            parts = line.strip().replace('"', '').split("\t")
            if len(parts) >= 9:
                try:
                    items.append({
                        "type": parts[0],
                        "sex": parts[1],
                        "speaker": int(parts[2]),
                        "vowel": parts[3],
                        "rep": int(parts[4]),
                        "f0": float(parts[5]),
                        "f1": float(parts[6]),
                        "f2": float(parts[7]),
                        "f3": float(parts[8]),
                    })
                except ValueError:
                    continue
    return items


# ----------------------------------------------------------------------------------------------
# CSTR Edinburgh FDA Pitch Evaluation Database
# ----------------------------------------------------------------------------------------------
def fda_dir():
    return os.path.join(DATA_ROOT, "fda")


def load_fda(max_per_speaker=10):
    """
    Loads sentences from the CSTR Edinburgh FDA pitch evaluation database
    (male speaker 'rl' and female speaker 'sb').
    Returns items with path, ref_f0, ref_times, ref_voiced at 10ms frame rate.
    """
    root = fda_dir()
    wav_dir = os.path.join(root, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    items = []
    for spk, grp in [("rl", "male"), ("sb", "female")]:
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue

        sigs = [f for f in sorted(os.listdir(spk_dir)) if f.endswith(".sig")]
        if max_per_speaker:
            sigs = sigs[:max_per_speaker]

        for sig_file in sigs:
            base = os.path.splitext(sig_file)[0]
            sig_path = os.path.join(spk_dir, sig_file)
            fx_path = os.path.join(spk_dir, base + ".fx")
            wav_path = os.path.join(wav_dir, base + ".wav")

            if not os.path.exists(fx_path):
                continue

            # Convert 20kHz 16-bit big-endian PCM to WAV if needed
            if not os.path.exists(wav_path):
                raw = open(sig_path, "rb").read()
                audio = np.frombuffer(raw, dtype=">i2").astype(np.float32) / 32768.0
                sf.write(wav_path, audio, 20000)
            else:
                info = sf.info(wav_path)
                dur = info.duration

            # Parse reference .fx pitch contour
            with open(fx_path, "r", encoding="latin1") as f:
                text = f.read()

            parts = text.split("\x0c")
            data_text = parts[1] if len(parts) > 1 else parts[0]
            segments = []
            curr_seg = []
            for line in data_text.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "=":
                    if curr_seg:
                        segments.append(curr_seg)
                        curr_seg = []
                    continue
                toks = line.split()
                if len(toks) == 2:
                    try:
                        curr_seg.append((float(toks[0]) / 1000.0, float(toks[1])))
                    except ValueError:
                        pass
            if curr_seg:
                segments.append(curr_seg)

            info = sf.info(wav_path)
            dur = info.duration
            hop = 0.010
            grid = np.arange(0, dur, hop, dtype=np.float32)
            ref_f0 = np.zeros(len(grid), dtype=np.float32)

            for seg in segments:
                if len(seg) >= 2:
                    st = [p[0] for p in seg]
                    sf0 = [p[1] for p in seg]
                    mask = (grid >= st[0]) & (grid <= st[-1])
                    if np.any(mask):
                        ref_f0[mask] = np.interp(grid[mask], st, sf0)
                elif len(seg) == 1:
                    idx = np.argmin(np.abs(grid - seg[0][0]))
                    if abs(grid[idx] - seg[0][0]) < hop:
                        ref_f0[idx] = seg[0][1]

            items.append({
                "name": base,
                "path": wav_path,
                "speaker": spk,
                "group": grp,
                "ref_f0": ref_f0,
                "ref_times": grid,
                "ref_voiced": ref_f0 > 0.0,
                "ref_hop_s": hop,
            })

    return items


# ----------------------------------------------------------------------------------------------
# Vocobox Human Voice Singing Dataset (GitHub: vocobox/human-voice-dataset)
# ----------------------------------------------------------------------------------------------
VOCOBOX_URL_BASE = "https://raw.githubusercontent.com/vocobox/human-voice-dataset/master/data/voices/martin/notes/exports/mono"
VOCOBOX_NOTE_HZ = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81,
    "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00
}


def vocobox_dir():
    return os.path.join(DATA_ROOT, "vocobox")


def load_vocobox(limit=14):
    """
    Downloads and loads human singing voice recordings from the Vocobox dataset
    with exact musical note ground-truth fundamental frequencies.
    """
    root = vocobox_dir()
    os.makedirs(root, exist_ok=True)

    target_notes = ["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"]
    target_notes = target_notes[:limit]

    items = []
    for note in target_notes:
        fname = f"{note}.wav"
        dest = os.path.join(root, fname)
        url = f"{VOCOBOX_URL_BASE}/{fname}"
        try:
            _download(url, dest, timeout=15)
        except Exception as e:
            continue

        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            continue

        info = sf.info(dest)
        dur = info.duration
        hop = 0.010
        grid = np.arange(0.0, dur, hop, dtype=np.float32)
        ref_note_f0 = VOCOBOX_NOTE_HZ.get(note, 220.0)
        # Pad initial attack and final release as lower energy
        ref_f0 = np.full(len(grid), ref_note_f0, dtype=np.float32)

        items.append({
            "name": f"vocobox_{note}",
            "path": dest,
            "speaker": "martin",
            "group": "male_singer",
            "ref_f0": ref_f0,
            "ref_times": grid,
            "ref_voiced": np.ones(len(grid), dtype=bool),
            "ref_hop_s": hop,
            "target_hz": ref_note_f0
        })

    return items


# ----------------------------------------------------------------------------------------------
# Dynamic Vowel Diphthong Glides (Continuous Formant Trajectories)
# ----------------------------------------------------------------------------------------------
def synth_diphthong_glide(f0=160.0, diphthong_type="a_to_i", sr=44100, dur=1.2, tilt=-12.0):
    """
    Synthesizes continuous human vowel diphthongs with exact analytical formant trajectories.
    Types:
      'a_to_i' (/aɪ/ as in 'fly'): F1 800 -> 280 Hz, F2 1200 -> 2300 Hz
      'o_to_i' (/ɔɪ/ as in 'boy'): F1 550 -> 280 Hz, F2 900 -> 2200 Hz
      'a_to_u' (/aʊ/ as in 'cow'): F1 800 -> 320 Hz, F2 1200 -> 800 Hz
    Returns (audio_samples, times_array, dict_of_ground_truth_formants).
    """
    diphthong_presets = {
        "a_to_i": {"f1": (800.0, 280.0), "f2": (1200.0, 2300.0), "f3": (2600.0, 2800.0), "f4": 3500.0},
        "o_to_i": {"f1": (550.0, 280.0), "f2": (900.0, 2200.0), "f3": (2500.0, 2800.0), "f4": 3500.0},
        "a_to_u": {"f1": (800.0, 320.0), "f2": (1200.0, 800.0), "f3": (2600.0, 2400.0), "f4": 3500.0}
    }
    cfg = diphthong_presets.get(diphthong_type, diphthong_presets["a_to_i"])
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)

    # Smooth S-curve glide from 20% to 80% of duration
    t_norm = np.clip((t - 0.20 * dur) / (0.60 * dur), 0.0, 1.0)
    s_curve = 0.5 * (1.0 - np.cos(np.pi * t_norm))

    f1_traj = cfg["f1"][0] + (cfg["f1"][1] - cfg["f1"][0]) * s_curve
    f2_traj = cfg["f2"][0] + (cfg["f2"][1] - cfg["f2"][0]) * s_curve
    f3_traj = cfg["f3"][0] + (cfg["f3"][1] - cfg["f3"][0]) * s_curve
    f4_const = cfg["f4"]

    # Generate glottal source
    from scipy import signal
    pulse_phase = 2.0 * np.pi * f0 * t
    source = np.sin(pulse_phase)
    # Apply spectral tilt
    if tilt < 0:
        b_tilt, a_tilt = signal.butter(1, 1000.0 / (sr / 2.0), btype='low')
        source = signal.lfilter(b_tilt, a_tilt, source)

    # Filter with time-varying formants via small 20ms blocks
    block_n = int(sr * 0.020)
    audio = np.zeros_like(source)
    zi_states = [np.zeros(2, dtype=np.float64) for _ in range(4)]

    for idx in range(0, len(source), block_n):
        seg = source[idx:idx + block_n]
        mid_idx = min(idx + block_n // 2, len(t) - 1)
        cur_f = [f1_traj[mid_idx], f2_traj[mid_idx], f3_traj[mid_idx], f4_const]
        cur_bw = [80.0, 110.0, 160.0, 250.0]

        sub_out = np.zeros_like(seg)
        for k in range(4):
            r = np.exp(-np.pi * cur_bw[k] / sr)
            theta = 2.0 * np.pi * cur_f[k] / sr
            b0 = 1.0 - r
            b = np.array([b0, 0.0, -b0], dtype=np.float64)
            a = np.array([1.0, -2.0 * r * np.cos(theta), r * r], dtype=np.float64)
            res, zi_states[k] = signal.lfilter(b, a, seg, zi=zi_states[k])
            sub_out += res

        audio[idx:idx + block_n] = sub_out

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.8
    ref_times = np.linspace(0.0, dur, len(audio))
    return audio.astype(np.float32), ref_times, {
        "F1": f1_traj,
        "F2": f2_traj,
        "F3": f3_traj,
        "F4": np.full(len(t), f4_const)
    }



