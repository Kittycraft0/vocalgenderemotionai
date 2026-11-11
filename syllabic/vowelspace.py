"""Your vowel space (running per-vowel averages) and target vowel spaces to compare it with.

Running averages are EXPONENTIALLY WEIGHTED: each new token of a vowel moves that vowel's average by a
fraction, so a vowel's point drifts as your pronunciation changes instead of being frozen by everything
you said in the past. `half_life` = after this many new tokens of a vowel, older tokens count half.

Targets are JSON files: {"name", "source", "vowels": {"IY": {"F1","F2","F3","F4","sd":[...],"n"}, ...}}.
Built in: Hillenbrand et al. (1995) group averages (men, women, boys, girls), either their published
measurements or the same recordings measured by THIS pipeline (same nucleus detection, same Burg
settings), which is the fairer target because both sides are measured the same way.
Make one from a recording of the voice you want: main14livesyllables.py --make-target voice.wav
"""
import csv
import json
import os

import numpy as np

from .pronounce import CHART_VOWELS, HILLENBRAND_CODE

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(HERE, "data", "hillenbrand_targets.json")
USER_TARGET_DIR = os.path.join(os.path.dirname(HERE), "vowel_targets")

# A vowel's average counts as settled from this many tokens on. Replaying the first live sessions, an
# average of 1-3 tokens sat a median ~0.8 Bark (1 in 10: 1.3-1.7 Bark) from where that vowel later
# settled, about the distance between neighbouring vowels; at 5 tokens 0.56, at 10 0.39.
SETTLED_N = 5

TOKEN_FIELDS = ["t", "utt", "word", "word_prob", "syllable", "vowel", "ipa", "stress", "chart", "reason",
                "F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "f0", "ceiling", "peak_db", "periodicity",
                "vtl_cm", "outlier"]


class VowelSpace:
    def __init__(self, half_life=25.0, gate=16.0):
        self.alpha = 1.0 - 0.5 ** (1.0 / half_life)
        self.gate = gate       # squared Mahalanobis distance beyond which a token is treated as an outlier
        self.stats = {}        # vowel -> {"n", "mean" [F1..F4], "cov" 2x2 over (F1, F2), "vtl"}

    def reset(self):
        self.stats = {}

    def add(self, tok):
        """Adds a chart token; returns False if it was rejected as an outlier (probably a formant error)."""
        v = tok["vowel"]
        x = np.array([tok["F1"], tok["F2"], tok["F3"], tok["F4"]], dtype=float)
        if x[0] <= 0 or x[1] <= 0:
            return False
        st = self.stats.get(v)
        if st is None:
            self.stats[v] = {"n": 1, "mean": x.copy(), "cov": np.diag([90.0 ** 2, 220.0 ** 2]),
                             "vtl": tok.get("vtl_cm") or 0.0}
            return True
        d = x[:2] - st["mean"][:2]
        if st["n"] >= 5:
            m2 = float(d @ np.linalg.solve(st["cov"] + np.eye(2) * 1e-6, d))
            if m2 > self.gate:
                return False
        a = max(self.alpha, 1.0 / (st["n"] + 1))   # plain average for the first tokens, then exponential
        old = st["mean"].copy()
        for i in range(4):
            if x[i] > 0:
                st["mean"][i] = old[i] + a * (x[i] - old[i])
        st["cov"] = (1 - a) * (st["cov"] + a * np.outer(d, d))
        if tok.get("vtl_cm"):
            st["vtl"] = st["vtl"] + a * (tok["vtl_cm"] - st["vtl"]) if st["vtl"] else tok["vtl_cm"]
        st["n"] += 1
        return True

    def means(self):
        return {v: {"F1": s["mean"][0], "F2": s["mean"][1], "F3": s["mean"][2], "F4": s["mean"][3],
                    "cov": s["cov"], "n": s["n"], "vtl": s["vtl"]} for v, s in self.stats.items()}


# --------------------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------------------
def target_from_tokens(tokens, name, source=""):
    """Per-vowel medians (robust to the odd formant error) from chart tokens."""
    by = {}
    for t in tokens:
        if t.get("chart") in (True, "True", "1", 1) and float(t["F1"]) > 0 and float(t["F2"]) > 0:
            by.setdefault(t["vowel"], []).append([float(t[f]) for f in ("F1", "F2", "F3", "F4")])
    vowels = {}
    for v, rows in by.items():
        a = np.array(rows)
        med = [float(np.median(c[c > 0])) if np.any(c > 0) else 0.0 for c in a.T]
        sd = [float(np.std(c[c > 0])) if np.sum(c > 0) > 1 else 0.0 for c in a.T]
        vowels[v] = {"F1": med[0], "F2": med[1], "F3": med[2], "F4": med[3], "sd": sd, "n": len(rows)}
    return {"name": name, "source": source, "vowels": vowels}


def hillenbrand_published(vowdata_path):
    """Group averages of Hillenbrand et al.'s own steady-state measurements (vowdata.dat)."""
    groups = {"m": "men", "w": "women", "b": "boys", "g": "girls"}
    rows = {}
    for line in open(vowdata_path):
        parts = line.split()
        if len(parts) != 16 or len(parts[0]) != 5 or parts[0][0] not in groups:
            continue
        v = HILLENBRAND_CODE.get(parts[0][3:5])
        F = [float(x) for x in parts[3:7]]
        if v and F[0] > 0 and F[1] > 0:
            rows.setdefault(groups[parts[0][0]], {}).setdefault(v, []).append(F)
    out = {}
    for g, by in rows.items():
        vowels = {}
        for v, fs in by.items():
            a = np.array(fs)
            vowels[v] = {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "sd": [0, 0, 0, 0], "n": len(fs)}
            for i, f in enumerate(("F1", "F2", "F3", "F4")):
                c = a[:, i][a[:, i] > 0]
                vowels[v][f] = float(np.mean(c)) if len(c) else 0.0
                vowels[v]["sd"][i] = float(np.std(c)) if len(c) > 1 else 0.0
        out[f"hillenbrand_{g}"] = {"name": f"Hillenbrand 1995 {g} (published)", "source": "Hillenbrand et al. 1995, "
                                   "J. Acoust. Soc. Am. 97(5), vowdata.dat group means", "vowels": vowels}
    return out


def builtin_targets():
    if os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def list_targets():
    """[(key, display name)] of built-in targets and the JSON files in vowel_targets/."""
    out = [(k, v["name"]) for k, v in builtin_targets().items()]
    if os.path.isdir(USER_TARGET_DIR):
        for fn in sorted(os.listdir(USER_TARGET_DIR)):
            if fn.lower().endswith(".json"):
                try:
                    with open(os.path.join(USER_TARGET_DIR, fn), encoding="utf-8") as f:
                        out.append((os.path.join(USER_TARGET_DIR, fn), json.load(f).get("name", fn)))
                except Exception:
                    pass
    return out


def load_target(key):
    """Built-in key (e.g. 'hillenbrand_women_measured') or a path to a target JSON."""
    if not key:
        return None
    b = builtin_targets()
    if key in b:
        return b[key]
    if os.path.exists(key):
        with open(key, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No target '{key}'. Built-in ones: {', '.join(b)}")


def save_target(target, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(target, f, indent=1, ensure_ascii=False)
    return path


def save_tokens(tokens, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TOKEN_FIELDS, extrasaction="ignore")
        w.writeheader()
        for t in tokens:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})


def load_tokens(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ("F1", "F2", "F3", "F4", "vtl_cm", "t"):
                row[k] = float(row.get(k) or 0)
            row["chart"] = row.get("chart") in ("True", "1", "true")
            out.append(row)
    return out


def gender_position(means, men, women, min_n=SETTLED_N, formants=("F1", "F2", "F3")):
    """Where your vowels sit on the lines from a men's reference to a women's, vowel by vowel.

    Worked in log frequency, because a shorter vocal tract raises the formants by a RATIO (Nearey 1978):
    for each vowel v, with M, W and U its log-formant vectors (men, women, you) and d = W - M,
        per_vowel[v] = (U - M) . d / |d|^2          0 = the men's average, 1 = the women's, >1 beyond it
        position     = sum_v (U - M) . d / sum_v |d|^2   (the least-squares single value for all vowels)
        off          = RMS over vowels and formants of U - M - per_vowel[v] * d, i.e. how far your vowels
                       sit to the side of their lines (log units, 0.05 ~ 5%): a pattern of your own
                       rather than more or less of the men-women difference.
    Only vowels with at least min_n tokens count. This is a description of formants, not a measure of how
    a voice is heard: pitch, voice quality and intonation weigh in too (Hillenbrand & Clark 2009)."""
    num = den = 0.0
    per, res = {}, []
    for v, m in means.items():
        if m.get("n", 0) < min_n or v not in men or v not in women:
            continue
        u = np.array([m.get(f, 0.0) for f in formants], dtype=float)
        a = np.array([men[v].get(f, 0.0) for f in formants], dtype=float)
        b = np.array([women[v].get(f, 0.0) for f in formants], dtype=float)
        if min(u.min(), a.min(), b.min()) <= 0:
            continue
        u, a, b = np.log(u), np.log(a), np.log(b)
        d = b - a
        dd = float(d @ d)
        if dd < 1e-6:
            continue
        lam = float((u - a) @ d) / dd
        per[v] = lam
        num += float((u - a) @ d)
        den += dd
        res.append(u - a - lam * d)
    if not per:
        return {"position": None, "off": None, "n_vowels": 0, "per_vowel": {}}
    return {"position": num / den, "off": float(np.sqrt(np.mean(np.square(res)))), "n_vowels": len(per),
            "per_vowel": per}


def chart_order():
    """Vowels in order around the edge of the vowel space (for drawing its outline)."""
    return [v for v in ("IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW") if v in CHART_VOWELS]
