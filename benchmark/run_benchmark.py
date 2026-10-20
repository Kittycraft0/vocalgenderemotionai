"""Benchmarks the vocal analysis code (main10's get_audio_data, exactly as main11 logs it) against datasets
with known answers and against standard tools, then writes an HTML report of what works, what doesn't, and
what to try next.

Usage (from the vocalgenderemotionai folder):
    python benchmark/run_benchmark.py                 # full run (downloads datasets the first time)
    python benchmark/run_benchmark.py --quick         # smaller subset, a few minutes
    python benchmark/run_benchmark.py --workers 4     # limit CPU use

Results are cached per file in benchmark_data/cache, keyed on the source code of main10 and the benchmark,
so after you change your analysis code only the affected work is redone. Each run writes
benchmark_reports/<date_time>/report.html + metrics.json, and the report compares against the previous run.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_datasets as ds  # noqa: E402
import bench_metrics as bm  # noqa: E402

PROJECT_DIR = os.path.dirname(HERE)
CACHE_DIR = os.path.join(ds.DATA_ROOT, "cache")
REPORT_ROOT = os.path.join(PROJECT_DIR, "benchmark_reports")

PITCH_METHODS = ["yours", "candidate", "praat", "pyin"]
WINDOW_END_METHODS = {"yours", "candidate"}   # these log the END of their analysis window
CEILINGS = [5000.0, 5500.0, 6500.0, 8000.0]
SNRS = [20, 10, 5]


def _hash_files(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def analysis_version():
    """Changes only when your analysis code (main10) changes. Shown in the report."""
    return _hash_files([os.path.join(PROJECT_DIR, "main10morevocalinformation.py")])


def code_hash():
    """Cache key: analysis code + benchmark code, so cached results are redone whenever either changes."""
    return _hash_files([os.path.join(PROJECT_DIR, "main10morevocalinformation.py"), os.path.join(HERE, "run_benchmark.py"),
                        os.path.join(HERE, "bench_extractors.py"), os.path.join(HERE, "bench_datasets.py")])


# ================================================================================================
# Worker tasks (run in separate processes). Each returns a dict of numpy arrays / numbers.
# ================================================================================================
def _cached(key, fn):
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    res = fn()
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **res)
    try:
        os.replace(tmp, path)
    except OSError:  # another worker just wrote the same result; keep theirs
        if os.path.exists(tmp):
            os.remove(tmp)
    return res


def add_noise(y, snr_db, seed):
    """White noise at the given SNR, measured against the louder half of the 10 ms frames (the speech),
    so silences don't make the SNR look better than it is."""
    rng = np.random.default_rng(seed)
    hop = 160
    frames = y[: len(y) // hop * hop].reshape(-1, hop)
    p = (frames ** 2).mean(axis=1)
    speech_power = p[p >= np.median(p)].mean()
    noise = rng.standard_normal(len(y)) * np.sqrt(speech_power / 10 ** (snr_db / 10))
    return (y + noise).astype(np.float32)


def task_ptdb(path, n_ref, snr, chash):
    def work():
        import bench_extractors as ex
        y, sr = ex.load_audio(path)
        if snr is not None:
            y = add_noise(y, snr, seed=zlib.crc32(os.path.basename(path).encode()))
        grid = np.arange(n_ref + 26) * 0.01
        yours = ex.run_yours(y, sr, grid, gates=True)
        out = {"yours": yours["pitch"], "ms_per_frame": yours["ms_per_frame"],
               "candidate": ex.run_candidates(y, sr, grid, pitch_only=True)["pitch"],
               "praat": ex.praat_pitch(y, sr, grid), "pyin": ex.pyin_pitch(y, sr, grid)}
        out["gate_rms"], out["gate_zcr"] = yours["gate_rms"], yours["gate_zcr"]
        return out
    return _cached(f"ptdb|{path}|{snr}|{chash}", work)


def task_hillenbrand(item, chash):
    def work():
        import bench_extractors as ex
        y, sr = ex.load_audio(item["path"])
        t_ss, t0, t1 = item["t_ss"], item["t_start"], item["t_end"]
        t_pct = [t0 + p * (t1 - t0) for p in (0.2, 0.5, 0.8)]
        # windows ending at the steady-state time (as logged) and 50 ms later (window centred on it)
        ends = [t_ss, t_ss + 0.05] + [t + 0.05 for t in t_pct]
        yours = ex.run_yours(y, sr, ends)
        praat_times = [t_ss] + t_pct
        praat = np.stack([ex.praat_formants(y, sr, praat_times, ceiling=c) for c in CEILINGS])
        cand = ex.run_candidates(y, sr, [t_ss + 0.05])
        return {"yours_pitch": yours["pitch"], "yours_F": np.stack([yours[f"F{j}"] for j in range(1, 5)], axis=1),
                "praat_F": praat, "praat_pitch": ex.praat_pitch(y, sr, [t_ss]),
                "pyin_pitch": ex.pyin_pitch(y, sr, [t_ss]),
                "cand_pitch": cand["pitch"], "cand_F": np.stack([cand[f"F{j}"] for j in range(1, 5)], axis=1)}
    return _cached(f"h95|{item['name']}|{chash}", work)


def _voice_for_f0(f0):
    return "male" if f0 < 150 else ("female" if f0 < 260 else "child")


def task_synth(kind, params, chash):
    """Synthetic tests. Every signal = optional silence + a 1 s steady vowel with known parameters."""
    def work():
        import bench_extractors as ex
        p = dict(params)
        sr = p.get("sr", 48000)
        if kind == "pitch_range":
            f0 = p["f0"]
            y = ds.synth_vowel(f0, ds.SYNTH_VOWELS[_voice_for_f0(f0)]["a"], sr=sr, lead_silence=0.5)
            ends = np.arange(0.85, 1.45, 0.01)
            return {"yours": ex.run_yours(y, sr, ends)["pitch"],
                    "candidate": ex.run_candidates(y, sr, ends, pitch_only=True)["pitch"],
                    "praat": ex.praat_pitch(y, sr, ends - 0.05), "pyin": ex.pyin_pitch(y, sr, ends - 0.05)}
        if kind == "quality":
            voice = p.get("voice", "female")
            y = ds.synth_vowel(p.get("f0", 180), ds.SYNTH_VOWELS[voice][p.get("vowel", "a")], sr=sr,
                               tilt=p.get("tilt", -12.0), breath=p.get("breath", 0.0), gain=p.get("gain", 0.3),
                               lead_silence=0.5)
            ends = np.arange(0.85, 1.45, 0.01)
            yours = ex.run_yours(y, sr, ends)
            return {k: yours[k] for k in ex.YOUR_FEATURES}
        if kind == "timing":
            f0 = 180.0
            fm = ds.SYNTH_VOWELS["female"]["a"]
            y = ds.synth_vowel(f0, fm, sr=sr, dur=1.0, lead_silence=1.0)
            y = np.concatenate([y, np.zeros(sr, dtype=np.float32)])
            ends = np.arange(0.8, 2.4, 0.005)
            yours = ex.run_yours(y, sr, ends)
            return {"ends": ends, "yours_pitch": yours["pitch"], "yours_F1": yours["F1"],
                    "yours_F2": yours["F2"], "cand_pitch": ex.run_candidates(y, sr, ends, pitch_only=True)["pitch"],
                    "praat_pitch": ex.praat_pitch(y, sr, ends)}
        raise ValueError(kind)
    return _cached(f"synth|{kind}|{json.dumps(params, sort_keys=True)}|{chash}", work)


def _run_task(fn_name, args):
    try:
        return globals()[fn_name](*args), None
    except Exception:
        return None, traceback.format_exc()


def run_all(tasks, workers, label):
    """tasks: list of (key, fn_name, args). Runs them in parallel; returns {key: result}."""
    results, failures = {}, []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_task, fn, args): key for key, fn, args in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            res, err = fut.result()
            if err:
                failures.append((futs[fut], err))
            else:
                results[futs[fut]] = res
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                print(f"  {label}: {i}/{len(tasks)} done ({time.time() - t0:.0f}s)", flush=True)
    for key, err in failures[:3]:
        print(f"  FAILED {key}:\n{err}")
    if failures:
        print(f"  {len(failures)} {label} task(s) failed and were left out.")
    return results


# ================================================================================================
# Analysis
# ================================================================================================
def analyse_ptdb(items, clean, noisy):
    out = {"files": len(items), "speakers": len({it["speaker"] for it in items})}
    refs = {it["name"]: it for it in items if it["name"] in clean}
    # 1. find each method's best time alignment (lag) against the laryngograph reference
    lags, curves = {}, {}
    for m in PITCH_METHODS:
        rng = range(-5, 26) if m in WINDOW_END_METHODS else range(-10, 11)
        pairs = [(refs[n]["ref_f0"] * refs[n]["ref_voiced"], clean[n][m]) for n in refs]
        lags[m], curves[m] = bm.best_lag(pairs, rng)
    offset = lags["praat"]  # how the reference frames line up with centred analysis frames
    out["dataset_offset_frames"] = offset
    out["latency_ms"] = {m: 10 * (lags[m] - offset) for m in PITCH_METHODS}
    out["lag_curves"] = {m: {str(10 * (k - offset)): v for k, v in curves[m].items()} for m in PITCH_METHODS}

    # 2. scores. "yours_as_logged" = no time correction (the log timestamp taken at face value).
    variants = {m: lags[m] for m in PITCH_METHODS}
    variants["yours_as_logged"] = offset
    table, bins, speed = {}, {}, []
    for v, lag in variants.items():
        m = "yours" if v == "yours_as_logged" else v
        totals = {"all": None, "male": None, "female": None}
        all_ref, all_est = [], []
        for n, it in refs.items():
            ref = it["ref_f0"] * it["ref_voiced"]
            est = bm.shift(clean[n][m], lag, len(ref))
            c = bm.pitch_counts(ref, est)
            totals["all"] = bm.add_counts(totals["all"], c)
            totals[it["group"]] = bm.add_counts(totals[it["group"]], c)
            all_ref.append(ref)
            all_est.append(est)
        table[v] = {g: bm.pitch_summary(c) for g, c in totals.items() if c is not None}
        bins[v] = bm.by_f0_bin(np.concatenate(all_ref), np.concatenate(all_est))
    out["table"], out["f0_bins"] = table, bins

    # 3. why were voiced frames missed / unvoiced frames flagged? (your two gates in get_pitch)
    miss_rms = miss_zcr = miss_total = fa_total = fa_near = 0
    fa_rel_rms = []
    from scipy.ndimage import distance_transform_edt
    for n, it in refs.items():
        ref = it["ref_f0"] * it["ref_voiced"]
        est = bm.shift(clean[n]["yours"], lags["yours"], len(ref))
        rms = bm.shift(clean[n]["gate_rms"], lags["yours"], len(ref))
        zcr = bm.shift(clean[n]["gate_zcr"], lags["yours"], len(ref))
        miss = (ref > 0) & (est <= 0)
        miss_total += int(miss.sum())
        miss_rms += int((miss & (rms < 0.001)).sum())
        miss_zcr += int((miss & (rms >= 0.001) & (zcr > 0.15)).sum())
        fa = (ref <= 0) & (est > 0)
        fa_total += int(fa.sum())
        if fa.any() and (ref > 0).any():
            fa_near += int((distance_transform_edt(ref <= 0)[fa] <= 5).sum())  # within 50 ms of real voicing
            fa_rel_rms.append(rms[fa] / max(np.max(clean[n]["gate_rms"]), 1e-12))
        speed.append(clean[n]["ms_per_frame"])
    out["gates"] = {"missed_frames": miss_total, "by_rms_gate": miss_rms, "by_zcr_gate": miss_zcr,
                    "other": miss_total - miss_rms - miss_zcr, "false_alarms": fa_total,
                    "false_alarms_near_voicing": fa_near,
                    "false_alarm_median_db_below_peak": float(-20 * np.log10(np.median(np.concatenate(fa_rel_rms))))
                    if fa_rel_rms else float("nan")}
    sp = np.concatenate(speed)
    out["speed"] = {"median_ms": float(np.median(sp)), "p95_ms": float(np.percentile(sp, 95)),
                    "max_ms": float(np.max(sp)), "frames": int(len(sp))}

    # 4. noise robustness, each method at its own clean best lag
    noise = {}
    for snr in SNRS:
        noise[str(snr)] = {}
        for m in PITCH_METHODS:
            tot = None
            for n, it in refs.items():
                if (n, snr) not in noisy:
                    continue
                ref = it["ref_f0"] * it["ref_voiced"]
                tot = bm.add_counts(tot, bm.pitch_counts(ref, bm.shift(noisy[(n, snr)][m], lags[m], len(ref))))
            if tot is not None:
                noise[str(snr)][m] = bm.pitch_summary(tot)
    # same files, clean, for a fair baseline
    noise["clean"] = {}
    noisy_names = {n for (n, _s) in noisy}
    for m in PITCH_METHODS:
        tot = None
        for n in noisy_names & set(refs):
            ref = refs[n]["ref_f0"] * refs[n]["ref_voiced"]
            tot = bm.add_counts(tot, bm.pitch_counts(ref, bm.shift(clean[n][m], lags[m], len(ref))))
        if tot is not None:
            noise["clean"][m] = bm.pitch_summary(tot)
    out["noise"] = noise
    out["noise_files"] = len(noisy_names)

    # why yours loses frames in noise (same gate attribution as above)
    noise_gates = {}
    for snr in SNRS:
        mr = mz = mt = 0
        for n, it in refs.items():
            if (n, snr) not in noisy:
                continue
            r = noisy[(n, snr)]
            ref = it["ref_f0"] * it["ref_voiced"]
            est = bm.shift(r["yours"], lags["yours"], len(ref))
            rms = bm.shift(r["gate_rms"], lags["yours"], len(ref))
            zcr = bm.shift(r["gate_zcr"], lags["yours"], len(ref))
            miss = (ref > 0) & (est <= 0)
            mt += int(miss.sum())
            mr += int((miss & (rms < 0.001)).sum())
            mz += int((miss & (rms >= 0.001) & (zcr > 0.15)).sum())
        noise_gates[str(snr)] = {"missed_frames": mt, "by_rms_gate": mr, "by_zcr_gate": mz}
    out["noise_gates"] = noise_gates

    # what if the loudness gate adapted to each recording's noise floor (like a calibration would)?
    whatif = {}
    for label, k in [("current: fixed 0.001", 0.0), ("2x noise floor", 2.0), ("3x noise floor", 3.0),
                     ("5x noise floor", 5.0), ("8x noise floor", 8.0)]:
        tot = None
        for n, it in refs.items():
            ref = it["ref_f0"] * it["ref_voiced"]
            est = bm.shift(clean[n]["yours"], lags["yours"], len(ref))
            rms = bm.shift(clean[n]["gate_rms"], lags["yours"], len(ref))
            floor = np.percentile(clean[n]["gate_rms"][clean[n]["gate_rms"] > 0], 10)
            est = np.where(rms >= max(0.001, k * floor), est, 0.0)
            tot = bm.add_counts(tot, bm.pitch_counts(ref, est))
        whatif[label] = bm.pitch_summary(tot)
    out["gate_whatif"] = whatif
    return out


def analyse_hillenbrand(items, res):
    items = [it for it in items if it["name"] in res]
    groups = ["men", "women", "boys", "girls"]
    out = {"files": len(items), "per_group": {g: sum(it["group"] == g for it in items) for g in groups}}

    def collect(get_est, formant_idx, time_key="ss"):
        per = {g: ([], []) for g in groups + ["all"]}
        for it in items:
            ref_vals = it["formants"] if time_key == "ss" else it[time_key]
            if formant_idx >= len(ref_vals):
                continue
            ref, est = ref_vals[formant_idx], get_est(it, res[it["name"]], formant_idx)
            for g in (it["group"], "all"):
                per[g][0].append(ref)
                per[g][1].append(est)
        return {g: bm.formant_summary(*v) for g, v in per.items() if v[0]}

    c5500 = CEILINGS.index(5500.0)
    variants = {
        "yours_as_logged": lambda it, r, j: r["yours_F"][0, j],
        "yours_centred": lambda it, r, j: r["yours_F"][1, j],
        "candidate": lambda it, r, j: r["cand_F"][0, j],
        "praat_5500": lambda it, r, j: r["praat_F"][c5500, 0, j],
    }
    out["formants"] = {v: {f"F{j + 1}": collect(fn, j) for j in range(4)} for v, fn in variants.items()}

    # ceiling sweep: which fixed ceiling suits each group best (whole-file Praat)
    sweep = {}
    for g in groups:
        sweep[g] = {}
        for ci, c in enumerate(CEILINGS):
            row = {}
            for j in range(3):
                refs = [it["formants"][j] for it in items if it["group"] == g]
                ests = [res[it["name"]]["praat_F"][ci, 0, j] for it in items if it["group"] == g]
                row[f"F{j + 1}"] = bm.formant_summary(refs, ests).get("median_pct", float("nan"))
            sweep[g][str(int(c))] = row
    out["ceiling_sweep"] = sweep
    best = {g: min(sweep[g], key=lambda c: np.nanmean([sweep[g][c][f] for f in ("F1", "F2", "F3")])) for g in groups}
    out["best_ceiling"] = best
    oracle = lambda it, r, j: r["praat_F"][CEILINGS.index(float(best[it["group"]])), 0, j]  # noqa: E731
    out["formants"]["praat_best_ceiling"] = {f"F{j + 1}": collect(oracle, j) for j in range(4)}

    # formant tracking through the vowel (20/50/80%), yours centred vs praat
    traj = {}
    for k, key in enumerate(["f_20", "f_50", "f_80"]):
        traj[key] = {
            "yours": {f"F{j + 1}": collect(lambda it, r, j, k=k: r["yours_F"][2 + k, j], j, key).get("all", {}) for j in range(3)},
            "praat_5500": {f"F{j + 1}": collect(lambda it, r, j, k=k: r["praat_F"][c5500, 1 + k, j], j, key).get("all", {}) for j in range(3)},
        }
    out["trajectory"] = traj

    # pitch at the steady state
    pitch = {}
    for name, key in [("yours_centred", None), ("candidate", "cand_pitch"), ("praat", "praat_pitch"), ("pyin", "pyin_pitch")]:
        per = {g: None for g in groups + ["all"]}
        for it in items:
            r = res[it["name"]]
            est = r["yours_pitch"][1] if key is None else r[key][0]
            c = bm.pitch_counts([it["f0"]], [est])
            for g in (it["group"], "all"):
                per[g] = bm.add_counts(per[g], c)
        pitch[name] = {g: bm.pitch_summary(c) for g, c in per.items() if c is not None}
    out["pitch"] = pitch

    # scatter data for charts (yours centred vs praat best, women + men + kids)
    out["_scatter"] = [{"group": it["group"], "vowel": it["vowel"], "ref": it["formants"][:2],
                        "yours": res[it["name"]]["yours_F"][1, :2].tolist(),
                        "praat": res[it["name"]]["praat_F"][CEILINGS.index(float(best[it["group"]])), 0, :2].tolist()}
                       for it in items]
    return out


def synth_plan(quick):
    plan = []
    f0s = [70, 85, 100, 130, 160, 200, 250, 300, 400, 500, 600, 700, 900, 1200]
    srs = [16000, 44100, 48000] if not quick else [16000, 48000]
    for sr in srs:
        for f0 in f0s:
            plan.append(("pitch_range", {"f0": f0, "sr": sr}))
    base = {"tilt": -12.0, "breath": 0.0, "gain": 0.3, "f0": 180, "sr": 48000, "vowel": "a"}
    factors = {"tilt": [-6.0, -9.0, -12.0, -15.0, -18.0], "breath": [0.0, 0.25, 0.5, 1.0],
               "gain": [0.01, 0.03, 0.1, 0.3, 0.9], "f0": [100, 140, 180, 240, 320],
               "sr": [16000, 22050, 44100, 48000], "vowel": ["a", "i", "u"]}
    for fac, vals in factors.items():
        for v in vals:
            p = dict(base)
            p[fac] = v
            if ("quality", p) not in plan:  # the base config appears in every factor list; run it once
                plan.append(("quality", p))
    plan.append(("timing", {"sr": 48000}))
    return plan, base, factors


def analyse_synth(plan, res, base, factors):
    out = {}
    # pitch range
    rows = []
    for kind, p in plan:
        if kind != "pitch_range" or (kind, json.dumps(p, sort_keys=True)) not in res:
            continue
        r = res[(kind, json.dumps(p, sort_keys=True))]
        row = {"f0": p["f0"], "sr": p["sr"]}
        for m in PITCH_METHODS:
            est = r[m]
            det = 100 * (est > 0).mean()
            ok = est > 0
            med = float(np.median(bm.cents(est[ok], p["f0"]))) if ok.any() else float("nan")
            acc = 100 * (np.abs(bm.cents(est[ok], p["f0"])) < 50).sum() / len(est) if ok.any() else 0.0
            row[m] = {"detected": det, "median_cents": med, "accuracy": acc}
        rows.append(row)
    out["pitch_range"] = rows

    # voice-quality features: sensitivity to the thing they should measure (tilt) vs nuisances
    def stats_for(p):
        key = ("quality", json.dumps(p, sort_keys=True))
        return res.get(key)

    feats = ["vocalweight", "spectral_slope_base", "spectral_slope_derived", "spectral_centroid_base",
             "spectral_centroid_derived", "harmonic_rolloff", "noise_rolloff"]
    q = {}
    for f in feats:
        info = {"by_factor": {}}
        for fac, vals in factors.items():
            meds, jit = [], []
            for v in vals:
                p = dict(base)
                p[fac] = v
                r = stats_for(p)
                if r is None:
                    meds.append(float("nan"))
                    jit.append(float("nan"))
                    continue
                x = np.asarray(r[f], float)
                meds.append(float(np.median(x)))
                jit.append(float(np.median(np.abs(np.diff(x)))))
            info["by_factor"][fac] = {"values": vals, "median": meds, "jitter": jit}
        tilt = info["by_factor"]["tilt"]
        tilt_range = float(np.nanmax(tilt["median"]) - np.nanmin(tilt["median"]))
        info["tilt_rho"] = bm.spearman(tilt["values"], tilt["median"])
        info["tilt_range"] = tilt_range
        info["nuisance_ratio"] = {fac: (float(np.nanmax(info["by_factor"][fac]["median"]) - np.nanmin(info["by_factor"][fac]["median"])) / tilt_range
                                        if tilt_range > 0 else float("inf"))
                                  for fac in ("gain", "f0", "sr", "vowel")}
        info["breath_rho"] = bm.spearman(info["by_factor"]["breath"]["values"], info["by_factor"]["breath"]["median"])
        base_jit = info["by_factor"]["tilt"]["jitter"][2]
        info["jitter_ratio"] = float(base_jit / tilt_range) if tilt_range > 0 else float("inf")
        q[f] = info
    out["quality"] = q

    # timing
    key = ("timing", json.dumps({"sr": 48000}, sort_keys=True))
    if key in res:
        r = res[key]
        ends = r["ends"]
        def first_last(est, good):
            on = ends[np.argmax(good)] if good.any() else float("nan")
            last = ends[len(good) - 1 - np.argmax(good[::-1])] if good.any() else float("nan")
            return on, last
        t = {}
        for name, est in (("yours", r["yours_pitch"]), ("candidate", r["cand_pitch"]), ("praat", r["praat_pitch"])):
            good = est > 0
            on, last = first_last(est, good)
            stable = good & (np.abs(est / 180.0 - 1) < 0.05)
            on_stable, _ = first_last(est, stable)
            t[name] = {"onset_ms": 1000 * (on - 1.0), "stable_ms": 1000 * (on_stable - 1.0), "offset_ms": 1000 * (last - 2.0),
                       "wrong_frames": int((good & (np.abs(est / 180.0 - 1) > 0.2)).sum())}
        f1_true = ds.SYNTH_VOWELS["female"]["a"][0]
        f1_good = np.abs(r["yours_F1"] / f1_true - 1) < 0.10
        on_f1, _ = first_last(r["yours_F1"], f1_good)
        t["yours"]["formant_ok_ms"] = 1000 * (on_f1 - 1.0)
        t["_trace"] = {"ends": ends.tolist(), "yours": r["yours_pitch"].tolist(), "candidate": r["cand_pitch"].tolist(),
                       "praat": r["praat_pitch"].tolist()}
        out["timing"] = t
    return out


# ================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small subset (2 sentences/speaker, 300 vowels)")
    ap.add_argument("--ptdb-per-speaker", type=int, default=8, help="PTDB-TUG sentences per speaker (20 speakers)")
    ap.add_argument("--hillenbrand-limit", type=int, default=None, help="max Hillenbrand vowels (default all 1668)")
    ap.add_argument("--noise-per-speaker", type=int, default=3, help="sentences per speaker for the noise test")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip", nargs="*", default=[], choices=["ptdb", "hillenbrand", "synthetic", "noise"])
    ap.add_argument("--out", default=REPORT_ROOT, help="folder for the report folders")
    args = ap.parse_args()
    if args.quick:
        args.ptdb_per_speaker = min(args.ptdb_per_speaker, 2)
        args.hillenbrand_limit = args.hillenbrand_limit or 300
        args.noise_per_speaker = min(args.noise_per_speaker, 1)

    t_start = time.time()
    chash = code_hash()
    print(f"Analysis code version {analysis_version()}, {args.workers} worker processes")

    # ---------------- datasets ----------------
    if "ptdb" not in args.skip:
        have = ds.load_ptdb()
        per_spk = {}
        for it in have:
            per_spk[it["speaker"]] = per_spk.get(it["speaker"], 0) + 1
        if len(per_spk) < 20 or min(per_spk.values()) < args.ptdb_per_speaker:
            print("Downloading PTDB-TUG subset...")
            ds.download_ptdb(per_speaker=args.ptdb_per_speaker)
    if "hillenbrand" not in args.skip and not ds.load_hillenbrand():
        print("Downloading Hillenbrand vowels...")
        ds.download_hillenbrand()

    results = {"meta": {"run": datetime.now().isoformat(timespec="seconds"), "code_version": analysis_version(),
                        "settings": vars(args)}}

    # ---------------- PTDB-TUG pitch ----------------
    if "ptdb" not in args.skip:
        items = ds.load_ptdb(max_per_speaker=args.ptdb_per_speaker)
        print(f"\nPTDB-TUG pitch: {len(items)} sentences")
        tasks = [(it["name"], "task_ptdb", (it["path"], len(it["ref_f0"]), None, chash)) for it in items]
        if "noise" not in args.skip:
            noise_items = [it for it in items if it["name"] in
                           {i["name"] for spk in {x["speaker"] for x in items}
                            for i in [x for x in items if x["speaker"] == spk][:args.noise_per_speaker]}]
            for it in noise_items:
                for snr in SNRS:
                    tasks.append(((it["name"], snr), "task_ptdb", (it["path"], len(it["ref_f0"]), snr, chash)))
        res = run_all(tasks, args.workers, "PTDB-TUG")
        clean = {k: v for k, v in res.items() if isinstance(k, str)}
        noisy = {k: v for k, v in res.items() if isinstance(k, tuple)}
        results["ptdb"] = analyse_ptdb(items, clean, noisy)

    # ---------------- Hillenbrand formants ----------------
    if "hillenbrand" not in args.skip:
        items = ds.load_hillenbrand(limit=args.hillenbrand_limit)
        print(f"\nHillenbrand vowels: {len(items)} recordings")
        res = run_all([(it["name"], "task_hillenbrand", (it, chash)) for it in items], args.workers, "Hillenbrand")
        results["hillenbrand"] = analyse_hillenbrand(items, res)

    # ---------------- synthetic ----------------
    if "synthetic" not in args.skip:
        plan, base, factors = synth_plan(args.quick)
        print(f"\nSynthetic tests: {len(plan)} signals")
        res = run_all([((k, json.dumps(p, sort_keys=True)), "task_synth", (k, p, chash)) for k, p in plan],
                      args.workers, "synthetic")
        results["synthetic"] = analyse_synth(plan, res, base, factors)

    results["meta"]["seconds"] = round(time.time() - t_start, 1)

    import bench_report
    folder = bench_report.write_report(results, args.out)
    print(f"\nReport written to: {os.path.join(folder, 'report.html')}")


if __name__ == "__main__":
    main()
