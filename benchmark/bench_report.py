"""Turns the benchmark results into benchmark_reports/<date_time>/report.html (+ metrics.json).

Verdicts and suggestions are rule-based: each one only appears when the measured numbers (or a check of the
current source code) support it, so re-running after a fix makes the matching suggestion disappear."""
import base64
import glob
import html
import inspect
import io
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# categorical slots (validated reference palette, light mode), fixed per method across every chart
COLORS = {"yours": "#2a78d6", "yours_as_logged": "#86b6ef", "praat": "#eb6834", "pyin": "#1baf7a",
          "candidate": "#6250d6"}
NAMES = {"yours": "Your program (time-aligned)", "yours_as_logged": "Your program (as logged)",
         "yours_centred": "Your program (time-aligned)", "praat": "Praat", "pyin": "pYIN (librosa)",
         "candidate": "Candidate fix", "praat_5500": "Praat, 5500 Hz ceiling",
         "praat_best_ceiling": "Praat, best ceiling per group"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ================================================================================================
# helpers
# ================================================================================================
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def g(d, *path, default=float("nan")):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return default
        d = d[p]
    return d


def f1(x, unit=""):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}{unit}"


def esc(s):
    return html.escape(str(s))


def fig_to_img(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def new_ax(w=7.2, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)
    return leg


def table(headers, rows, cls=""):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def details(summary, inner):
    return f"<details><summary>{esc(summary)}</summary>{inner}</details>"


# ================================================================================================
# source-code checks (things visible in the code itself)
# ================================================================================================
def code_checks():
    out = {}
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    try:
        import main10morevocalinformation as m10
        src = inspect.getsource(m10.get_audio_data)
        body = "\n".join(l.split("#")[0] for l in src.splitlines()[1:])
        out["noise_profile_unused"] = "noise_power_profile" not in body
        ind = inspect.getsource(m10.get_spectral_indicators)
        out["slope_per_decade"] = "log10" in ind
        rol = inspect.getsource(m10.get_dual_rolloffs)
        out["rolloff_98"] = bool(re.search(r"total \* 0\.98", rol))
        wsrc = inspect.getsource(m10.get_vocal_weight)
        m = re.search(r"buffer_size\s*=\s*(\d+)", wsrc)
        out["weight_buffer"] = int(m.group(1)) if m else None
        psrc = inspect.getsource(m10.get_pitch)
        fmin = re.search(r"fmin\s*=\s*([\d.]+)", psrc)
        fmax = re.search(r"fmax\s*=\s*([\d.]+)", psrc)
        out["fmin"] = float(fmin.group(1)) if fmin else None
        out["fmax"] = float(fmax.group(1)) if fmax else None
    except Exception as e:  # never let a code check break the report
        out["error"] = repr(e)
    try:
        m11 = open(os.path.join(PROJECT_DIR, "main11vocalanalysiswindowalgorithmiconly.py"), encoding="utf-8").read()
        out["label_db_per_khz"] = "dB/kHz" in m11
    except OSError:
        out["label_db_per_khz"] = False
    return out


# ================================================================================================
# verdicts + suggestions
# ================================================================================================
WORKS, PARTLY, NO = "Works", "Partly works", "Doesn't work yet"


def build_verdicts(r):
    v = []
    p, h, s = r.get("ptdb", {}), r.get("hillenbrand", {}), r.get("synthetic", {})
    if p:
        y, pr = g(p, "table", "yours", "all"), g(p, "table", "praat", "all")
        rpa, gpe = g(y, "RPA"), g(y, "GPE")
        verdict = WORKS if rpa >= g(pr, "RPA") - 3 and gpe <= 6 else (PARTLY if rpa >= 70 else NO)
        v.append(("Pitch on clean speech", verdict,
                  f"{f1(rpa, '%')} of voiced frames within half a semitone (Praat {f1(g(pr, 'RPA'), '%')}). "
                  f"Gross errors {f1(gpe, '%')} (Praat {f1(g(pr, 'GPE'), '%')})."))
        vde, fa = g(y, "VDE"), g(y, "false_alarm")
        verdict = WORKS if vde <= 8 else (PARTLY if vde <= 12 else NO)
        v.append(("Voiced vs. silent/unvoiced decision", verdict,
                  f"Reports a pitch on {f1(fa, '%')} of frames that have no voice (Praat {f1(g(pr, 'false_alarm'), '%')}). "
                  f"Misses {f1(100 - g(y, 'voicing_recall'), '%')} of voiced frames."))
        lat = g(p, "latency_ms", "yours")
        asl = g(p, "table", "yours_as_logged", "all", "RPA")
        verdict = WORKS if abs(lat) <= 10 else (PARTLY if abs(lat) <= 20 else NO)
        v.append(("Timestamps in the logs", verdict,
                  f"Values are logged about {lat:.0f} ms after the sound they describe. "
                  f"Taken at face value, pitch accuracy drops from {f1(rpa, '%')} to {f1(asl, '%')}."))
        clean, n10 = g(p, "noise", "clean", "yours", "RPA"), g(p, "noise", "10", "yours", "RPA")
        ratio = n10 / clean if clean else 0
        verdict = WORKS if ratio >= 0.9 else (PARTLY if ratio >= 0.8 else NO)
        v.append(("Pitch with background noise", verdict,
                  f"At 10 dB signal-to-noise, accuracy falls from {f1(clean, '%')} to {f1(n10, '%')} "
                  f"(Praat: {f1(g(p, 'noise', '10', 'praat', 'RPA'), '%')})."))
        sp = g(p, "speed", "p95_ms")
        v.append(("Speed", WORKS if sp < 30 else (PARTLY if sp < 60 else NO),
                  f"95% of frames take under {sp:.1f} ms, "
                  + ("so a 30 ms dashboard timestep keeps up." if sp < 30 else "so a 30 ms dashboard timestep falls behind.")))
    if s.get("pitch_range"):
        rows = [row for row in s["pitch_range"] if row["sr"] == 48000]
        ok = [row["f0"] for row in rows if row["yours"]["accuracy"] >= 90]
        bad = [row["f0"] for row in rows if row["yours"]["accuracy"] < 90]
        verdict = WORKS if not bad else (PARTLY if any(100 <= f <= 500 for f in ok) else NO)
        v.append(("Pitch range", verdict,
                  f"Correct for synthetic voices from {min(ok) if ok else '-'} to {max(ok) if ok else '-'} Hz. "
                  f"Wrong at {', '.join(str(b) for b in bad) or 'none'} Hz."))
    if h:
        y = g(h, "formants", "yours_centred")
        asl = g(h, "formants", "yours_as_logged")
        best = g(h, "formants", "praat_best_ceiling")
        w = [g(y, f, "all", "within_10") for f in ("F1", "F2", "F3")]
        wb = [g(best, f, "all", "within_10") for f in ("F1", "F2", "F3")]
        verdict = WORKS if min(w) >= min(wb) - 3 and min(w) >= 80 else (PARTLY if min(w) >= 65 else NO)
        v.append(("Formants F1-F3 (real vowels)", verdict,
                  f"Within 10% of the hand-checked values for {f1(w[0], '%')} / {f1(w[1], '%')} / {f1(w[2], '%')} "
                  f"of vowels (best Praat setting: {f1(wb[0], '%')} / {f1(wb[1], '%')} / {f1(wb[2], '%')}). "
                  f"As logged (no time correction): {f1(g(asl, 'F1', 'all', 'within_10'), '%')} / "
                  f"{f1(g(asl, 'F2', 'all', 'within_10'), '%')} / {f1(g(asl, 'F3', 'all', 'within_10'), '%')}."))
    q = s.get("quality", {})
    labels = {"vocalweight": "Vocal weight (thickness %)", "spectral_slope_base": "Spectral slope, base (magenta)",
              "spectral_slope_derived": "Spectral slope, derived (yellow)",
              "spectral_centroid_base": "Spectral centre of gravity", "harmonic_rolloff": "Harmonic rolloff (tone)",
              "noise_rolloff": "Noise rolloff (breath)"}
    for f, label in labels.items():
        if f not in q:
            continue
        info = q[f]
        rho = info["tilt_rho"] if f != "noise_rolloff" else info["breath_rho"]
        nuis = max(info["nuisance_ratio"].values())
        worst = max(info["nuisance_ratio"], key=info["nuisance_ratio"].get)
        what = "voice brightness (spectral tilt)" if f != "noise_rolloff" else "breathiness"
        if abs(rho) >= 0.9 and nuis <= 0.5:
            verdict = WORKS
        elif abs(rho) >= 0.7:
            verdict = PARTLY
        else:
            verdict = NO
        v.append((label, verdict,
                  f"Tracks {what} with rank correlation {rho:+.2f}. "
                  f"Changing {_factor_name(worst)} moves it {nuis:.1f}x as much as the full brightness range."))
    return v


def _factor_name(f):
    return {"gain": "loudness", "f0": "pitch", "sr": "sample rate", "vowel": "the vowel"}.get(f, f)


def build_suggestions(r, cc):
    """Each suggestion: (priority, title, evidence, action). Only added when the numbers call for it."""
    sug = []
    p, h, s = r.get("ptdb", {}), r.get("hillenbrand", {}), r.get("synthetic", {})
    if p:
        lat = g(p, "latency_ms", "yours")
        if abs(lat) > 10:
            sug.append((1, "Log the time each value describes, not the end of its window",
                        f"Your values describe audio about {lat:.0f} ms before the logged timestamp. "
                        f"Comparing logs to a dataset at face value gives {f1(g(p, 'table', 'yours_as_logged', 'all', 'RPA'), '%')} "
                        f"pitch accuracy instead of {f1(g(p, 'table', 'yours', 'all', 'RPA'), '%')}. "
                        + (f"Formants within 10% drop from {f1(g(h, 'formants', 'yours_centred', 'F1', 'all', 'within_10'), '%')} "
                           f"to {f1(g(h, 'formants', 'yours_as_logged', 'F1', 'all', 'within_10'), '%')} (F1)." if h else ""),
                        "Each feature uses a different amount of recent audio: pitch the last 50 ms, formants the last 100 ms "
                        "(read at the middle, so 50 ms back), vocal weight the last 256 samples. In main11's logger, store "
                        "the window centre per feature (for example pitch_time = t - 0.025, formant_time = t - 0.05), or "
                        "shift audio_timestamp back by a single agreed amount. Any dataset comparison should use these times."))
        fa, fa_praat = g(p, "table", "yours", "all", "false_alarm"), g(p, "table", "praat", "all", "false_alarm")
        if fa > fa_praat + 5:
            wi = p.get("gate_whatif", {})
            best = min(wi, key=lambda k: g(wi, k, "VDE")) if wi else None
            gates = p.get("gates", {})
            ev = (f"{f1(fa, '%')} of frames with no voice still get a pitch (Praat {f1(fa_praat, '%')}). "
                  f"Your get_pitch only checks loudness (RMS > 0.001) and zero-crossing rate, and librosa.yin always returns "
                  f"some pitch, so any quiet hum or breath that passes both gates gets a pitch.")
            near = 100 * gates.get("false_alarms_near_voicing", 0) / max(gates.get("false_alarms", 1), 1)
            ev += (f" {near:.0f}% of these frames are within 50 ms of real voicing, where the analysis window still holds "
                   f"the start or end of a voiced sound. The typical one is only "
                   f"{g(gates, 'false_alarm_median_db_below_peak'):.0f} dB below the loudest frame, so it is not quiet noise.")
            if best:
                cur_vde, best_vde = g(wi, "current: fixed 0.001", "VDE"), g(wi, best, "VDE")
                if cur_vde - best_vde >= 2:
                    ev += (f" A loudness gate at {best} cuts voicing errors from {f1(cur_vde, '%')} to {f1(best_vde, '%')}.")
                else:
                    ev += (f" A loudness gate tied to each recording's noise floor barely helps (voicing error "
                           f"{f1(cur_vde, '%')} at best becomes {f1(best_vde, '%')}), so the fix has to look at periodicity.")
            sug.append((1, "Add a real voicing check instead of loudness + zero-crossing gates", ev,
                        "Add a periodicity test: librosa.pyin's voiced probability, Praat's voicing threshold, or YIN's "
                        "own aperiodicity value (reject frames above about 0.2-0.3). Judge voicing on the newest 20-30 ms "
                        "rather than the whole 50 ms window so it switches off sooner after a vowel ends, and require 2-3 "
                        "voiced frames in a row before reporting a pitch to remove one-frame blips."))
        n10, n5 = g(p, "noise", "10", "yours", "RPA"), g(p, "noise", "5", "yours", "RPA")
        c10 = g(p, "noise", "10", "candidate", "RPA")
        ng = g(p, "noise_gates", "10", default={})
        if n10 < 0.9 * g(p, "noise", "clean", "yours", "RPA"):
            zshare = 100 * ng.get("by_zcr_gate", 0) / max(ng.get("missed_frames", 1), 1)
            sug.append((1, "Stop using zero-crossing rate to reject noisy frames",
                        f"Pitch accuracy falls to {f1(n10, '%')} at 10 dB and {f1(n5, '%')} at 5 dB SNR. "
                        f"At 10 dB, {zshare:.0f}% of the voiced frames you miss were thrown out by the zero-crossing gate, "
                        f"because noise adds zero crossings even when a voice is clearly there. The candidate fix "
                        f"(Praat on the same 100 ms window) keeps {f1(c10, '%')} at 10 dB.",
                        "Replace the ZCR > 0.15 rule with a periodicity measure (see the voicing suggestion). For live use, "
                        "a microphone with noise suppression off plus a periodicity-based gate is more reliable than ZCR."))
        rows = p.get("f0_bins", {}).get("yours", [])
        low = [row for row in rows if row[0] == 0]
        if (cc.get("fmin") or 0) >= 75 or (cc.get("fmax") or 9999) <= 600:
            low_txt = ""
            if low and low[0][2]:
                low_txt = (f"In the PTDB recordings, {low[0][2]} voiced frames are below 75 Hz (deep or creaky male voices), "
                           f"and only {f1(low[0][4], '%')} of them get a correct pitch. ")
            hi = [row["f0"] for row in s.get("pitch_range", []) if row["sr"] == 48000 and row["f0"] > 600
                  and row["yours"]["accuracy"] < 90]
            if hi:
                low_txt += f"Synthetic voices at {', '.join(map(str, hi))} Hz come out an octave low (half the true pitch). "
            sug.append((2, "Widen the pitch search range",
                        low_txt + f"get_pitch searches {cc.get('fmin')}-{cc.get('fmax')} Hz.",
                        "Use fmin around 55-60 Hz and fmax around 1000-1100 Hz (your notes mention singing up to 1000-2000 Hz). "
                        "YIN needs at least two periods of the lowest pitch in its frame, so pass frame_length explicitly "
                        "(about 2.5/fmin seconds of samples) instead of librosa's default of 2048 samples, which is a different "
                        "duration at every sample rate."))
        gpe, gpe_p = g(p, "table", "yours", "all", "GPE"), g(p, "table", "praat", "all", "GPE")
        if gpe > gpe_p + 1.5:
            sug.append((2, "Smooth out octave jumps over time",
                        f"{f1(gpe, '%')} of your voiced frames are more than 20% off (Praat {f1(gpe_p, '%')}), "
                        f"{f1(g(p, 'table', 'yours', 'all', 'octave'), '%')} are octave jumps. Each frame is estimated on its own, "
                        "so nothing stops a single frame from jumping an octave.",
                        "Keep the last few pitch values and reject a new value that jumps by about an octave unless the jump "
                        "persists for 2-3 frames (or run a median filter over the last 3-5 frames). pYIN and Praat do this "
                        "with a Viterbi path. This per-frame error data is also exactly what the correction network idea "
                        "(learning true values from recent frames) would train on."))
    if h:
        y, c = h["formants"].get("yours_centred", {}), h["formants"].get("candidate", {})
        gains = []
        for grp in ("men", "women", "boys", "girls"):
            for f in ("F1", "F2", "F3"):
                d = g(c, f, grp, "within_10") - g(y, f, grp, "within_10")
                if np.isfinite(d):
                    gains.append((d, grp, f))
        best_gain = max(gains) if gains else (0, "", "")
        worst = min(gains) if gains else (0, "", "")
        if best_gain[0] >= 2:
            overall = np.nanmean([g(c, f, "all", "within_10") - g(y, f, "all", "within_10") for f in ("F1", "F2", "F3")])
            sug.append((3, "Choose the formant ceiling from the voice instead of always 5500 Hz",
                        f"The Praat ceiling sweep picks {', '.join(f'{k} {v} Hz' for k, v in h.get('best_ceiling', {}).items())}. "
                        f"A ceiling chosen from the detected pitch (the candidate fix) raises {best_gain[2]} within-10% for "
                        f"{best_gain[1]} by {best_gain[0]:.1f} points, but lowers {worst[2]} for {worst[1]} by {-worst[0]:.1f}. "
                        f"Averaged over F1-F3 for everyone the change is {overall:+.1f} points. "
                        "Pitch alone is a rough guide to vocal tract length, which is what the ceiling should follow.",
                        "Rather than a fixed pitch rule, try 2-3 ceilings in get_formants_praat and keep the one whose "
                        "formant tracks are smoothest over the last few frames (the Escudero et al. 2009 method). Let a "
                        "user setting override it, since a trained voice can have a high pitch with a long vocal tract."))
    q = s.get("quality", {})
    if q.get("spectral_slope_base") and abs(q["spectral_slope_base"]["tilt_rho"]) < 0.7:
        sb, sd = q["spectral_slope_base"], q.get("spectral_slope_derived", {})
        sug.append((2, "Rebuild the base spectral slope",
                    f"It barely reacts to voice brightness (rank correlation {sb['tilt_rho']:+.2f}, total change "
                    f"{sb['tilt_range']:.1f} across a 12 dB/octave range), while the derived slope tracks it perfectly "
                    f"({g(sd, 'tilt_rho'):+.2f}). The fit runs over every FFT bin up to 4500 Hz with a -60 dB floor, so "
                    "the many high, quiet bins and the floor dominate the line.",
                    "Fit dB against log-frequency on band energies (for example octave bands from 100 to 4000 Hz) or use a "
                    "standard measure such as the alpha ratio (energy 50-1000 Hz vs 1-5 kHz) or the Hammarberg index. "
                    "Or drop it and keep the derived slope."))
    if cc.get("slope_per_decade") and cc.get("label_db_per_khz"):
        sug.append((3, "Fix the spectral slope unit on the dashboard",
                    "The slope is fitted against log10(frequency), so it is in dB per decade (per 10x in frequency). "
                    "The dashboard readout says dB/kHz.",
                    "Change the label to dB/decade, or divide by 3.32 and call it dB/octave, the unit most voice "
                    "research uses."))
    if q.get("vocalweight"):
        vw = q["vocalweight"]
        worst = max(vw["nuisance_ratio"], key=vw["nuisance_ratio"].get)
        if vw["nuisance_ratio"][worst] > 0.5 or vw["jitter_ratio"] > 0.2:
            buf = cc.get("weight_buffer") or 256
            sug.append((2, "Make vocal weight independent of pitch, loudness and sample rate",
                        f"Changing {_factor_name(worst)} moves vocal weight {vw['nuisance_ratio'][worst]:.1f}x as much as the full "
                        f"brightness range (loudness {vw['nuisance_ratio']['gain']:.2f}x, sample rate {vw['nuisance_ratio']['sr']:.2f}x). "
                        f"It also jumps {100 * vw['jitter_ratio']:.0f}% of its useful range from one frame to the next on a perfectly steady vowel. "
                        f"It analyses only {buf} samples ({1000 * buf / 48000:.1f} ms at 48 kHz, {1000 * buf / 16000:.0f} ms at 16 kHz), "
                        "which is less than one pitch period for most voices, and log(energy + 1) depends on the input level.",
                        "Use a fixed duration (25-40 ms) instead of a fixed sample count, resample to one rate first, "
                        "normalise the frame's level before taking logs, and average over a few frames. Then re-run this "
                        "benchmark: the pitch and loudness ratios should fall well below 0.5."))
    if q.get("noise_rolloff") and abs(q["noise_rolloff"]["breath_rho"]) < 0.7:
        sug.append((3, "Check the breath (noise) rolloff",
                    f"Adding breath noise barely changes it (rank correlation {q['noise_rolloff']['breath_rho']:+.2f}), "
                    "so it does not measure breathiness yet."
                    + (" The helper named for an 85% rolloff actually uses 98%." if cc.get("rolloff_98") else ""),
                    "Use harmonics-to-noise ratio (Praat's to_harmonicity, or the cepstral peak prominence CPP) as the "
                    "breathiness measure. Both are standard and have published norms."))
    if cc.get("noise_profile_unused"):
        sug.append((3, "The 2-second noise calibration currently has no effect",
                    "get_audio_data receives noise_power_profile but never uses it (the denoised harmonic function is "
                    "commented out). The live mode waits 2 s at startup for nothing.",
                    "Either use it, for example to set the voicing loudness gate from the room's noise level (see above), "
                    "or skip the recording."))
    sug.sort(key=lambda x: x[0])
    return sug


# ================================================================================================
# charts
# ================================================================================================
def chart_f0_bins(p):
    fig, ax = new_ax()
    methods = ["yours", "praat", "pyin", "candidate"]
    bins = p["f0_bins"]["yours"]
    labels = [f"{lo}-{hi}" if hi < 2000 else f"{lo}+" for lo, hi, *_ in bins]
    x = np.arange(len(labels))
    wdt = 0.2
    for i, m in enumerate(methods):
        vals = [row[4] if row[4] is not None and np.isfinite(row[4]) else 0 for row in p["f0_bins"][m]]
        ax.bar(x + (i - 1.5) * wdt, vals, wdt - 0.02, color=COLORS[m], label=NAMES[m])
    ax.set_xticks(x, [f"{l}\n(n={row[2]})" for l, row in zip(labels, bins)])
    ax.set_xlabel("True pitch range (Hz)", color=INK2, fontsize=9)
    ax.set_ylabel("Frames correct (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 100)
    legend(ax, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    rows = [[f"{l} Hz", row[2]] + [f1(p["f0_bins"][m][i][4], "%") for m in methods] for i, (l, row) in enumerate(zip(labels, bins))]
    return fig_to_img(fig, "Pitch accuracy by true pitch range"), table(["True pitch", "Frames"] + [NAMES[m] for m in methods], rows)


def chart_lag(p):
    fig, ax = new_ax(h=3.0)
    for m in ("yours", "praat"):
        cur = p["lag_curves"][m]
        xs = sorted(int(k) for k in cur)
        ax.plot(xs, [cur[str(k)] for k in xs], color=COLORS[m], linewidth=2, label=NAMES[m].replace(" (time-aligned)", ""))
    ax.axvline(0, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("Shift applied to the logged timestamps (ms; positive = value describes earlier audio)", color=INK2, fontsize=9)
    ax.set_ylabel("Frames correct (%)", color=INK2, fontsize=9)
    legend(ax)
    return fig_to_img(fig, "Pitch accuracy versus time shift")


def chart_noise(p):
    fig, ax = new_ax(h=3.2)
    levels = ["clean"] + [str(s) for s in (20, 10, 5)]
    xl = ["clean", "20 dB", "10 dB", "5 dB"]
    rows = []
    for m in ("yours", "praat", "pyin", "candidate"):
        vals = [g(p, "noise", lv, m, "RPA") for lv in levels]
        ax.plot(xl, vals, color=COLORS[m], linewidth=2, marker="o", markersize=8, label=NAMES[m].replace(" (time-aligned)", ""))
        rows.append([NAMES[m].replace(" (time-aligned)", "")] + [f1(v, "%") for v in vals])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Frames correct (%)", color=INK2, fontsize=9)
    ax.set_xlabel("Signal-to-noise ratio (white noise added)", color=INK2, fontsize=9)
    legend(ax, loc="lower left")
    return fig_to_img(fig, "Pitch accuracy versus noise"), table(["Method"] + xl, rows)


def chart_formants(h):
    fig, ax = new_ax()
    variants = ["yours_as_logged", "yours_centred", "praat_best_ceiling"]
    cols = {"yours_as_logged": COLORS["yours_as_logged"], "yours_centred": COLORS["yours"], "praat_best_ceiling": COLORS["praat"]}
    groups = ["men", "women", "boys", "girls"]
    cats = [(grp, f) for grp in groups for f in ("F1", "F2", "F3")]
    x = np.arange(len(cats))
    wdt = 0.27
    for i, v in enumerate(variants):
        vals = [g(h, "formants", v, f, grp, "within_10", default=0) for grp, f in cats]
        ax.bar(x + (i - 1) * wdt, vals, wdt - 0.02, color=cols[v], label=NAMES[v])
    ax.set_xticks(x, [f"{f}\n{grp}" if f == "F2" else f for grp, f in cats], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Vowels within 10% (%)", color=INK2, fontsize=9)
    legend(ax, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    rows = []
    for v in variants + ["candidate", "praat_5500"]:
        for f in ("F1", "F2", "F3", "F4"):
            d = g(h, "formants", v, f, default={})
            if not d:
                continue
            rows.append([NAMES[v], f] + [f"{f1(g(d, grp, 'within_10'), '%')} · {f1(g(d, grp, 'median_pct'), '%')}"
                                          for grp in groups + ["all"]])
    tbl = table(["Method", "Formant"] + groups + ["all"], rows)
    return fig_to_img(fig, "Formant accuracy by speaker group"), tbl


def chart_vowel_space(h):
    pts = h.get("_scatter", [])
    if not pts:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title in ((axes[0], "yours", "Your program (time-aligned)"), (axes[1], "praat", "Praat, best ceiling")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ref = np.array([pt["ref"] for pt in pts], float)
        est = np.array([pt[key] for pt in pts], float)
        ok = (ref[:, 0] > 0) & (est[:, 0] > 0)
        ax.scatter(ref[ok, 0], est[ok, 0], s=9, color=COLORS["yours" if key == "yours" else "praat"], alpha=0.5, linewidths=0)
        lim = [150, 1400]
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=":")
        ax.plot(lim, [l * 1.1 for l in lim], color=GRID, linewidth=1)
        ax.plot(lim, [l * 0.9 for l in lim], color=GRID, linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("True F1 (Hz)", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("Measured F1 (Hz)", color=INK2, fontsize=9)
    return fig_to_img(fig, "Measured versus true F1 for every vowel")


def chart_pitch_range(s):
    rows = [row for row in s["pitch_range"] if row["sr"] == 48000]
    if not rows:
        return "", ""
    fig, ax = new_ax(h=3.0)
    f0s = [row["f0"] for row in rows]
    for m in ("yours", "praat", "pyin", "candidate"):
        ax.plot(f0s, [row[m]["accuracy"] for row in rows], color=COLORS[m], linewidth=2, marker="o", markersize=8,
                label=NAMES[m].replace(" (time-aligned)", ""))
    ax.set_xscale("log")
    ax.set_xticks(f0s, [str(f) for f in f0s], fontsize=8)
    ax.minorticks_off()
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Synthetic voice pitch (Hz)", color=INK2, fontsize=9)
    ax.set_ylabel("Frames correct (%)", color=INK2, fontsize=9)
    legend(ax, loc="lower left")
    trows = []
    for row in s["pitch_range"]:
        trows.append([row["sr"], row["f0"]] + [f"{f1(row[m]['accuracy'], '%')} ({f1(row[m]['median_cents'])} c)"
                                              for m in ("yours", "praat", "pyin", "candidate")])
    return (fig_to_img(fig, "Pitch accuracy across the pitch range"),
            table(["Sample rate", "True pitch (Hz)", "Your program", "Praat", "pYIN", "Candidate fix"], trows))


def chart_quality(q):
    feats = [f for f in ("vocalweight", "spectral_slope_base", "spectral_slope_derived", "spectral_centroid_base",
                         "harmonic_rolloff", "noise_rolloff") if f in q]
    short = {"vocalweight": "Vocal weight", "spectral_slope_base": "Slope (base)",
             "spectral_slope_derived": "Slope (derived)", "spectral_centroid_base": "Centre of gravity",
             "harmonic_rolloff": "Harmonic rolloff", "noise_rolloff": "Noise rolloff"}
    two_line = {f: label.replace(" ", chr(10), 1) for f, label in short.items()}  # wrap tick labels onto 2 lines
    facs = ["gain", "f0", "sr", "vowel"]
    fcol = {"gain": "#2a78d6", "f0": "#eb6834", "sr": "#1baf7a", "vowel": "#eda100"}
    fig, ax = new_ax(h=3.3)
    x = np.arange(len(feats))
    wdt = 0.2
    for i, fac in enumerate(facs):
        vals = [min(q[f]["nuisance_ratio"][fac], 3.0) for f in feats]
        ax.bar(x + (i - 1.5) * wdt, vals, wdt - 0.02, color=fcol[fac], label=f"changing {_factor_name(fac)}")
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=":")
    ax.set_xticks(x, [two_line[f] for f in feats], fontsize=8)
    ax.set_ylabel("Change ÷ brightness range\n(lower is better, capped at 3)", color=INK2, fontsize=9)
    legend(ax, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    rows = []
    for f in feats:
        info = q[f]
        bt = info["by_factor"]["tilt"]
        rows.append([short[f], f"{info['tilt_rho']:+.2f}", " → ".join(f1(v) for v in bt["median"]),
                     f"{info['breath_rho']:+.2f}"] + [f"{info['nuisance_ratio'][fac]:.2f}" for fac in facs]
                    + [f"{info['jitter_ratio']:.2f}"])
    tbl = table(["Feature", "Tracks tilt (ρ)", "Value at tilt −6 → −18 dB/oct", "Tracks breath (ρ)",
                 "Loudness", "Pitch", "Sample rate", "Vowel", "Frame jitter"], rows)
    return fig_to_img(fig, "How much nuisance factors move each voice-quality feature"), tbl


def chart_timing(s):
    t = s.get("timing", {}).get("_trace")
    if not t:
        return ""
    fig, ax = new_ax(h=2.8)
    ends = np.array(t["ends"])
    for m in ("yours", "candidate", "praat"):
        v = np.array(t[m], float)
        v[v <= 0] = np.nan
        ax.plot(ends, v, color=COLORS[m], linewidth=2, label=NAMES[m].replace(" (time-aligned)", ""))
    ax.axvspan(1.0, 2.0, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xlabel("Logged time (s). Shaded = the vowel is actually sounding", color=INK2, fontsize=9)
    ax.set_ylabel("Pitch (Hz)", color=INK2, fontsize=9)
    ax.set_ylim(0, 700)
    legend(ax, loc="upper center", ncols=3)
    return fig_to_img(fig, "Pitch around a vowel onset and offset")


# ================================================================================================
# comparison with the previous run
# ================================================================================================
KEY_METRICS = [
    ("Pitch accuracy, time-aligned (%)", ("ptdb", "table", "yours", "all", "RPA"), +1),
    ("Pitch accuracy, as logged (%)", ("ptdb", "table", "yours_as_logged", "all", "RPA"), +1),
    ("Voicing error (%)", ("ptdb", "table", "yours", "all", "VDE"), -1),
    ("Pitch on silent frames (%)", ("ptdb", "table", "yours", "all", "false_alarm"), -1),
    ("Gross pitch errors (%)", ("ptdb", "table", "yours", "all", "GPE"), -1),
    ("Pitch accuracy at 10 dB SNR (%)", ("ptdb", "noise", "10", "yours", "RPA"), +1),
    ("Log timestamp lag (ms)", ("ptdb", "latency_ms", "yours"), -1),
    ("F1 within 10% (%)", ("hillenbrand", "formants", "yours_centred", "F1", "all", "within_10"), +1),
    ("F2 within 10% (%)", ("hillenbrand", "formants", "yours_centred", "F2", "all", "within_10"), +1),
    ("F3 within 10% (%)", ("hillenbrand", "formants", "yours_centred", "F3", "all", "within_10"), +1),
    ("Vocal weight: pitch influence (×)", ("synthetic", "quality", "vocalweight", "nuisance_ratio", "f0"), -1),
    ("Base slope tracks tilt (ρ)", ("synthetic", "quality", "spectral_slope_base", "tilt_rho"), +1),
    ("Slowest 5% of frames (ms)", ("ptdb", "speed", "p95_ms"), -1),
]


def previous_metrics(out_root, current_folder):
    folders = sorted(f for f in glob.glob(os.path.join(out_root, "*", "metrics.json")) if os.path.dirname(f) != current_folder)
    if not folders:
        return None, None
    try:
        return json.load(open(folders[-1])), os.path.basename(os.path.dirname(folders[-1]))
    except (OSError, ValueError):
        return None, None


def comparison_section(cur, prev, prev_name):
    if not prev:
        return "<p class='muted'>No earlier run found. The next run will show changes against this one.</p>"
    rows = []
    for label, path, good in KEY_METRICS:
        a, b = g(prev, *path), g(cur, *path)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = b - a
        tag = "same" if abs(d) < 0.05 else ("better" if d * good > 0 else "worse")
        rows.append([esc(label), f1(a), f1(b), f'<span class="chg {tag}">{d:+.1f} ({tag})</span>'])
    return (f"<p class='muted'>Compared with the run in <code>{esc(prev_name)}</code>. "
            "Code version and settings are in each run's metrics.json.</p>" + table(["Metric", "Previous", "Now", "Change"], rows))


# ================================================================================================
# page
# ================================================================================================
CSS = """
:root{--bg:#f6f5f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--ok:#0f7b3d;--part:#9a6200;--no:#b42318;
--okbg:#e5f4ea;--partbg:#fcf0d9;--nobg:#fde8e7;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--line:#34332f;--ok:#5cc58a;--part:#e0a53a;--no:#f07c74;--okbg:#16301f;--partbg:#3a2d12;--nobg:#3d1a18;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#34332f;--ok:#5cc58a;--part:#e0a53a;
--no:#f07c74;--okbg:#16301f;--partbg:#3a2d12;--nobg:#3d1a18;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 10px;
padding-top:8px;border-top:1px solid var(--line)}h3{font-size:16px;margin:24px 0 8px}p{margin:8px 0}.muted{color:var(--ink2)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--ink2);font-weight:600}
td:not(:first-child){font-variant-numeric:tabular-nums}
.v{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.v.ok{background:var(--okbg);color:var(--ok)}.v.part{background:var(--partbg);color:var(--part)}.v.no{background:var(--nobg);color:var(--no)}
img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fcfcfb;display:block;margin:10px 0}
.sug{border-left:4px solid var(--accent)}.sug h3{margin-top:0}.prio{font-size:12px;color:var(--ink2);font-weight:600;letter-spacing:.02em}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}code{font-size:13px}
.chg.better{color:var(--ok)}.chg.worse{color:var(--no)}.chg.same{color:var(--ink2)}
nav a{color:var(--accent);margin-right:14px;font-size:14px;text-decoration:none}
"""


def write_report(results, out_root):
    folder = os.path.join(out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    cc = code_checks()
    results["code_checks"] = cc
    verdicts = build_verdicts(results)
    sugs = build_suggestions(results, cc)
    clean = _clean(results)
    clean["verdicts"] = [list(v) for v in verdicts]
    clean["suggestions"] = [list(x) for x in sugs]
    prev, prev_name = previous_metrics(out_root, folder)
    json.dump(clean, open(os.path.join(folder, "metrics.json"), "w"), indent=1)

    p, h, s = results.get("ptdb", {}), results.get("hillenbrand", {}), results.get("synthetic", {})
    meta = results["meta"]
    parts = []
    parts.append(f"""<h1>Vocal Analysis Benchmark</h1>
<p class="muted">Run {esc(meta['run'])} · analysis code version <code>{esc(meta['code_version'])}</code> ·
{meta.get('seconds', 0) / 60:.1f} min · settings in metrics.json</p>
<nav><a href="#summary">Summary</a><a href="#suggestions">Suggestions</a><a href="#pitch">Pitch</a><a href="#formants">Formants</a>
<a href="#quality">Voice quality</a><a href="#timing">Timing</a><a href="#changes">Changes</a><a href="#method">Method</a></nav>
<div class="card"><p><b>What this tests.</b> Your analysis code (get_audio_data, run exactly the way main11 logs it in headless
mode) is compared with datasets that have known answers, and with Praat and pYIN, two standard tools, on the same audio.
"Time-aligned" scores shift your values back by your program's measured delay, which isolates how accurate the numbers are.
"As logged" scores take the timestamps in your log files at face value.</p></div>""")

    cls = {WORKS: "ok", PARTLY: "part", NO: "no"}
    parts.append('<h2 id="summary">Summary</h2>')
    parts.append(table(["Capability", "Verdict", "Evidence"],
                       [[esc(a), f'<span class="v {cls[b]}">{esc(b)}</span>', esc(c)] for a, b, c in verdicts]))

    parts.append('<h2 id="suggestions">Suggestions, most important first</h2>')
    if not sugs:
        parts.append("<p>No rule fired. Everything measured is within the targets.</p>")
    for prio, title, ev, act in sugs:
        parts.append(f'<div class="card sug"><div class="prio">PRIORITY {prio}</div><h3>{esc(title)}</h3>'
                     f'<p><b>Evidence.</b> {esc(ev)}</p><p><b>What to change.</b> {esc(act)}</p></div>')

    if p:
        parts.append(f'<h2 id="pitch">Pitch and voicing: PTDB-TUG ({p["files"]} sentences, {p["speakers"]} speakers)</h2>')
        parts.append("<p>The reference pitch comes from a laryngograph, a sensor on the throat that measures the vocal folds "
                     "directly, recorded at the same time as the microphone. Scores are per 10 ms frame.</p>")
        heads = ["Method", "Group", "Voiced frames found", "Pitch on silent frames", "Voicing error", "Correct (±50 cents)",
                 "Gross errors (>20%)", "Octave errors", "Typical error (cents)"]
        rows = []
        for v in ("yours", "yours_as_logged", "candidate", "praat", "pyin"):
            for grp in ("all", "male", "female"):
                d = g(p, "table", v, grp, default={})
                if d:
                    rows.append([NAMES[v] if grp == "all" else "", grp] + [f1(d.get(k), "%") for k in
                                ("voicing_recall", "false_alarm", "VDE", "RPA", "GPE", "octave")] + [f1(d.get("fine_cents"))])
        parts.append(table(heads, rows))
        img, tbl = chart_f0_bins(p)
        parts.append("<h3>Accuracy by pitch range</h3>" + img + details("Table", tbl))
        parts.append(f"<h3>Timing</h3><p>Your values line up best with the reference when shifted by "
                     f"{p['latency_ms']['yours']:.0f} ms, so each logged value describes audio from about that long before its "
                     f"timestamp. Praat is the zero point.</p>" + chart_lag(p))
        gt = p.get("gates", {})
        wi = p.get("gate_whatif", {})
        parts.append("<h3>Why frames are missed or wrongly voiced</h3>" + table(
            ["", "Frames"], [["Voiced frames missed", gt.get("missed_frames")],
                             ["… rejected by the loudness gate (RMS < 0.001)", gt.get("by_rms_gate")],
                             ["… rejected by the zero-crossing gate (ZCR > 0.15)", gt.get("by_zcr_gate")],
                             ["Silent or unvoiced frames given a pitch", gt.get("false_alarms")]]))
        if wi:
            parts.append("<p>What-if: the same frames with a loudness gate set relative to each recording's noise floor "
                         "(its quietest 10% of frames), like a calibration would.</p>" + table(
                ["Loudness gate", "Voiced frames found", "Pitch on silent frames", "Voicing error", "Correct"],
                [[esc(k), f1(g(v, "voicing_recall"), "%"), f1(g(v, "false_alarm"), "%"), f1(g(v, "VDE"), "%"), f1(g(v, "RPA"), "%")]
                 for k, v in wi.items()]))
        img, tbl = chart_noise(p)
        ng = p.get("noise_gates", {})
        parts.append(f"<h3>Background noise ({p.get('noise_files', 0)} sentences)</h3>" + img + details("Table", tbl + table(
            ["SNR", "Your voiced frames missed", "by loudness gate", "by zero-crossing gate"],
            [[f"{k} dB", d.get("missed_frames"), d.get("by_rms_gate"), d.get("by_zcr_gate")] for k, d in ng.items()])))
        sp = p["speed"]
        parts.append(f"<h3>Speed</h3><p>Median {sp['median_ms']:.1f} ms per frame, 95th percentile {sp['p95_ms']:.1f} ms, "
                     f"slowest {sp['max_ms']:.0f} ms (usually the very first call, while libraries warm up), over {sp['frames']} frames.</p>")
        if h.get("pitch"):
            parts.append("<h3>Pitch at the vowel's steady state (Hillenbrand)</h3>" + table(
                ["Method", "Group", "Correct (±50 cents)", "Gross errors", "Missed"],
                [[NAMES.get(m, m) if grp == "all" else "", grp, f1(g(d, "RPA"), "%"), f1(g(d, "GPE"), "%"),
                  f1(100 - g(d, "voicing_recall"), "%")]
                 for m, dd in h["pitch"].items() for grp, d in dd.items() if grp in ("all", "men", "women", "boys", "girls")]))

    if h:
        parts.append(f'<h2 id="formants">Formants: Hillenbrand vowels ({h["files"]} recordings)</h2>')
        parts.append("<p>Twelve vowels in “h_d” words (heed, hid, head, had…) from men, women and children, with formants "
                     "measured and hand-checked by the authors at each vowel's steady state.</p>")
        img, tbl = chart_formants(h)
        parts.append(img + details("Table: within 10% · median error, per group", tbl))
        parts.append(chart_vowel_space(h))
        sw = h.get("ceiling_sweep", {})
        parts.append("<h3>Which formant ceiling suits each group (Praat, median % error F1 / F2 / F3)</h3>" + table(
            ["Group"] + [f"{c} Hz" for c in next(iter(sw.values()), {}).keys()],
            [[grp] + [" / ".join(f1(v[f]) for f in ("F1", "F2", "F3")) for v in d.values()] for grp, d in sw.items()]))
        tr = h.get("trajectory", {})
        if tr:
            parts.append("<h3>Through the vowel (20%, 50%, 80% of its duration, all groups, within 10%)</h3>" + table(
                ["Point", "Method", "F1", "F2", "F3"],
                [[k.replace("f_", "") + "%", NAMES.get(m, m)] + [f1(g(d, f, "within_10"), "%") for f in ("F1", "F2", "F3")]
                 for k, dd in tr.items() for m, d in dd.items()]))

    if s:
        parts.append('<h2 id="quality">Synthetic tests</h2>')
        parts.append("<p>Vowels generated with exactly known settings. They are cleaner than real voices, so treat these as "
                     "a check that each measure behaves sensibly, not as a precision score.</p>")
        img, tbl = chart_pitch_range(s)
        parts.append("<h3>Pitch range (48 kHz)</h3>" + img + details("Table, all sample rates (median error in cents)", tbl))
        if s.get("quality"):
            img, tbl = chart_quality(s["quality"])
            parts.append("<h3>Voice-quality features</h3><p>Each feature should respond to voice brightness (spectral "
                         "tilt, varied from −6 to −18 dB per octave) and stay put when only loudness, pitch, sample rate or "
                         "vowel change. Bars show how far each of those moves the feature, as a fraction of how far the full "
                         "brightness range moves it. Below the dotted 0.5 line is good.</p>" + img + details("Table", tbl))
        tm = s.get("timing")
        if tm:
            parts.append('<h3 id="timing">Onset and offset</h3>' + chart_timing(s) + table(
                ["Method", "First pitch after onset", "Within 5% after onset", "Last pitch after the voice stops",
                 "Frames more than 20% off"],
                [[NAMES[m].replace(" (time-aligned)", ""), f1(d.get("onset_ms"), " ms"), f1(d.get("stable_ms"), " ms"),
                  f1(d.get("offset_ms"), " ms"), d.get("wrong_frames", "n/a")] for m, d in tm.items() if not m.startswith("_")]))
            parts.append(f"<p>F1 settles within 10% about {f1(tm['yours'].get('formant_ok_ms'), ' ms')} after the onset.</p>")

    parts.append('<h2 id="changes">Changes since the previous run</h2>' + comparison_section(clean, prev, prev_name))

    parts.append("""<h2 id="method">Datasets, method and limits</h2>
<ul>
<li><b>PTDB-TUG</b>, Pirker et al., Interspeech 2011. Read English sentences from 10 female and 10 male speakers, with
laryngograph-based reference pitch every 10 ms. Open Database License.
<a href="https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html">spsc.tugraz.at</a></li>
<li><b>Hillenbrand et al. 1995</b>, JASA 97(5). 1,668 vowel recordings at 16 kHz from 45 men, 48 women and 46 children.
MIT license, hosted with the author's permission. <a href="https://github.com/santiagobarreda/hillenbrand_et_al_1995">github.com/santiagobarreda/hillenbrand_et_al_1995</a></li>
<li><b>Synthetic vowels</b> built in benchmark/bench_datasets.py: a harmonic source with exact spectral tilt through
four formant resonators.</li>
</ul>
<p><b>Metrics.</b> Correct = within 50 cents (half a semitone) of the reference. Gross error = more than 20% off. Voicing error =
frames where voiced/unvoiced is called wrong. Formants: share of vowels within 10% of the measured value, and the median error.
Each tool's time alignment is found by testing shifts and keeping the best one.</p>
<p><b>Limits.</b> Hillenbrand's formants were measured with LPC, the same family of method Praat and your code use, so formant
scores may look better than they would against X-ray or MRI data. The laryngograph reference marks creaky voice as voiced,
which no audio-only tracker handles well. Noise here is white noise only. Live microphones add automatic gain control and room
echo, which is not tested. Only a subset of PTDB-TUG is used; raise <code>--ptdb-per-speaker</code> for more.</p>""")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vocal Analysis Benchmark</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"""
    with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return folder
