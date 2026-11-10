"""How good are the live syllables, the Whisper reference, and the vowel labels? Measured, not guessed.

  python benchmark/eval_syllables.py                 # everything (~10-15 min with a GPU)
  python benchmark/eval_syllables.py --parts arctic  # just one part (arctic, ravdess, hillenbrand, noise)
  python benchmark/eval_syllables.py --quick         # fewer sentences, for a fast check

Parts
  arctic       CMU ARCTIC sentences have word/syllable/phone boundaries from forced alignment. Each
               speaker's sentences are joined into one long stream, like a live session.
               Detector settings are TUNED on speakers clb + rms and REPORTED on slt + bdl (never
               seen while tuning). Measures: nuclei found inside the right vowel (precision/recall/F1),
               syllable-start error, Whisper's own word-start error, whether the live Whisper-based
               score tracks the true score, and how often a vowel-chart point has the right vowel.
  ravdess      24 actors x 8 emotions saying two 7-syllable sentences: is the count right?
  hillenbrand  1,668 /hVd/ words with hand-checked formants: F1-F3 at the detected nucleus, for each
               formant-ceiling option (incl. the speaker-fitted one, per group). Rebuilds the
               "hillenbrand_*_measured" targets.
  targets      running-speech targets: 300 ARCTIC sentences each from 2 women and 2 men, measured exactly
               like a live session (true word times as labels). Writes syllabic/data/hillenbrand_targets.json.
  noise        the arctic test streams with white noise at 10 and 5 dB SNR.
Results go to benchmark_reports/syllables_<time>/results.json (+ a printed report).
"""
import argparse
import difflib
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (HERE, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_datasets as ds  # noqa: E402
from syllabic import formants as fmt  # noqa: E402
from syllabic import pronounce, vowelspace  # noqa: E402
from syllabic.frontend import SR, FrameAnalyzer, to_16k  # noqa: E402
from syllabic.pipeline import Pipeline, PipelineConfig  # noqa: E402
from syllabic.syllables import DetectorParams, SyllableDetector  # noqa: E402

warnings.simplefilter("ignore")
FEST2ARPA = {"aa": "AA", "ae": "AE", "ah": "AH", "ao": "AO", "aw": "AW", "ax": "AH", "axr": "ER", "ay": "AY",
             "eh": "EH", "er": "ER", "ey": "EY", "ih": "IH", "iy": "IY", "ow": "OW", "oy": "OY", "uh": "UH",
             "uw": "UW"}
TUNE_SPK, TEST_SPK = ("clb", "rms"), ("slt", "bdl")


def add_noise(y, snr_db, seed):
    """Same as run_benchmark.add_noise: white noise at snr_db against the louder half of the frames."""
    rng = np.random.default_rng(seed)
    frames = y[: len(y) // 160 * 160].reshape(-1, 160)
    p = (frames ** 2).mean(axis=1)
    speech_power = p[p >= np.median(p)].mean()
    return (y + rng.standard_normal(len(y)) * np.sqrt(speech_power / 10 ** (snr_db / 10))).astype(np.float32)


def r1(x):
    return None if x is None else round(float(x), 1)


# ------------------------------------------------------------------------------------------------
# ARCTIC streams with ground truth
# ------------------------------------------------------------------------------------------------
def build_stream(items, gap_s=0.3):
    """Joins one speaker's sentences into a single stream; ground truth shifted to stream time."""
    audio, words, t = [], [], 0.0
    for it in items:
        y, sr = sf.read(it["path"], dtype="float32")
        assert sr == SR
        for w in it["words"]:
            sylls = [dict(s, start=s["start"] + t, end=s["end"] + t,
                          vowel=(s["vowel"][0], s["vowel"][1] + t, s["vowel"][2] + t),
                          phones=[(ph, a + t, b + t) for ph, a, b in s["phones"]])
                     for s in w["syllables"] if s["vowel"] is not None]
            if w["word"].startswith("'") and words and words[-1]["utt"] == it["name"]:
                words[-1]["word"] += w["word"]          # festival splits "Pierre's" into Pierre + 's
                words[-1]["end"] = w["end"] + t
                continue
            if sylls:
                words.append({"word": w["word"], "start": w["start"] + t, "end": w["end"] + t, "syllables": sylls,
                              "utt": it["name"]})
        audio.append(y)
        audio.append(np.zeros(int(gap_s * SR), np.float32))
        t += len(y) / SR + gap_s
    # a word is "after a pause" if the previous word ended >= 0.15 s earlier (or it starts a sentence)
    for i, w in enumerate(words):
        w["after_pause"] = i == 0 or w["utt"] != words[i - 1]["utt"] or w["start"] - words[i - 1]["end"] >= 0.15
        for j, s in enumerate(w["syllables"]):
            s["word_i"], s["first"] = i, j == 0
    return np.concatenate(audio), words


def stream_frames(y16, band=DetectorParams().band):
    fa = FrameAnalyzer(band=tuple(band))
    a, b = fa.push(y16), fa.flush()
    return {k: np.concatenate([a[k], b[k]]) if b is not None else a[k] for k in a}


def run_detector(frames, params):
    det = SyllableDetector(params)
    n = len(frames["t"])
    for i in range(0, n, 3):   # 30 ms chunks, like the live GUI tick
        det.push({k: v[i: i + 3] for k, v in frames.items()})
    det.flush()
    # the re-check the live pipeline does when a phrase ends: at the very start the "loudest speech"
    # reference does not exist yet, so a quiet noise peak can get through the live check
    return [s for s in det.syllables if det.ref is None or s.peak_db >= det.ref - params.silence_rel_db]


def score_detector(sylls, words, tol_vowel=0.02):
    """Detected nuclei vs ground-truth vowels (one-to-one), and syllable-start errors."""
    gt = [s for w in words for s in w["syllables"]]
    gt.sort(key=lambda s: s["vowel"][1])
    starts = np.array([s["vowel"][1] - tol_vowel for s in gt])
    taken = np.zeros(len(gt), bool)
    pairs = []
    for d in sylls:
        k = int(np.searchsorted(starts, d.t_peak, side="right")) - 1
        for kk in (k, k - 1):
            if 0 <= kk < len(gt) and not taken[kk] and gt[kk]["vowel"][1] - tol_vowel <= d.t_peak <= gt[kk]["vowel"][2] + tol_vowel:
                taken[kk] = True
                pairs.append((d, gt[kk]))
                break
    hits, n_det, n_gt = len(pairs), len(sylls), len(gt)
    prec = hits / n_det if n_det else 0.0
    rec = hits / n_gt if n_gt else 0.0
    err = np.array([d.t_onset - g["start"] for d, g in pairs])
    first = np.array([g["first"] for _, g in pairs])
    pause = np.array([words[g["word_i"]]["after_pause"] and g["first"] for _, g in pairs])
    pos = np.array([(d.t_peak - g["vowel"][1]) / max(g["vowel"][2] - g["vowel"][1], 1e-3) for d, g in pairs])

    def stats(e):
        if len(e) == 0:
            return {}
        return {"n": int(len(e)), "median_ms": r1(1000 * np.median(e)), "mae_ms": r1(1000 * np.mean(np.abs(e))),
                "within_50ms": r1(100 * np.mean(np.abs(e) <= 0.05)), "within_100ms": r1(100 * np.mean(np.abs(e) <= 0.1))}
    # per word: detected nuclei inside the true word vs its true syllable count
    tp = np.array([d.t_peak for d in sylls])
    exact = [int(np.sum((tp >= w["start"]) & (tp <= w["end"]))) == len(w["syllables"]) for w in words]
    return {"precision": r1(100 * prec), "recall": r1(100 * rec),
            "f1": r1(200 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "n_true": n_gt, "n_detected": n_det,
            "nucleus_position_in_vowel_median": r1(100 * np.median(pos)) if len(pos) else None,
            "word_count_exact": r1(100 * np.mean(exact)),
            "onset_all_syllables": stats(err), "onset_word_initial": stats(err[first]) if len(err) else {},
            "onset_after_pause": stats(err[pause]) if len(err) else {},
            "onset_connected_word_initial": stats(err[first & ~pause]) if len(err) else {}}


def tune(streams, log):
    """Grid search on the TUNE speakers only. Stage 1: band, dip, voicing, smoothing for the best
    nucleus F1. Stage 2: the syllable-start rule and pause settings for the most starts within 50 ms.
    streams: [(y16, words)]."""
    cache = {}

    def frames(i, band):
        if (i, band) not in cache:
            cache[(i, band)] = stream_frames(streams[i][0], band)
        return cache[(i, band)]

    def mean_score(p, key):
        sc = [score_detector(run_detector(frames(i, tuple(p.band)), p), w) for i, (_, w) in enumerate(streams)]
        return float(np.mean([key(s) for s in sc]))

    best = None
    for band in ((300.0, 2500.0), (400.0, 2500.0), (500.0, 3000.0)):
        for dip in (1.0, 1.5, 2.0):
            for voi in (0.5, 0.6):
                for sm in (3, 5):
                    p = DetectorParams(band=band, min_dip_db=dip, voicing=voi, smooth=sm)
                    f1 = mean_score(p, lambda s: s["f1"])
                    if best is None or f1 > best[0]:
                        best = (f1, p)
    log(f"  stage 1 best nucleus F1 on tune speakers: {best[0]:.1f}  (band {best[1].band}, dip "
        f"{best[1].min_dip_db} dB, voicing {best[1].voicing}, smooth {best[1].smooth})")
    best2 = None
    for rule, frac in (("argmin", 0.5), ("end", 0.5), ("fall", 0.5)):
        for pmargin in (6.0, 10.0):
            for pmin in (0.05, 0.1):
                for rel in (0.0, 20.0, 30.0):
                    for shift in (-0.02, 0.0):
                        q = replace(best[1], onset_rule=rule, fall_frac=frac, pause_margin_db=pmargin,
                                    pause_min_s=pmin, onset_rel_db=rel, onset_shift_s=shift)
                        within = mean_score(q, lambda s: s["onset_all_syllables"].get("within_50ms", 0))
                        if best2 is None or within > best2[0]:
                            best2 = (within, q)
    q = best2[1]
    log(f"  stage 2 best syllable-start rule: {q.onset_rule} (fall_frac {q.fall_frac}), pause margin "
        f"{q.pause_margin_db} dB, pause {q.pause_min_s} s, breath guard {q.onset_rel_db} dB, shift "
        f"{1000 * q.onset_shift_s:+.0f} ms -> {best2[0]:.1f}% of starts within 50 ms")
    return q


def norm_words(ws):
    return [pronounce.normalize(w).replace("'", "") for w in ws]


def whisper_vs_truth(p, words):
    """Whisper's word starts vs the true word starts (words matched by text)."""
    hyp = p.words
    a, b = norm_words([w["word"] for w in words]), norm_words([h.word for h in hyp])
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    err, err_first, pairs = [], [], []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            g, h = words[blk.a + k], hyp[blk.b + k]
            e = h.start - g["start"]
            err.append(e)
            pairs.append((g, h))
            if h.after_pause:
                err_first.append(e)
    err, err_first = np.array(err), np.array(err_first)

    def stats(e):
        return {} if len(e) == 0 else {"n": int(len(e)), "median_ms": r1(1000 * np.median(e)),
                                       "mae_ms": r1(1000 * np.mean(np.abs(e))),
                                       "within_50ms": r1(100 * np.mean(np.abs(e) <= 0.05)),
                                       "within_100ms": r1(100 * np.mean(np.abs(e) <= 0.1))}
    return {"words_true": len(words), "words_whisper": len(hyp), "words_matched": len(pairs),
            "start_error_all": stats(err), "start_error_after_pause": stats(err_first),
            "start_error_connected": stats(np.array([h.start - g["start"] for g, h in pairs if not h.after_pause]))}


def live_vs_true(p, words, tol=0.10):
    """The live score (judged against Whisper) next to the same score judged against the truth."""
    live = p.score.summary("all")
    tp = np.array([s.t_peak for s in p.syllables if s.status != "quiet"])
    on = np.array([s.t_onset for s in p.syllables if s.status != "quiet"])
    exact, hit = [], []
    for w in words:
        inside = (tp >= w["start"]) & (tp <= w["end"])
        exact.append(int(inside.sum()) == len(w["syllables"]))
        v0 = w["syllables"][0]["vowel"]
        m = np.where((tp >= v0[1] - 0.02) & (tp <= v0[2] + 0.02))[0]
        hit.append(len(m) > 0 and abs(on[m[0]] - w["start"]) <= tol)
    # like for like: on exactly the words the live score judges (scored, not first in their phrase), its
    # verdict against Whisper's start vs the same detected start judged against the true word start
    a, b = norm_words([w["word"] for w in words]), norm_words([h.word for h in p.words])
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    same_live, same_true = [], []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            g, h = words[blk.a + k], p.words[blk.b + k]
            if h.scored and not h.phrase_initial:
                n0 = h.nuclei[0] if h.nuclei else None
                same_live.append(n0 is not None and abs(n0.t_onset - h.start) <= tol)
                same_true.append(n0 is not None and abs(n0.t_onset - g["start"]) <= tol)
    return {"live_count_exact": r1(live.get("count_exact")), "true_count_exact": r1(100 * np.mean(exact)),
            "live_onset_hit": r1(live.get("onset_hit")), "true_onset_hit_all_words": r1(100 * np.mean(hit)),
            "same_words": len(same_live),
            "same_words_live_onset_hit": r1(100 * np.mean(same_live)) if same_live else None,
            "same_words_true_onset_hit": r1(100 * np.mean(same_true)) if same_true else None,
            "live_recall": r1(live.get("recall")), "live_precision": r1(live.get("precision")),
            "live_onset_median_ms": r1(live.get("onset_median_ms"))}


def token_accuracy(p, words):
    """Is each vowel-chart point really in a vowel, and in the vowel its label says?"""
    gt = [s for w in words for s in w["syllables"]]
    res = {"chart": [0, 0, 0], "all": [0, 0, 0]}  # tokens, in a true vowel, same vowel
    for t in p.tokens:
        g = next((s for s in gt if s["vowel"][1] - 0.02 <= t["t"] <= s["vowel"][2] + 0.02), None)
        for key in ("all", "chart") if t["chart"] else ("all",):
            res[key][0] += 1
            if g is not None:
                res[key][1] += 1
                if FEST2ARPA.get(g["vowel"][0]) == t["vowel"]:
                    res[key][2] += 1
    return {k: {"tokens": v[0], "in_a_vowel": r1(100 * v[1] / v[0]) if v[0] else None,
                "same_vowel": r1(100 * v[2] / v[0]) if v[0] else None} for k, v in res.items()}


def part_arctic(args, log, whisper):
    log("\n=== ARCTIC (tune: clb, rms | test: slt, bdl) ===")
    n = 40 if args.quick else 120
    items = ds.load_arctic(max_per_speaker=n, seed=1)
    streams = {}
    for spk in TUNE_SPK + TEST_SPK:
        raw, words = build_stream([it for it in items if it["speaker"] == spk])
        y = to_16k(raw, SR)   # same 60 Hz high-pass as the live path
        streams[spk] = (y, words, stream_frames(y), raw)
        log(f"  {spk}: {len(words)} words, {sum(len(w['syllables']) for w in words)} syllables, "
            f"{len(y) / SR / 60:.1f} min")
    default = DetectorParams()
    out = {"default_params": asdict(default)}
    out["default_on_test"] = {s: score_detector(run_detector(streams[s][2], default), streams[s][1]) for s in TEST_SPK}
    if args.no_tune:
        tuned = default
    else:
        tuned = tune([(streams[s][0], streams[s][1]) for s in TUNE_SPK], log)
    out["tuned_params"] = asdict(tuned)
    if tuple(tuned.band) != tuple(default.band):
        for s in TEST_SPK:
            streams[s] = (streams[s][0], streams[s][1], stream_frames(streams[s][0], tuned.band), streams[s][3])
    out["tuned_on_test"] = {s: score_detector(run_detector(streams[s][2], tuned), streams[s][1]) for s in TEST_SPK}
    for s in TEST_SPK:
        d, t = out["default_on_test"][s], out["tuned_on_test"][s]
        log(f"  {s}: nucleus F1 {d['f1']} -> {t['f1']} (P {t['precision']} R {t['recall']}), "
            f"syllable starts within 50 ms {d['onset_all_syllables'].get('within_50ms')} -> "
            f"{t['onset_all_syllables'].get('within_50ms')}%, word count exact {t['word_count_exact']}%")
    if whisper is not None:
        cfg = PipelineConfig(detector=tuned)
        out["whisper"], out["live_vs_true"], out["tokens"] = {}, {}, {}
        for s in TEST_SPK:
            y, words = streams[s][0], streams[s][1]
            t0 = time.time()
            p = Pipeline(SR, cfg, whisper=whisper, store_seconds=len(y) / SR + 10)
            for i in range(0, len(y), 800):
                p.feed16(y[i: i + 800])
            p.finish()
            out["whisper"][s] = whisper_vs_truth(p, words)
            out["live_vs_true"][s] = live_vs_true(p, words)
            out["tokens"][s] = token_accuracy(p, words)
            w, lv, tk = out["whisper"][s], out["live_vs_true"][s], out["tokens"][s]
            log(f"  {s} with Whisper ({time.time() - t0:.0f} s): Whisper word starts within 50 ms "
                f"{w['start_error_all'].get('within_50ms')}% (median {w['start_error_all'].get('median_ms')} ms; "
                f"after a pause {w['start_error_after_pause'].get('median_ms')} ms)")
            log(f"     live score vs truth: count exact {lv['live_count_exact']} vs {lv['true_count_exact']}, "
                f"word-start hit on the same {lv['same_words']} words {lv['same_words_live_onset_hit']} vs "
                f"{lv['same_words_true_onset_hit']}")
            log(f"     vowel-chart points: {tk['chart']['tokens']}, in a real vowel {tk['chart']['in_a_vowel']}%, "
                f"same vowel as the alignment {tk['chart']['same_vowel']}%")
    return out, tuned, streams


def part_noise(args, log, tuned, streams):
    log("\n=== noise (arctic test speakers) ===")
    out = {}
    for snr in (None, 10, 5):
        out[str(snr or "clean")] = {}
        for s in TEST_SPK:
            y, words, fr, raw = streams[s]
            if snr is not None:
                fr = stream_frames(to_16k(add_noise(raw, snr, seed=7), SR), tuned.band)
            sc = score_detector(run_detector(fr, tuned), words)
            out[str(snr or "clean")][s] = sc
        f1 = np.mean([out[str(snr or "clean")][s]["f1"] for s in TEST_SPK])
        w50 = np.mean([out[str(snr or "clean")][s]["onset_all_syllables"].get("within_50ms", 0) for s in TEST_SPK])
        log(f"  {'clean' if snr is None else f'{snr} dB'}: nucleus F1 {f1:.1f}, syllable starts within 50 ms {w50:.1f}%")
    return out


def part_ravdess(args, log, tuned, whisper):
    log("\n=== RAVDESS (7 syllables per sentence) ===")
    items = ds.load_ravdess()
    if args.quick:
        items = items[::8]
    by = {}
    for it in items:
        y, sr = sf.read(it["path"], dtype="float32")
        n = len(run_detector(stream_frames(to_16k(y, sr), tuned.band), tuned))
        by.setdefault(it["emotion"], []).append(n)
    out = {}
    for emo, ns in by.items():
        ns = np.array(ns)
        out[emo] = {"clips": int(len(ns)), "exact_7": r1(100 * np.mean(ns == 7)), "within_1": r1(100 * np.mean(np.abs(ns - 7) <= 1)),
                    "mean_detected": r1(np.mean(ns))}
    allv = np.concatenate([np.array(v) for v in by.values()])
    out["all"] = {"clips": int(len(allv)), "exact_7": r1(100 * np.mean(allv == 7)),
                  "within_1": r1(100 * np.mean(np.abs(allv - 7) <= 1)), "mean_detected": r1(np.mean(allv))}
    for emo in list(by) + ["all"]:
        log(f"  {emo:10s} exactly 7: {out[emo]['exact_7']:5.1f}%  within 1: {out[emo]['within_1']:5.1f}%  "
            f"mean {out[emo]['mean_detected']}")
    return out


def part_hillenbrand(args, log, tuned):
    log("\n=== Hillenbrand 1995 (formants at the detected nucleus vs hand-checked steady state) ===")
    import bench_extractors as ex
    items = ds.load_hillenbrand()
    if args.quick:
        items = items[::6]
    C = list(fmt.AUTO_CEILINGS)
    rows, dt = [], []
    t0 = time.time()
    for it in items:
        y, sr = sf.read(it["path"], dtype="float32")
        y16 = to_16k(y, sr)
        sylls = run_detector(stream_frames(y16, tuned.band), tuned)
        sylls = [s for s in sylls if it["t_start"] - 0.05 <= s.t_peak <= it["t_end"] + 0.05] or sylls
        t = max(sylls, key=lambda s: s.peak_db).t_peak if sylls else it["t_ss"]
        dt.append(t - it["t_ss"])
        cands = fmt.measure_all(y16, t)
        A = np.array([cands[c][0] for c in C])                       # (ceilings, F1..F4)
        rows.append({"group": it["group"], "vowel": pronounce.HILLENBRAND_CODE.get(it["vowel"]),
                     "ref": np.array(it["formants"][:3], float), "A": A,
                     "auto": np.array(fmt.measure(y16, t, "auto")[0]),
                     "pitch": A[C.index(ex.adaptive_ceiling(it["f0"]))]})
    # what "speaker" mode settles on, per group (each Hillenbrand speaker has one token per vowel, so the
    # within-vowel spread is taken over the group): the ceiling that makes same-vowel tokens most alike
    group_ceiling = {}
    for g in sorted({r["group"] for r in rows}):
        sc = fmt.SpeakerCeiling(window=100000, min_tokens=1, min_vowels=1)
        for r in rows:
            if r["group"] == g and r["vowel"]:
                sc.tokens.append((r["vowel"], r["A"][:, :2]))
        sp = sc.spread()
        group_ceiling[g] = min(sp, key=sp.get)
    log("  'speaker' ceiling per group (smallest within-vowel spread): " +
        ", ".join(f"{g} {int(c)} Hz" for g, c in group_ceiling.items()))
    options = {"5000": lambda r: r["A"][C.index(5000.0)], "5500": lambda r: r["A"][C.index(5500.0)],
               "pitch-based": lambda r: r["pitch"], "auto": lambda r: r["auto"],
               "speaker": lambda r: r["A"][C.index(group_ceiling[r["group"]])]}
    groups = np.array([r["group"] for r in rows])
    out = {"files": len(items), "seconds": r1(time.time() - t0), "speaker_ceiling_per_group": group_ceiling,
           "nucleus_minus_steady_state_ms_median": r1(1000 * np.median(dt)),
           "nucleus_minus_steady_state_ms_mae": r1(1000 * np.mean(np.abs(dt)))}
    for k, get in options.items():
        e = np.array([np.where(r["ref"] > 0, np.abs(get(r)[:3] - r["ref"]) / np.where(r["ref"] > 0, r["ref"], 1),
                               np.nan) for r in rows])
        out[k] = {}
        for g in ("all", "men", "women", "boys", "girls"):
            sub = e if g == "all" else e[groups == g]
            out[k][g] = {f"F{i + 1}_within_10pct": r1(100 * np.nanmean(sub[:, i] <= 0.10)) for i in range(3)}
            out[k][g].update({f"F{i + 1}_median_err_pct": r1(100 * np.nanmedian(sub[:, i])) for i in range(3)})
        log(f"  ceiling {k:11s} within 10%: F1 {out[k]['all']['F1_within_10pct']}  F2 {out[k]['all']['F2_within_10pct']}"
            f"  F3 {out[k]['all']['F3_within_10pct']}   (men F2 {out[k]['men']['F2_within_10pct']}, "
            f"women F2 {out[k]['women']['F2_within_10pct']})")
    log(f"  detected nucleus vs Hillenbrand's steady-state time: median {out['nucleus_minus_steady_state_ms_median']} ms")
    measured = {}
    for r in rows:
        F = options["speaker"](r)
        if r["vowel"] and F[0] > 0 and F[1] > 0:
            measured.setdefault(r["group"], {}).setdefault(r["vowel"], []).append(F)
    return out, measured, group_ceiling


class TruthWords:
    """Stands in for Whisper with the true (forced-alignment) word times, for building reference targets."""

    def __init__(self, words):
        self.words = words

    def transcribe(self, audio, t0=0.0, prompt=None, max_tokens=None):
        t1 = t0 + len(audio) / SR
        ws = [{"word": w["word"], "start": w["start"], "end": w["end"], "prob": 1.0} for w in self.words
              if t0 <= (w["start"] + w["end"]) / 2 <= t1]
        return " ".join(w["word"] for w in ws), ws


def part_targets(args, log):
    """Running-speech targets: ARCTIC sentences read by 2 women and 2 men, measured exactly like a live
    session (same detector, same speaker-ceiling fit), labelled with the true word times."""
    log("\n=== running-speech targets (CMU ARCTIC read sentences, measured like your voice) ===")
    n = 60 if args.quick else 300
    per = {}
    for spk in ("slt", "clb", "bdl", "rms"):
        items = ds.load_arctic(speakers=(spk,), max_per_speaker=n, seed=11)
        raw, words = build_stream(items)
        y = to_16k(raw, SR)
        p = Pipeline(SR, PipelineConfig(), whisper=TruthWords(words), store_seconds=len(y) / SR + 5)
        for i in range(0, len(y), 800):
            p.feed16(y[i: i + 800])
        p.finish()
        tgt = vowelspace.target_from_tokens(p.tokens, spk)
        per[spk] = tgt
        v = tgt["vowels"]
        log(f"  {spk} ({ds.ARCTIC_SPEAKERS[spk]}): {sum(x['n'] for x in v.values())} chart tokens, speaker ceiling "
            f"{p.ceiling_now():.0f} Hz; /i/ F1 {v['IY']['F1']:.0f} F2 {v['IY']['F2']:.0f}, /u/ F2 "
            f"{v.get('UW', {}).get('F2', 0):.0f}, /oU/ F2 {v.get('OW', {}).get('F2', 0):.0f}, /a/ F1 {v['AA']['F1']:.0f}")
    out = {}
    for g, spks in (("women", ("slt", "clb")), ("men", ("bdl", "rms"))):
        vowels = {}
        for vw in pronounce.CHART_VOWELS:
            ms = [per[s]["vowels"][vw] for s in spks if vw in per[s]["vowels"]]
            if ms:
                vowels[vw] = {f: float(np.mean([m[f] for m in ms])) for f in ("F1", "F2", "F3", "F4")}
                vowels[vw]["sd"] = [float(np.mean([m["sd"][i] for m in ms])) for i in range(4)]
                vowels[vw]["n"] = int(sum(m["n"] for m in ms))
        out[f"arctic_{g}_measured"] = {
            "name": f"ARCTIC read sentences, {g} (measured like your voice)",
            "source": f"CMU ARCTIC speakers {' + '.join(spks)}, {n} sentences each: running speech measured by this "
                      "pipeline (speaker-fitted formant ceiling), labelled with the true word times",
            "vowels": vowels}
    return out


def write_targets(log, hill_measured=None, group_ceiling=None, arctic=None):
    """syllabic/data/hillenbrand_targets.json: running-speech ARCTIC targets, Hillenbrand measured by this
    pipeline, and Hillenbrand's published group means. Parts not re-run keep their previous entries."""
    targets = vowelspace.builtin_targets()
    if hill_measured is not None:
        targets.update(vowelspace.hillenbrand_published(os.path.join(ds.hillenbrand_dir(), "vowdata.dat")))
        for g, by in hill_measured.items():
            vowels = {}
            for v, rows in by.items():
                a = np.array(rows)
                med = [float(np.median(c[c > 0])) if np.any(c > 0) else 0.0 for c in a.T]
                sd = [float(np.std(c[c > 0])) if np.sum(c > 0) > 1 else 0.0 for c in a.T]
                vowels[v] = {"F1": med[0], "F2": med[1], "F3": med[2], "F4": med[3], "sd": sd, "n": len(rows)}
            targets[f"hillenbrand_{g}_measured"] = {
                "name": f"Hillenbrand 1995 {g}, isolated words (measured like your voice)",
                "source": "Hillenbrand et al. 1995 /hVd/ recordings, measured by this pipeline (formant ceiling "
                          f"{int(group_ceiling[g])} Hz, the group's speaker-fitted value)",
                "vowels": vowels}
    if arctic:
        targets.update(arctic)

    def order(k):
        return (0 if k.startswith("arctic") else 1 if k.endswith("_measured") else 2, k)
    targets = {k: targets[k] for k in sorted(targets, key=order)}
    os.makedirs(os.path.dirname(vowelspace.TARGET_FILE), exist_ok=True)
    with open(vowelspace.TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=1, ensure_ascii=False)
    log(f"  wrote {len(targets)} targets to {vowelspace.TARGET_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="arctic,noise,ravdess,hillenbrand,targets")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="use the DetectorParams defaults as they are")
    ap.add_argument("--whisper", default="small.en", help="Whisper model for the arctic part ('none' to skip)")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    folder = os.path.join(PROJ, "benchmark_reports", "syllables_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    ds.download_arctic(log=log)
    whisper = None
    if "arctic" in parts and args.whisper != "none":
        from syllabic.whisper_words import WhisperWords, warm_up
        whisper = WhisperWords(args.whisper)
        warm_up(whisper)
    results = {"started": datetime.now().isoformat(timespec="seconds"), "quick": args.quick,
               "whisper_model": args.whisper}
    tuned = DetectorParams()
    streams = None
    if "arctic" in parts or "noise" in parts:
        results["arctic"], tuned, streams = part_arctic(args, log, whisper if "arctic" in parts else None)
    if "noise" in parts:
        results["noise"] = part_noise(args, log, tuned, streams)
    if "ravdess" in parts:
        results["ravdess"] = part_ravdess(args, log, tuned, whisper)
    hill = gc = arctic = None
    if "hillenbrand" in parts:
        results["hillenbrand"], hill, gc = part_hillenbrand(args, log, tuned)
    if "targets" in parts:
        arctic = part_targets(args, log)
    if hill is not None or arctic is not None:
        write_targets(log, hill, gc, arctic)
    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    with open(os.path.join(folder, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {folder}")


if __name__ == "__main__":
    main()
