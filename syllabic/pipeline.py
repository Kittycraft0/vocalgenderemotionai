r"""The whole chain, used the same way by the live window, file mode and the evaluation:

  mic / file audio -> 16 kHz -> 10 ms frames -> syllable detector -> formants at each nucleus
                                           \-> phrase cutter (pauses) -> Whisper (own process) -> words
  words + dictionary + detected syllables -> alignment -> live scores + labelled vowel tokens

Times everywhere are seconds of audio since the start of the session (or file).
"""
import csv
import json
import os
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

import numpy as np

from . import formants as fmt
from . import pronounce
from .align import AlignParams, LiveScore, align
from .frontend import FRAME_S, SR, AudioStore, FrameAnalyzer, HighPass, Resampler
from .syllables import DetectorParams, SyllableDetector
from .whisper_words import token_budget

CEILS = fmt.AUTO_CEILINGS
CEIL_INDEX = {c: i for i, c in enumerate(CEILS)}


@dataclass
class PipelineConfig:
    detector: DetectorParams = field(default_factory=DetectorParams)
    align: AlignParams = field(default_factory=AlignParams)
    ceiling: object = "speaker"     # "speaker" (one per speaker, re-estimated live), "auto" (per vowel) or Hz
    end_pause_s: float = 0.45       # silence that ends a phrase (then it goes to Whisper)
    max_phrase_s: float = 25.0      # Whisper reads 30 s at a time; longer phrases get cut between syllables
    pad_s: float = 0.25             # audio added before/after a phrase for Whisper
    vad_margin_db: float = 10.0     # broadband level this far above the noise floor = someone talking
    exercise_rate: float = 0.8      # a phrase >= 1 s with <= 3 syllables at < this many per second is a held
                                    # vowel ("eeee"): no words to find, and those made Whisper loop for minutes
    tokens_need_count_ok: bool = True   # only label vowels of words whose syllable count came out right
    cursor: bool = False            # live formant cursor (GUI only)
    autosave_s: float = 120.0       # session files rewritten this often, so a crash loses at most this much
    wav_refresh_s: float = 30.0     # the audio file's header is brought up to date this often


class Pipeline:
    def __init__(self, in_sr=SR, config=None, whisper=None, store_seconds=240.0, save_dir=None, save_audio=True):
        """whisper: a WhisperWorker (live, asynchronous), a WhisperWords (offline, synchronous) or None.
        save_dir: folder for the session files; save_audio also records the 16 kHz audio there."""
        self.cfg = config or PipelineConfig()
        self.resampler = Resampler(in_sr)
        self.highpass = HighPass()
        self.frames = FrameAnalyzer(band=tuple(self.cfg.detector.band))
        self.det = SyllableDetector(self.cfg.detector)
        self.audio = AudioStore(store_seconds)
        self.whisper = whisper
        self.syllables = []
        self.words = []
        self.utts = []
        self.tokens = []
        self.score = LiveScore(window=100, tol=self.cfg.align.onset_tol_s)
        self.cursor = deque(maxlen=25)
        self.speaker = fmt.SpeakerCeiling() if self.cfg.ceiling == "speaker" else None
        self.ceiling_log = []               # (session time, new speaker ceiling)
        self.clip = deque(maxlen=250)       # (samples, clipped samples) per input block, ~5 s
        self.clip_total = [0, 0]
        self.info = {}                      # extra facts for summary.json (source, model, ...)
        self._pending_fmt = deque()
        self._phrase_start = None
        self._last_speech = None
        self._next_cursor = 0.0
        self._finished = False
        self.whisper_secs = deque(maxlen=20)
        self.listeners = []                 # callables(event, payload), e.g. the GUI
        self.save_dir = save_dir
        self._wav = None
        self.wav_path = None                # the session's 16 kHz recording (the window plays older audio from it)
        self._last_autosave = self._last_wav_refresh = time.time()
        if save_dir and save_audio:
            import soundfile as sf
            os.makedirs(save_dir, exist_ok=True)
            self.wav_path = os.path.join(save_dir, "audio_16k.wav")
            self._wav = sf.SoundFile(self.wav_path, "w", SR, 1, "PCM_16")

    # ----------------------------------------------------------------------------------------------
    @property
    def now(self):
        """Seconds of audio received so far."""
        return self.audio.total / SR

    @property
    def clip_recent(self):
        """Fraction of recent input samples at full scale (clipping distorts the formants)."""
        n = sum(a for a, _ in self.clip)
        return sum(b for _, b in self.clip) / n if n else 0.0

    def ceiling_now(self):
        if self.speaker is not None:
            return self.speaker.current
        if self.cfg.ceiling == "auto":
            cs = [s.ceiling for s in self.syllables[-20:] if s.ceiling]
            return float(np.median(cs)) if cs else 5500.0
        return float(self.cfg.ceiling)

    def poll(self):
        """Collects finished Whisper phrases (the GUI calls this every frame)."""
        self._collect_whisper()

    def feed(self, block):
        """Raw samples (mono, any rate given at construction)."""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(x):
            k = int(np.count_nonzero(np.abs(x) >= 0.999))
            self.clip.append((len(x), k))
            self.clip_total[0] += len(x)
            self.clip_total[1] += k
        self.feed16(self.highpass.process(self.resampler.process(x)))

    def feed16(self, x16):
        if len(x16) == 0:
            return
        self.audio.append(x16)
        if self._wav is not None:
            self._wav.write(np.clip(x16, -1, 1))
        self._on_frames(self.frames.push(x16))
        self._measure_pending()
        if self.cfg.cursor:
            self._update_cursor()
        self._collect_whisper()
        self._maybe_autosave()

    def finish(self, wait=True, timeout=60.0):
        """End of input: flush everything, close the last phrase and (optionally) wait for Whisper.
        Safe to call again later (then it only waits for / collects Whisper's remaining phrases)."""
        if not self._finished:
            self._finished = True
            tail = self.resampler.process(np.zeros(0, np.float32), last=True)
            if len(tail):
                self.feed16(self.highpass.process(tail))
            fr = self.frames.flush()
            self._on_frames(fr, final=True)
            self._measure_pending(force=True)
            if self._phrase_start is not None:
                self._close_phrase(self._last_speech if self._last_speech is not None else self.now)
        if wait and self.whisper is not None and hasattr(self.whisper, "poll"):
            t0 = time.time()
            while any(u["status"] == "sent" for u in self.utts) and time.time() - t0 < timeout:
                time.sleep(0.05)
                self._collect_whisper()
        self._collect_whisper()
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    # ----------------------------------------------------------------------------------------------
    def _emit(self, event, payload):
        for cb in self.listeners:
            try:
                cb(event, payload)
            except Exception:
                pass

    def _on_frames(self, fr, final=False):
        new = self.det.push(fr) if fr is not None else []
        if final:
            new += self.det.flush()
        for s in new:
            self.syllables.append(s)
            self._pending_fmt.append(s)
            self._emit("syllable", s)
        if fr is not None:
            self._cut_phrases(fr)

    def _measure_pending(self, force=False):
        span = 0.08
        while self._pending_fmt and (force or self._pending_fmt[0].t_peak + span <= self.now):
            s = self._pending_fmt.popleft()
            if self.speaker is not None:
                m = fmt.measure_all(self.audio, s.t_peak, CEILS)
                cands = np.array([[m[c][0], m[c][1]] for c in CEILS], dtype=np.float32)   # (ceiling, F/B, 4)
                s.extra["cands"] = cands
                c = self.speaker.current
                F, B = cands[CEIL_INDEX[c], 0], cands[CEIL_INDEX[c], 1]
            else:
                F, B, c = fmt.measure(self.audio, s.t_peak, self.cfg.ceiling)
            s.formants, s.bandwidths, s.ceiling = [float(f) for f in F], [float(b) for b in B], float(c)
            self._emit("formants", s)

    def _update_cursor(self):
        """Live F1/F2 of whatever you are saying right now (only while it is voiced and loud enough)."""
        t = self.now - 0.07
        if t < self._next_cursor or self.det.ref is None:
            return
        self._next_cursor = t + 0.04
        k = int(round(t / FRAME_S))
        if not (self.det.h.base <= k < self.det.h.end):
            return
        if self.det.h.get("per", k) < self.cfg.detector.voicing + 0.1 or self.det.h.get("band", k) < self.det.ref - 18:
            return
        F, _, _ = fmt.measure(self.audio, t, self.ceiling_now(), half=0.01)
        if F[0] > 0 and F[1] > 0:
            self.cursor.append((t, *F))

    def _apply_ceiling(self, c):
        """The speaker ceiling changed: re-express every stored measurement at the new ceiling."""
        self.ceiling_log.append((round(self.now, 2), c))
        i = CEIL_INDEX[c]
        for s in self.syllables:
            cands = s.extra.get("cands")
            if cands is not None:
                s.formants, s.bandwidths, s.ceiling = [float(x) for x in cands[i, 0]], [float(x) for x in cands[i, 1]], c
        for tok in self.tokens:
            cands = tok.get("cands")
            if cands is not None:
                F, B = cands[i, 0], cands[i, 1]
                tok.update(F1=float(F[0]), F2=float(F[1]), F3=float(F[2]), F4=float(F[3]), B1=float(B[0]),
                           B2=float(B[1]), B3=float(B[2]), B4=float(B[3]), ceiling=c,
                           vtl_cm=fmt.apparent_vtl_cm(F))
        self._emit("ceiling", c)

    # ---- phrases for Whisper --------------------------------------------------------------------
    def _cut_phrases(self, fr):
        if self.det.bfloor is None:
            return
        thr = self.det.bfloor + self.cfg.vad_margin_db
        for t, b in zip(fr["t"], fr["broad"]):
            if b >= thr:
                if self._phrase_start is None:
                    self._phrase_start = float(t)
                self._last_speech = float(t)
            elif self._phrase_start is not None and t - self._last_speech >= self.cfg.end_pause_s:
                self._close_phrase(self._last_speech)
            if self._phrase_start is not None and t - self._phrase_start >= self.cfg.max_phrase_s:
                cut = self._best_cut(float(t))
                self._close_phrase(cut, next_start=cut)

    def _best_cut(self, t):
        """Where to split a phrase that is too long: at the start of a syllable (a valley between two
        vowels) in the last 4 s, the deepest one. Cutting at the quietest frame instead split words."""
        best = None
        for s in reversed(self.syllables):
            if s.t_peak < t - 4.0:
                break
            if t - 4.0 <= s.t_onset <= t - 0.3 and s.t_onset > (self._phrase_start or 0) + 1.0:
                k = int(round(s.t_onset / FRAME_S))
                if self.det.h.base <= k < self.det.h.end:
                    lvl = self.det.h.get("sband", k)
                    if np.isfinite(lvl) and (best is None or lvl < best[0]):
                        best = (lvl, s.t_onset)
        if best is not None:
            return best[1]
        k0, k1 = int(round((t - 2.0) / FRAME_S)), int(round(t / FRAME_S))
        sb = self.det.h.view("sband", k0, k1)
        sb = np.where(np.isnan(sb), np.inf, sb)
        return (k0 + int(np.argmin(sb))) * FRAME_S if len(sb) else t

    def _phrase_nuclei(self, t0, t1):
        """Nuclei of a phrase. Re-checks loudness now that the phrase is over: at the very start of a
        session the 'loudest speech' reference does not exist yet, so a quiet noise peak can slip
        through the live check; those get status 'quiet' and are left out."""
        out = []
        for s in reversed(self.syllables):   # newest first: phrases are recent, so this stops early
            if s.t_peak < t0 - 0.05:
                break
            if s.t_peak <= t1 + 0.05 and s.word_i is None and s.status not in ("quiet", "exercise"):
                if self.det.ref is not None and s.peak_db < self.det.ref - self.cfg.detector.silence_rel_db:
                    s.status = "quiet"
                else:
                    out.append(s)
        return out[::-1]

    def _close_phrase(self, t_end, next_start=None):
        t0, t1 = self._phrase_start, t_end
        self._phrase_start, self._last_speech = next_start, next_start
        if t0 is None or t1 <= t0:
            return
        nuclei = self._phrase_nuclei(t0, t1)
        u = {"id": len(self.utts), "t0": t0, "t1": t1, "status": "skipped", "text": "", "n_nuclei": len(nuclei),
             "secs": None, "words": []}
        self.utts.append(u)
        dur = t1 - t0
        if nuclei and dur >= 1.0 and len(nuclei) <= 3 and len(nuclei) / dur < self.cfg.exercise_rate:
            u["status"] = "exercise"        # a held vowel: the live cursor shows it, Whisper has nothing to read
            for s in nuclei:
                s.status = "exercise"
            self._emit("phrase", u)
            return
        whisper_ok = self.whisper is not None and not getattr(self.whisper, "error", None)
        if not nuclei or not whisper_ok:
            if nuclei:
                u["status"] = "no whisper"
            return
        a0 = max(0.0, t0 - self.cfg.pad_s)
        audio = self.audio.get(a0, t1 + self.cfg.pad_s)
        budget = token_budget(len(nuclei))
        u["status"] = "sent"
        if hasattr(self.whisper, "submit"):
            self.whisper.submit(u["id"], a0, audio, budget)
        else:
            t_start = time.perf_counter()
            text, words = self.whisper.transcribe(audio, a0, max_tokens=budget)
            self._apply_words({"utt": u["id"], "text": text, "words": words,
                               "secs": time.perf_counter() - t_start, "error": None})
        self._emit("phrase", u)

    def _collect_whisper(self):
        if self.whisper is None or not hasattr(self.whisper, "poll"):
            return
        for r in self.whisper.poll():
            if 0 <= r["utt"] < len(self.utts):
                self._apply_words(r)

    def _apply_words(self, r):
        u = self.utts[r["utt"]]
        if r.get("dropped"):
            u["status"] = "behind" if r["dropped"] == "behind" else "timeout"
            self._emit("words", {"utt": u, "results": [], "tokens": []})
            return
        u["status"], u["text"], u["secs"] = "done", r["text"], r["secs"]
        if r["secs"] is not None:
            self.whisper_secs.append(r["secs"])
        if r.get("error"):
            u["status"], u["text"] = "error", r["error"]
            return
        nuclei = self._phrase_nuclei(u["t0"], u["t1"])
        res = align(r["words"], nuclei, self.cfg.align, gi0=len(self.words), utt=u["id"])
        u["words"] = list(range(len(self.words), len(self.words) + len(res)))
        self.words += res
        self.score.add(res)
        new_tokens, changed = [], False
        for w in res:
            if not w.scored or (self.cfg.tokens_need_count_ok and not w.count_ok):
                continue
            for j, n in enumerate(w.nuclei):
                if n is None or not n.formants or n.formants[0] <= 0 or n.formants[1] <= 0:
                    continue
                syl = w.pron.syllables[j]
                ok, reason = pronounce.chart_eligibility(syl, w.word)
                tok = {"t": n.t_peak, "utt": u["id"], "word": pronounce.normalize(w.word), "word_prob": w.prob,
                       "syllable": syl.ipa, "vowel": syl.vowel, "ipa": syl.vowel_ipa, "stress": syl.stress,
                       "chart": ok, "reason": reason, "F1": n.formants[0], "F2": n.formants[1],
                       "F3": n.formants[2], "F4": n.formants[3], "B1": n.bandwidths[0], "B2": n.bandwidths[1],
                       "B3": n.bandwidths[2], "B4": n.bandwidths[3], "f0": n.f0, "ceiling": n.ceiling,
                       "peak_db": n.peak_db, "periodicity": n.per,
                       "vtl_cm": fmt.apparent_vtl_cm(n.formants), "outlier": False,
                       "cands": n.extra.get("cands")}
                new_tokens.append(tok)
                if ok and self.speaker is not None and tok["cands"] is not None:
                    changed |= self.speaker.add(syl.vowel, tok["cands"][:, 0, :2])
        self.tokens += new_tokens
        if changed:
            self._apply_ceiling(self.speaker.current)
        self._emit("words", {"utt": u, "results": res, "tokens": new_tokens})

    # ---- saving ---------------------------------------------------------------------------------
    def _maybe_autosave(self):
        if not self.save_dir:
            return
        now = time.time()
        if self._wav is not None and now - self._last_wav_refresh >= self.cfg.wav_refresh_s:
            self._last_wav_refresh = now
            self._refresh_wav()
        if now - self._last_autosave >= self.cfg.autosave_s:
            self._last_autosave = now
            try:
                self.save()
            except Exception as e:
                print(f"autosave failed: {type(e).__name__}: {e}")

    def _refresh_wav(self):
        """Close and reopen the recording so its header is valid even if the program is killed later
        (the 11-09 session's audio was never finalised because the window never closed normally)."""
        import soundfile as sf
        path = self._wav.name
        self._wav.close()
        self._wav = sf.SoundFile(path, "r+")
        self._wav.seek(0, sf.SEEK_END)

    def summary(self):
        status = Counter(u["status"] for u in self.utts)
        n, k = self.clip_total
        info = {"summary_recent": self.score.summary("recent"), "summary_session": self.score.summary("all"),
                "phrase_status": dict(status),
                "whisper_restarts": getattr(self.whisper, "restarts", 0),
                "clipped_input_pct": round(100.0 * k / n, 4) if n else None,
                "ceiling_mode": self.cfg.ceiling, "ceiling_final": self.ceiling_now(),
                "speaker_ceiling_changes": self.ceiling_log,
                "speaker_ceiling_spread_bark": getattr(self.speaker, "last_spread", None),
                "config": {"detector": asdict(self.cfg.detector), "align": asdict(self.cfg.align),
                           "end_pause_s": self.cfg.end_pause_s, "max_phrase_s": self.cfg.max_phrase_s},
                "time_meaning": "seconds of audio since the start (frame times are window CENTRES)"}
        info.update(self.info)
        info["phrases"] = [{k: v for k, v in u.items() if k != "words"} for u in self.utts]
        return info

    def save(self, folder=None, extra_info=None):
        folder = folder or self.save_dir
        if not folder:
            return None
        if extra_info:
            self.info.update(extra_info)
        os.makedirs(folder, exist_ok=True)
        from .vowelspace import save_tokens
        with open(os.path.join(folder, "syllables.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "onset_s", "peak_s", "end_s", "after_pause", "peak_db", "periodicity", "f0",
                        "F1", "F2", "F3", "F4", "ceiling", "status", "word_i", "syl_i", "label", "vowel"])
            for s in list(self.syllables):
                F = s.formants or [0, 0, 0, 0]
                w.writerow([s.idx, round(s.t_onset, 3), round(s.t_peak, 3),
                            "" if s.t_end is None else round(s.t_end, 3), s.after_pause, round(s.peak_db, 2),
                            round(s.per, 3), round(s.f0, 1), *[round(x, 1) for x in F], s.ceiling, s.status,
                            s.word_i, s.syl_i, s.label or "", s.vowel or ""])
        with open(os.path.join(folder, "words.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gi", "utt", "word", "start_s", "end_s", "whisper_prob", "ipa", "syll_expected",
                        "syll_detected", "count_ok", "first_syllable_onset_err_ms", "after_pause",
                        "phrase_initial", "in_dictionary"])
            for r in list(self.words):
                e = r.onset_err
                w.writerow([r.gi, r.utt, r.word, round(r.start, 3), round(r.end, 3), round(r.prob, 3),
                            r.pron.ipa if r.pron else "", r.n_exp, r.n_det, r.count_ok if r.scored else "",
                            "" if e is None else round(1000 * e, 1), r.after_pause, r.phrase_initial, r.scored])
        save_tokens(list(self.tokens), os.path.join(folder, "tokens.csv"))
        with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=1, default=str, ensure_ascii=False)
        return folder


def run_file(y, sr, whisper=None, config=None, chunk_s=0.05):
    """Feeds a whole recording through the live pipeline in mic-sized blocks (same code path as live)."""
    p = Pipeline(sr, config, whisper=whisper, store_seconds=len(y) / sr + 5)
    n = max(1, int(sr * chunk_s))
    for i in range(0, len(y), n):
        p.feed(y[i: i + n])
    p.finish()
    return p
