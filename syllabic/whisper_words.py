"""Whisper word timestamps, run in a separate PROCESS so the live window never slows it down.

What the first live sessions (audio_logs/syllables_live_20261105_*) showed:
  * In a background THREAD, Whisper ran 3-10x slower than on its own (the window's drawing competes for
    Python's interpreter lock), fell behind and never caught up: 41 and 216 phrases were never answered
    ("Whisper stops working if I talk long enough").
  * Sustained vowels and shouting ("eeee", "OHHHH") made it loop: when its output repeats it retries at
    higher temperatures, so single phrases took 2-4 minutes.
So now: its own process; temperature 0 only (no retries); a token budget from the number of detected
syllables (a phrase with 3 syllables cannot produce 200 tokens of "eeeee"); at most `max_backlog`
phrases waiting (older ones are skipped so it stays live); and a watchdog that restarts the process if
one phrase takes longer than `timeout_s`.

Word timestamps come from aligning the decoder's cross-attention to the audio (dynamic time warping).
Good to a few tens of ms inside a phrase; the FIRST word of a clip is pinned to the clip's start
(measured: median exactly -250 ms = the padding), so it is left out of the word-start score.
"""
import multiprocessing as mp
import queue
import threading
import time
import warnings
from collections import deque

import numpy as np

from .frontend import SR


def default_model():
    try:
        import torch
        return "small.en" if torch.cuda.is_available() else "base.en"
    except Exception:
        return "base.en"


def token_budget(n_syllables):
    """Max tokens Whisper may produce for a phrase: ~1 token per syllable in normal speech, x3 for safety."""
    return int(min(224, 12 + 3 * max(1, n_syllables)))


class WhisperWords:
    """Synchronous wrapper: transcribe(audio_16k, t0) -> (text, [{'word','start','end','prob'}])."""

    def __init__(self, model_name=None, device=None, language="en"):
        import torch
        import whisper
        self.model_name = model_name or default_model()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio, t0=0.0, prompt=None, max_tokens=None):
        audio = np.asarray(audio, dtype=np.float32)
        opts = dict(word_timestamps=True, language=self.language, fp16=self.device == "cuda",
                    condition_on_previous_text=False, initial_prompt=prompt, verbose=None, temperature=0.0)
        if max_tokens:
            opts["sample_len"] = int(min(224, max_tokens))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = self.model.transcribe(audio, **opts)
        words = []
        for seg in r.get("segments", []):
            for w in seg.get("words", []):
                text = w["word"].strip()
                if text:
                    words.append({"word": text, "start": t0 + float(w["start"]), "end": t0 + float(w["end"]),
                                  "prob": float(w.get("probability", 1.0))})
        return r.get("text", "").strip(), words


class SavedWords:
    """Stands in for Whisper when a saved session is opened again: each phrase gets the words Whisper gave
    for that stretch of audio when the session was recorded, so no model is needed and a session re-runs
    at ~30x real time. A word goes to the first phrase whose audio contains its middle, so phrases that
    come out cut slightly differently this time (newer code) still get the right words."""

    def __init__(self, words, source=""):
        self.words = sorted(words, key=lambda w: w["start"])
        self.mids = [(w["start"] + w["end"]) / 2 for w in self.words]
        self.i = 0
        self.error = None
        self.model_name = "saved words"
        self.source = source

    def transcribe(self, audio, t0=0.0, prompt=None, max_tokens=None):
        import bisect
        t1 = t0 + len(audio) / SR
        i = bisect.bisect_left(self.mids, t0, lo=self.i)   # words before this clip belong to no phrase now
        j = bisect.bisect_right(self.mids, t1, lo=i)
        self.i = j
        out = [dict(w) for w in self.words[i:j]]
        return " ".join(w["word"] for w in out), out

    @classmethod
    def from_csv(cls, path):
        """A session's words.csv (live or reprocessed)."""
        import csv
        words = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    words.append({"word": r["word"], "start": float(r["start_s"]), "end": float(r["end_s"]),
                                  "prob": float(r.get("whisper_prob") or 1.0)})
                except (KeyError, ValueError):
                    continue
        return cls(words, source=path)


def warm_up(ww):
    """The first real phrase would otherwise take several seconds: CUDA kernels and the numba-compiled
    DTW used for word timing are both built on first use."""
    try:
        import torch
        from whisper.timing import dtw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtw(torch.rand(30, 20))
        t = np.arange(SR) / SR
        vowel = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 20)) * 0.05
        ww.transcribe(vowel.astype(np.float32), max_tokens=20)
    except Exception:
        pass


def _worker_main(model_name, device, language, q_in, q_out, max_backlog):
    """Runs in the Whisper process."""
    warnings.simplefilter("ignore")
    try:
        ww = WhisperWords(model_name, device, language)
        warm_up(ww)
        q_out.put({"type": "ready", "device": ww.device, "model": ww.model_name})
    except Exception as e:  # model download / CUDA problems: the rest keeps working without words
        q_out.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
        return
    waiting = deque()
    while True:
        try:
            item = q_in.get(timeout=0.05) if waiting else q_in.get()
            while True:
                if item is None:
                    return
                waiting.append(item)
                item = q_in.get_nowait()
        except queue.Empty:
            pass
        if not waiting:
            continue
        while len(waiting) > max_backlog:          # fell behind: skip the oldest so it stays live
            utt, t0, audio, _ = waiting.popleft()
            q_out.put({"type": "result", "utt": utt, "t0": t0, "text": "", "words": [], "secs": None,
                       "error": None, "dropped": "behind"})
        utt, t0, audio, max_tokens = waiting.popleft()
        q_out.put({"type": "start", "utt": utt})
        t_start = time.perf_counter()
        try:
            text, words = ww.transcribe(audio, t0, max_tokens=max_tokens)
            err = None
        except Exception as e:
            text, words, err = "", [], f"{type(e).__name__}: {e}"
        q_out.put({"type": "result", "utt": utt, "t0": t0, "text": text, "words": words,
                   "secs": time.perf_counter() - t_start, "error": err, "dropped": None})


class WhisperWorker:
    """Whisper in its own process. submit(utt_id, t0, audio, max_tokens) and poll() for results.
    poll() must be called regularly (the live window does it every frame): it also runs the watchdog."""

    def __init__(self, model_name=None, device=None, language="en", max_backlog=2, timeout_s=30.0):
        self.model_name = model_name or "small.en"
        self.device, self.language = device, language
        self.max_backlog, self.timeout_s = max_backlog, timeout_s
        self.ready = threading.Event()
        self.error = None
        self.restarts = 0
        self._ctx = mp.get_context("spawn")
        self._outstanding = {}       # utt id -> submit time
        self._busy = None            # (utt id, time the process started on it)
        self._make_process()

    def _make_process(self):
        self.q_in, self.q_out = self._ctx.Queue(), self._ctx.Queue()
        self.proc = self._ctx.Process(target=_worker_main, daemon=True, name="whisper",
                                      args=(self.model_name, self.device, self.language, self.q_in, self.q_out,
                                            self.max_backlog))

    def start(self):
        self.proc.start()
        return self

    def submit(self, utt_id, t0, audio, max_tokens=None):
        self._outstanding[utt_id] = time.time()
        self.q_in.put((utt_id, t0, np.asarray(audio, dtype=np.float32), max_tokens))

    def pending(self):
        return len(self._outstanding)

    def lag(self):
        """Seconds the oldest unanswered phrase has been waiting."""
        return time.time() - min(self._outstanding.values()) if self._outstanding else 0.0

    def poll(self):
        out = []
        while True:
            try:
                m = self.q_out.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            if m["type"] == "ready":
                self.device, self.model_name = m["device"], m["model"]
                self.ready.set()
            elif m["type"] == "error":
                self.error = m["error"]
                self.ready.set()
            elif m["type"] == "start":
                self._busy = (m["utt"], time.time())
            elif m["type"] == "result":
                if self._busy and self._busy[0] == m["utt"]:
                    self._busy = None
                self._outstanding.pop(m["utt"], None)
                out.append(m)
        if self._busy and time.time() - self._busy[1] > self.timeout_s:
            out += self._restart()
        elif not self.error and not self.proc.is_alive() and self.proc.exitcode is not None:
            if self.ready.is_set():
                out += self._restart()
            else:
                # died while loading (a crash, out of memory, a broken install): say so, instead of showing
                # "loading..." forever while phrases pile up for a process that is gone
                self.error = f"the Whisper process exited while loading (exit code {self.proc.exitcode})"
                self.ready.set()
        if self.error and self._outstanding:
            # nothing will answer these any more: hand them back as errors so nobody waits for them
            out += [{"type": "result", "utt": u, "t0": None, "text": "", "words": [], "secs": None,
                     "error": self.error, "dropped": None} for u in self._outstanding]
            self._outstanding.clear()
        return out

    def _restart(self):
        """A phrase hung (or the process died): kill it, report what was waiting, start a fresh one."""
        lost = [{"type": "result", "utt": u, "t0": None, "text": "", "words": [], "secs": None, "error": None,
                 "dropped": "timeout"} for u in self._outstanding]
        try:
            self.proc.terminate()
            self.proc.join(2)
        except Exception:
            pass
        self._outstanding.clear()
        self._busy = None
        self.restarts += 1
        self.ready.clear()
        self._make_process()
        self.proc.start()
        return lost

    def stop(self):
        try:
            self.q_in.put(None)
            self.proc.join(3)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
