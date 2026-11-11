"""main14: live syllabic transcription, checked against Whisper, feeding a live vowel-space chart.

  python main14livesyllables.py                        live microphone (asks which input device)
  python main14livesyllables.py --device 1             live microphone, device 1 (see --list-devices)
  python main14livesyllables.py --target arctic_women_measured
  python main14livesyllables.py --input talk.wav       a recording through the live window (--play to hear it)
  python main14livesyllables.py --input talk.wav --headless      no window: CSVs + summary only
  python main14livesyllables.py --open talk.wav        review a recording: processed as fast as possible
                                                       (Whisper on every phrase), then scroll, hover, listen
  python main14livesyllables.py --open audio_logs/syllables_live_20261110_084650
                                                       review a saved session with the words it already has
                                                       (no Whisper needed; ~30x real time)
  python main14livesyllables.py --reprocess audio_logs/syllables_live_20261109_195418
                                                       re-run a saved session (results in <folder>/reprocessed)
  python main14livesyllables.py --make-target voice.wav --name "Voice I want"   build a target
  python main14livesyllables.py --list-targets
  python main14livesyllables.py --history audio_logs/syllables_live_20261105_120000   keep building on a session

How it works (details in syllabic/README.md and benchmark/eval_syllables.py):
  * syllables are found live from the vowel-band loudness (peaks = vowels, dips = consonants), ~0.1 s
    after each vowel, with F1-F4 measured at the vowel's loudest point (Praat Burg, one formant ceiling
    fitted to your voice);
  * after each pause, Whisper transcribes the phrase with word times (in its own process);
  * each word's dictionary syllables (CMUdict) are lined up with the detected ones, which gives
    the live accuracy scores AND tells which vowel each detected syllable was meant to be;
  * those labelled vowels build up your vowel space (recent tokens count more), next to a target.
Session files go to audio_logs/syllables_<source>_<date>_<time>/ (audio, syllables.csv, words.csv,
tokens.csv, summary.json) and are re-saved every 2 minutes, so a crash loses at most that much.
The window keeps the whole session: drag the timeline back, point at it to see the word, syllable and
formants at that moment, click a word to hear it (while paused), "Open…" reviews another recording.
"""
import argparse
import os
import struct
import sys
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from syllabic import vowelspace  # noqa: E402
from syllabic.pipeline import Pipeline, PipelineConfig  # noqa: E402

LOG_ROOT = os.path.join(HERE, "audio_logs")


def list_inputs():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], sd.query_hostapis(d["hostapi"])["name"], int(d["default_samplerate"])))
    return out


def prompt_device():
    import sounddevice as sd
    default = sd.default.device[0]
    print("\nInput devices:")
    for i, name, api, sr in list_inputs():
        print(f"  {i:3d}  {name}  [{api}, {sr} Hz]{'   <- default' if i == default else ''}")
    while True:
        txt = input(f"Device number (Enter for {default}): ").strip()
        if not txt:
            return default
        if txt.isdigit() and any(i == int(txt) for i, *_ in list_inputs()):
            return int(txt)
        print("  not an input device number, try again")


def load_audio(path):
    import soundfile as sf
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        return y.mean(axis=1), sr
    except Exception:
        import librosa  # mp3/m4a etc.
        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), sr


def audio_blocks(path, seconds=1.0):
    """(sample rate, generator of mono float32 blocks): streams from disk so long sessions fit in memory."""
    import soundfile as sf
    try:
        info = sf.info(path)
        sr = info.samplerate

        def gen():
            for b in sf.blocks(path, blocksize=int(sr * seconds), dtype="float32", always_2d=True):
                yield b.mean(axis=1)
        return sr, gen(), info.duration
    except Exception:
        y, sr = load_audio(path)
        return sr, (y[i: i + int(sr * seconds)] for i in range(0, len(y), int(sr * seconds))), len(y) / sr


def repair_wav_header(path):
    """A recording whose program was killed before it closed the file has a header that says 'no data'
    (the 11-09 session). The samples are all there; this rewrites the two size fields so every program
    (Praat, Audacity, ...) can open it. Only touches files whose header is clearly unfinished."""
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        head = f.read(4096)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        i = head.find(b"data", 12)
        if i < 0:
            return False
        riff = struct.unpack("<I", head[4:8])[0]
        data = struct.unpack("<I", head[i + 4:i + 8])[0]
        real = size - (i + 8)
        if riff == size - 8 and data == real:
            return False
        if size - 8 >= 2 ** 32:
            print(f"{path} is over 4 GB, too big for a plain WAV header; leaving it as is")
            return False
        f.seek(4)
        f.write(struct.pack("<I", size - 8))
        f.seek(i + 4)
        f.write(struct.pack("<I", real - real % 2))
    print(f"repaired the unfinished WAV header of {path} ({real / 32000 / 60:.1f} min of audio)")
    return True


def session_folder(source):
    base = "live" if source is None else os.path.splitext(os.path.basename(source))[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)[:60]
    return os.path.join(LOG_ROOT, f"syllables_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def make_config(args):
    c = args.ceiling
    return PipelineConfig(ceiling=c if c in ("speaker", "auto") else float(c))


def load_history(paths):
    toks = []
    for p in paths or []:
        f = os.path.join(p, "tokens.csv") if os.path.isdir(p) else p
        toks += vowelspace.load_tokens(f)
        print(f"history: {f}")
    return toks


def run_offline(path, args, folder=None):
    """Whole file through the same pipeline, Whisper run synchronously; returns the pipeline."""
    from syllabic import pronounce
    from syllabic.whisper_words import WhisperWords, warm_up
    pronounce.preload()
    sr, blocks, dur = audio_blocks(path)
    print(f"{path}: {dur / 60:.1f} min at {sr} Hz")
    ww = None
    if args.whisper_model != "none":
        print(f"loading Whisper '{args.whisper_model}' ...")
        ww = WhisperWords(args.whisper_model)
        warm_up(ww)
    p = Pipeline(sr, make_config(args), whisper=ww, store_seconds=240.0, save_dir=folder, save_audio=False)
    p.info.update({"source": os.path.abspath(path), "whisper_model": args.whisper_model})
    t0, last = time.time(), time.time()
    for b in blocks:
        for i in range(0, len(b), max(1, int(sr * 0.05))):
            p.feed(b[i: i + int(sr * 0.05)])
        if time.time() - last > 10:
            last = time.time()
            print(f"  {p.now / 60:5.1f} / {dur / 60:.1f} min  ({len(p.words)} words so far)", flush=True)
    p.finish()
    print(f"processed in {(time.time() - t0) / 60:.1f} min")
    return p


def print_summary(p, last=12):
    s = p.score.summary("all")
    print(f"\n{len(p.syllables)} syllables, {len(p.words)} words, {len(p.tokens)} vowel tokens "
          f"({sum(1 for t in p.tokens if t['chart'])} for the chart); formant ceiling {p.ceiling_now():.0f} Hz")
    if s.get("words"):
        hit = f"{s['onset_hit']:.0f}%" if s.get("onset_hit") is not None else "-"
        print(f"syllable count right on {s['count_exact']:.0f}% of words; word starts matched (+-100 ms, not counting "
              f"phrase-first words) {hit}; dictionary syllables found {s['recall']:.0f}%; detected syllables "
              f"that belong to a word {s['precision']:.0f}%")
    from collections import Counter
    print("phrases:", dict(Counter(u["status"] for u in p.utts)))
    shown = [u for u in p.utts if u["text"]][-last:]
    for u in shown:
        ws = [p.words[i] for i in u["words"]]
        line = "  ".join(f"{w.word}/{'.'.join(x.ipa for x in w.pron.syllables)}/{'' if w.count_ok else '*'}"
                         if w.scored else f"{w.word}/?/" for w in ws)
        print(f"  {u['t0']:7.2f}s  {line}")
    if shown:
        print("  (* = detected syllable count differs from the dictionary)")


def make_target(path, name, args):
    p = run_offline(path, args)
    tgt = vowelspace.target_from_tokens(p.tokens, name, source=f"{os.path.basename(path)}, measured by main14 "
                                        f"(formant ceiling {p.ceiling_now():.0f} Hz)")
    fn = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_") or "target"
    out = vowelspace.save_target(tgt, os.path.join(vowelspace.USER_TARGET_DIR, fn + ".json"))
    from syllabic.pronounce import CHART_VOWELS, IPA, KEYWORD
    print(f"\nTarget '{name}' -> {out}")
    print("  vowel        n     F1     F2     F3")
    for v in CHART_VOWELS:
        m = tgt["vowels"].get(v)
        if m:
            print(f"  {IPA[v]:3s} {KEYWORD[v]:7s} {m['n']:3d}  {m['F1']:5.0f}  {m['F2']:5.0f}  {m['F3']:5.0f}"
                  + ("   (few tokens: record more speech with this vowel)" if m["n"] < 3 else ""))
    missing = [f"{IPA[v]} ({KEYWORD[v]})" for v in CHART_VOWELS if v not in tgt["vowels"]]
    if missing:
        print("  no tokens for: " + ", ".join(missing))


def reprocess(folder, args):
    wav = os.path.join(folder, "audio_16k.wav")
    if not os.path.exists(wav):
        print(f"no audio_16k.wav in {folder}")
        return
    repair_wav_header(wav)
    out = os.path.join(folder, "reprocessed")
    p = run_offline(wav, args, out)
    p.save(extra_info={"reprocessed_from": os.path.abspath(wav)})
    print(f"\nresults in {out}")
    print_summary(p)


SESSION_FILES = ("audio_16k.wav", "summary.json", "words.csv", "tokens.csv", "syllables.csv")


def resolve_open(path):
    """--open PATH: a recording, or a saved session (its folder or any file in it).
    Returns (audio file, words.csv to reuse or None, session folder or None)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        folder = path
    elif os.path.basename(path).lower() in SESSION_FILES:
        folder = os.path.dirname(path)
    else:
        return path, None, None
    session = folder
    if not os.path.exists(os.path.join(folder, "audio_16k.wav")) and \
            os.path.exists(os.path.join(os.path.dirname(folder), "audio_16k.wav")):
        session = os.path.dirname(folder)          # e.g. <session>/reprocessed: the audio is one level up
    wav = os.path.join(session, "audio_16k.wav")
    if not os.path.exists(wav):
        raise SystemExit(f"no audio_16k.wav in {folder}")
    repair_wav_header(wav)
    # the reprocessed words (every phrase transcribed) beat the live ones (Whisper may have skipped some)
    cands = [os.path.join(folder, "words.csv")] if folder != session else \
        [os.path.join(session, "reprocessed", "words.csv"), os.path.join(session, "words.csv")]
    return wav, next((c for c in cands if os.path.exists(c)), None), session


def run_review(args):
    """--open: a recording or a saved session in the window, processed as fast as the computer allows.
    The timeline fills in while you can already scroll, point and listen."""
    import signal
    import pyqtgraph as pg
    from syllabic import pronounce
    from syllabic.gui import FastFileSource, LiveWindow
    from syllabic.whisper_words import SavedWords, WhisperWorker
    pronounce.preload()
    audio, words_csv, session = resolve_open(args.open)
    saved = SavedWords.from_csv(words_csv) if words_csv else None
    whisper = None
    if saved is None and args.whisper_model != "none":
        # reviewing: every phrase is transcribed (no skipping to stay live); the file waits for Whisper instead
        whisper = WhisperWorker(args.whisper_model, max_backlog=10 ** 9, timeout_s=60.0).start()
        print(f"Whisper '{whisper.model_name}' is loading in its own process.")
    name = os.path.basename(session) if session else os.path.basename(audio)
    src = FastFileSource(audio, name, whisper=whisper)
    if args.no_save or saved is not None:
        folder = None                                  # a saved session with its words: look only, write nothing
    elif session:
        folder = os.path.join(session, "reprocessed")  # a session without words: same place as --reprocess
    else:
        folder = session_folder(audio)
    p = Pipeline(src.sr, make_config(args), whisper=saved if saved is not None else whisper, save_dir=folder,
                 save_audio=folder is not None and session is None)
    p.info.update({"source": audio, "mode": "review", "words_from": words_csv,
                   "whisper_model": "saved words" if saved is not None else args.whisper_model,
                   "started": datetime.now().isoformat(timespec="seconds")})
    note = (f"words from {os.path.relpath(words_csv, session)} ({len(saved.words)} words, no Whisper needed)"
            if saved is not None else None)
    app = pg.mkQApp("Review")
    closed = []

    def on_close():
        if closed:
            return
        closed.append(True)
        if not src.done:
            print(f"closed at {src.pos / src.sr / 60:.1f} of {src.n / src.sr / 60:.1f} min: what was processed "
                  "so far is kept")
        for step in (src.stop, lambda: p.finish(wait=whisper is not None, timeout=10)):
            try:
                step()
            except Exception as e:
                print(f"while closing: {type(e).__name__}: {e}")
        if folder:
            try:
                p.save()
                print(f"\nSaved in {folder}")
            except Exception as e:
                print(f"saving failed: {type(e).__name__}: {e}")
        print_summary(p)
        if whisper:
            whisper.stop()

    win = LiveWindow(p, src.start(), theme=args.theme, target_key=args.target, title=f"Review  —  {name}",
                     on_close=on_close, whisper=whisper, review=True, words_note=note,
                     open_cmd=[sys.executable, os.path.abspath(__file__)])
    signal.signal(signal.SIGINT, lambda *a: win.close())
    win.show()
    print(f"Reviewing {audio}" + (f" with {note}" if note else "") + ". Drag/scroll the timeline, point at it, "
          "click a word to hear it.")
    try:
        app.exec()
    finally:
        on_close()


def run_gui(args):
    import signal
    import pyqtgraph as pg
    from syllabic import pronounce
    from syllabic.gui import FileSource, LiveWindow, MicSource
    from syllabic.whisper_words import WhisperWorker
    pronounce.preload()   # ~1 s now rather than a freeze when the first words come back
    app = pg.mkQApp("Live syllables")
    whisper = None
    if args.whisper_model != "none":
        whisper = WhisperWorker(args.whisper_model).start()
        print(f"Whisper '{whisper.model_name}' is loading in its own process (syllables work right away).")
    if args.input:
        y, sr = load_audio(args.input)
        src = FileSource(y, sr, os.path.basename(args.input), play=args.play)
        name = os.path.basename(args.input)
    else:
        import sounddevice as sd
        dev = args.device if args.device is not None else prompt_device()
        info = sd.query_devices(dev)
        sr = int(info["default_samplerate"])
        src = MicSource(dev, sr, info["name"])
        name = info["name"]
    folder = None if args.no_save else session_folder(args.input)
    cfg = make_config(args)
    cfg.cursor = True
    p = Pipeline(sr, cfg, whisper=whisper, save_dir=folder, save_audio=args.input is None)
    p.info.update({"source": args.input or f"microphone: {name}", "sample_rate_in": sr,
                   "whisper_model": args.whisper_model, "target": args.target,
                   "started": datetime.now().isoformat(timespec="seconds")})
    history = load_history(args.history)
    closed = []

    def on_close():
        if closed:
            return
        closed.append(True)
        for step in (src.stop, lambda: p.finish(wait=True, timeout=10)):
            try:
                step()
            except Exception as e:
                print(f"while closing: {type(e).__name__}: {e}")
        if folder:
            try:
                p.save()
                print(f"\nSession saved in {folder}")
            except Exception as e:
                print(f"saving failed: {type(e).__name__}: {e}")
        print_summary(p)
        if whisper:
            whisper.stop()

    win = LiveWindow(p, src.start(), theme=args.theme, target_key=args.target, history=history,
                     title=f"Live syllables & vowel space  —  {name}", on_close=on_close, whisper=whisper,
                     open_cmd=[sys.executable, os.path.abspath(__file__)])
    # Ctrl+C in the terminal: close the window normally (saves the session) instead of aborting
    signal.signal(signal.SIGINT, lambda *a: win.close())
    win.show()
    print("Window open. Speak normally; pause briefly between phrases so Whisper can label them. "
          "Space = pause; close the window (or Ctrl+C here) to stop and save.")
    try:
        app.exec()
    finally:
        on_close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # IPA symbols in the Windows console
    except Exception:
        pass
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, help="microphone device number (default: ask)")
    ap.add_argument("--input", help="audio file to run instead of the microphone")
    ap.add_argument("--play", action="store_true", help="with --input: play the file while analysing it")
    ap.add_argument("--headless", action="store_true", help="with --input: no window, just files + summary")
    ap.add_argument("--open", metavar="PATH", help="review a recording, or a saved session folder (or any file "
                    "in it), in the window: processed as fast as possible, then scroll/hover/listen")
    ap.add_argument("--reprocess", metavar="FOLDER", help="re-run a saved session folder's audio_16k.wav")
    ap.add_argument("--make-target", metavar="WAV", help="build a vowel-space target from a recording")
    ap.add_argument("--name", default=None, help="name for --make-target")
    ap.add_argument("--target", default=None, help="target key (see --list-targets) or a target .json")
    ap.add_argument("--history", action="append", help="earlier session folder or tokens.csv to start from")
    ap.add_argument("--whisper-model", default="small.en", help="tiny.en/base.en/small.en/medium.en/turbo, or none")
    ap.add_argument("--ceiling", default="speaker",
                    help="formant ceiling: speaker (fitted to your voice, default), auto (per vowel) or Hz")
    ap.add_argument("--theme", choices=("dark", "light"), default="dark")
    ap.add_argument("--no-save", action="store_true", help="do not write a session folder")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--list-targets", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        for i, name, api, sr in list_inputs():
            print(f"{i:3d}  {name}  [{api}, {sr} Hz]")
        return
    if args.list_targets:
        for key, name in vowelspace.list_targets():
            print(f"  {key:40s} {name}")
        return
    if args.open:
        run_review(args)
        return
    if args.reprocess:
        reprocess(args.reprocess, args)
        return
    if args.make_target:
        make_target(args.make_target, args.name or os.path.splitext(os.path.basename(args.make_target))[0], args)
        return
    if args.input and args.headless:
        folder = None if args.no_save else session_folder(args.input)
        p = run_offline(args.input, args, folder)
        if folder:
            p.save()
            print(f"saved in {folder}")
        print_summary(p)
        return
    run_gui(args)


if __name__ == "__main__":
    main()
