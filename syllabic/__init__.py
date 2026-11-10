"""Live syllable detection, checked against Whisper's word timestamps, feeding a live vowel-space chart.

Modules
  frontend      mic audio -> 16 kHz -> 10 ms frames (vowel-band level, broadband level, periodicity, f0)
  syllables     live syllable nucleus + start detector (causal de Jong & Wempe style peak picking)
  formants      F1-F4 at each nucleus (Praat Burg via Parselmouth), apparent vocal-tract length
  pronounce     CMUdict words -> syllables with IPA and stress
  whisper_words Whisper word timestamps on a background thread
  align         detected syllables <-> Whisper words + dictionary syllables, live accuracy scores
  vowelspace    running per-vowel averages, targets (Hillenbrand 1995 or your own recording)
  pipeline      everything above glued together (used by the GUI, file mode and the evaluation)
  gui           the live window (pyqtgraph)
Entry point: main14livesyllables.py.  Evaluation: benchmark/eval_syllables.py
"""
