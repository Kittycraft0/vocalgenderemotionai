"""Lining the detected syllables up with Whisper's words, and scoring the detector against them.

Whisper gives each word a start and end time; CMUdict gives each word its syllables. An order-keeping
dynamic-programming alignment (like DTW) then pairs dictionary syllables with detected nuclei:
  * pairing costs little when the nucleus lies inside its word's span and near where that syllable
    "should" be (syllables spread evenly over the word), and grows as it moves outside the span;
  * a dictionary syllable left without a nucleus = a MISSED syllable (cost_miss);
  * a nucleus left without a dictionary syllable = an EXTRA syllable (cost_extra).
Because order is kept, a word boundary that Whisper puts 100 ms too late does not steal the next
word's vowel, which simple "which span is the nucleus in" counting would.

Live scores (per word, on dictionary words only):
  * syllable count right: nuclei matched + extras inside the word == dictionary syllables
  * word start hit: the first syllable was found and starts within onset_tol_s of Whisper's word start
Whisper's own word times are off by tens of ms (more at the start of a phrase), so these numbers
measure the detector and Whisper together; eval_syllables.py measures both against hand-made-quality
alignments to show how much is whose.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import pronounce


@dataclass
class AlignParams:
    sigma_out_s: float = 0.10     # cost scale for a nucleus outside its word's span
    max_out_s: float = 0.25       # never pair a nucleus this far outside its word
    lam_pos: float = 0.25         # weight of "where in the word" (only matters for 2+ syllable words)
    cost_miss: float = 1.0
    cost_extra: float = 1.0
    onset_tol_s: float = 0.10
    pause_s: float = 0.15         # a gap this long before a word = the word starts after a pause


@dataclass
class WordResult:
    gi: int                       # word number in the session
    word: str
    start: float
    end: float
    prob: float
    pron: object                  # pronounce.Pron, or None if not in the dictionary
    n_exp: int                    # dictionary syllables
    nuclei: list                  # per dictionary syllable: the matched Syllable or None
    extra: list = field(default_factory=list)   # nuclei inside the word that matched nothing
    after_pause: bool = False
    utt: int = -1
    phrase_initial: bool = False  # first word Whisper returned for the phrase: its start time is pinned
                                  # to the start of the audio clip, so it is not used for the start score

    @property
    def scored(self):
        return self.pron is not None and self.n_exp > 0

    @property
    def n_matched(self):
        return sum(n is not None for n in self.nuclei)

    @property
    def n_det(self):
        return self.n_matched + len(self.extra)

    @property
    def count_ok(self):
        return self.n_det == self.n_exp

    @property
    def onset_err(self):
        """(start of the matched first syllable) - (Whisper's word start), seconds."""
        if not self.nuclei or self.nuclei[0] is None:
            return None
        return self.nuclei[0].t_onset - self.start


def _same_shape(a, b):
    """Two variants of a word differ only by syllables that come and go ('ev(e)ry', 'fam(i)ly'): the
    stressed vowels of one are an in-order subset of the other's."""
    x = [s.vowel for s in a.syllables if s.stress > 0]
    y = [s.vowel for s in b.syllables if s.stress > 0]
    if len(x) > len(y):
        x, y = y, x
    it = iter(y)
    return all(v in it for v in x)


def _choose_pron(prons, n_inside):
    """Pick the dictionary variant whose syllable count matches what was detected ('every' has 2- and
    3-syllable variants); otherwise the closest, preferring the dictionary's first (most common).
    Only variants with the same stressed vowels as the first one can win on count, or one vowel split
    in two by the detector could relabel a word with a different variant's vowels."""
    ok = [p for p in prons if _same_shape(prons[0], p)]
    return min(ok, key=lambda p: (abs(len(p.syllables) - n_inside), prons.index(p)))


def align(words, nuclei, params=None, gi0=0, utt=-1):
    """words: [{'word','start','end','prob'}] with absolute times; nuclei: [Syllable] in time order.
    Returns [WordResult] and marks every nucleus (status, label, vowel, word_i, syl_i)."""
    p = params or AlignParams()
    times = np.array([n.t_peak for n in nuclei], dtype=float)
    results, ref = [], []   # ref: (word index, syllable index, expected time, start, end)
    prev_end = None
    for wi, w in enumerate(words):
        prons = pronounce.lookup(w["word"])
        inside = int(np.sum((times >= w["start"]) & (times <= w["end"]))) if len(times) else 0
        pron = _choose_pron(prons, inside) if prons else None
        n = len(pron.syllables) if pron else pronounce.guess_syllable_count(w["word"])
        gap = None if prev_end is None else w["start"] - prev_end
        results.append(WordResult(gi=gi0 + wi, word=w["word"], start=w["start"], end=w["end"],
                                  prob=w.get("prob", 1.0), pron=pron, n_exp=n if pron else 0,
                                  nuclei=[None] * n, after_pause=gap is None or gap >= p.pause_s, utt=utt,
                                  phrase_initial=wi == 0))
        dur = max(w["end"] - w["start"], 0.02)
        for j in range(n):
            ref.append((wi, j, w["start"] + dur * (j + 0.5) / n, w["start"], w["end"], dur / n))
        prev_end = w["end"]

    K, M = len(ref), len(nuclei)
    if K and M:
        st = np.array([r[3] for r in ref])[:, None]
        en = np.array([r[4] for r in ref])[:, None]
        ex = np.array([r[2] for r in ref])[:, None]
        slot = np.maximum(np.array([r[5] for r in ref]), 0.05)[:, None]
        out = np.maximum(0.0, np.maximum(st - times[None, :], times[None, :] - en))
        cost = (out / p.sigma_out_s) ** 2 + p.lam_pos * ((times[None, :] - ex) / slot) ** 2
        cost[out > p.max_out_s] = np.inf
        D = np.full((K + 1, M + 1), np.inf)
        B = np.zeros((K + 1, M + 1), np.int8)
        D[0, :] = np.arange(M + 1) * p.cost_extra
        D[:, 0] = np.arange(K + 1) * p.cost_miss
        B[0, 1:], B[1:, 0] = 2, 1
        for k in range(1, K + 1):
            for m in range(1, M + 1):
                c0 = D[k - 1, m - 1] + cost[k - 1, m - 1]
                c1 = D[k - 1, m] + p.cost_miss
                c2 = D[k, m - 1] + p.cost_extra
                if c0 <= c1 and c0 <= c2:
                    D[k, m], B[k, m] = c0, 0
                elif c1 <= c2:
                    D[k, m], B[k, m] = c1, 1
                else:
                    D[k, m], B[k, m] = c2, 2
        k, m, pairs, extras = K, M, [], []
        while k > 0 or m > 0:
            b = B[k, m]
            if b == 0:
                pairs.append((k - 1, m - 1))
                k, m = k - 1, m - 1
            elif b == 1:
                k -= 1
            else:
                extras.append(m - 1)
                m -= 1
    else:
        pairs, extras = [], list(range(M))

    for n in nuclei:
        n.status, n.word_i, n.syl_i, n.label, n.vowel = "pending", None, None, None, None
    for k, m in pairs:
        wi, j = ref[k][0], ref[k][1]
        r, n = results[wi], nuclei[m]
        r.nuclei[j] = n
        n.word_i, n.syl_i = r.gi, j
        if r.pron is not None:
            s = r.pron.syllables[j]
            n.status, n.label, n.vowel = "matched", s.ipa, s.vowel
        else:
            n.status = "unscored"
    for m in extras:
        n = nuclei[m]
        if not results:
            n.status = "extra"
            continue
        dist = [max(0.0, w.start - n.t_peak, n.t_peak - w.end) for w in results]
        r = results[int(np.argmin(dist))]
        r.extra.append(n)
        n.word_i = r.gi
        n.status = "extra" if r.scored else "unscored"
    return results


class LiveScore:
    """Rolling and whole-session scores over the scored (dictionary) words."""

    def __init__(self, window=100, tol=0.10):
        self.tol = tol
        self.recent = deque(maxlen=window)
        self.all = []

    def add(self, results):
        for r in results:
            if r.scored:
                self.recent.append(r)
                self.all.append(r)

    def summary(self, which="recent"):
        ws = list(self.recent) if which == "recent" else self.all
        if not ws:
            return {"words": 0}
        exp = sum(w.n_exp for w in ws)
        det = sum(w.n_det for w in ws)
        mat = sum(w.n_matched for w in ws)
        inner = [w for w in ws if not w.phrase_initial]   # Whisper's phrase-initial starts are not usable
        errs = np.array([w.onset_err for w in inner if w.onset_err is not None])
        hits = [w.onset_err is not None and abs(w.onset_err) <= self.tol for w in inner]
        pause = [h for h, w in zip(hits, inner) if w.after_pause]
        conn = [h for h, w in zip(hits, inner) if not w.after_pause]
        return {
            "words": len(ws),
            "count_exact": float(100.0 * np.mean([w.count_ok for w in ws])),
            "syll_expected": exp, "syll_detected": det, "syll_matched": mat,
            "recall": 100.0 * mat / exp if exp else 0.0,
            "precision": 100.0 * mat / det if det else 0.0,
            "onset_words": len(inner),
            "onset_hit": float(100.0 * np.mean(hits)) if hits else None,
            "onset_hit_after_pause": float(100.0 * np.mean(pause)) if pause else None,
            "onset_hit_connected": float(100.0 * np.mean(conn)) if conn else None,
            "onset_median_ms": 1000.0 * float(np.median(errs)) if len(errs) else None,
        }
