"""Words -> syllables with IPA, from the CMU Pronouncing Dictionary (CMUdict, ~126k words).

Why a dictionary instead of recognising the vowels from the audio: a vowel-space chart needs to know
which vowel the speaker MEANT (the word's dictionary vowel) so it can show how they actually said it.
A phoneme recogniser would label the vowel by how it sounded, which hides exactly the differences you
want to see. Sociophonetic vowel tools (e.g. FAVE) label vowels the same way: transcript + CMUdict.

Syllables are split with the maximal-onset rule (consonants between two vowels go to the second
syllable as long as English allows that cluster at the start of a word: "ex.tra" -> /ɛk.stɹə/).
"""
import re
from dataclasses import dataclass
from functools import lru_cache

ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
IPA_UNSTRESSED = {"AH": "ə", "ER": "ɚ"}

# The 12 vowels Hillenbrand et al. (1995) measured (heed hid hayed head had hod hawed hoed hood
# who'd hud heard): the ones the vowel chart plots and has built-in targets for.
CHART_VOWELS = ("IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "AH", "ER")
HILLENBRAND_CODE = {"iy": "IY", "ih": "IH", "ei": "EY", "eh": "EH", "ae": "AE", "ah": "AA", "aw": "AO",
                    "oa": "OW", "oo": "UH", "uw": "UW", "uh": "AH", "er": "ER"}
KEYWORD = {"IY": "heed", "IH": "hid", "EY": "hayed", "EH": "head", "AE": "had", "AA": "hod", "AO": "hawed",
           "OW": "hoed", "UH": "hood", "UW": "who'd", "AH": "hud", "ER": "heard",
           "AY": "hide", "AW": "how'd", "OY": "hoyed"}

_SINGLE = {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "P", "R", "S", "SH", "T", "TH",
           "V", "W", "Y", "Z", "ZH"}
_CLUSTERS = """P R|P L|P Y|B R|B L|B Y|T R|T W|D R|D W|K R|K L|K W|K Y|G R|G L|G W|F R|F L|F Y|V Y|
TH R|TH W|SH R|S P|S T|S K|S M|S N|S L|S W|S F|M Y|HH Y|S P R|S P L|S P Y|S T R|S K R|S K W|S K Y|S K L"""
LEGAL_ONSETS = {()} | {(c,) for c in _SINGLE} | {tuple(x.split()) for x in _CLUSTERS.replace("\n", "").split("|")}

# Words whose vowels are usually reduced in running speech (their dictionary vowel is not what is said).
FUNCTION_WORDS = set("""a an the and or but nor of to in on at for with by from as into onto upon than
is are was were be been being am do does did have has had having will would shall should can could may
might must it its it's itself this that these those i me my mine he him his she her hers we us our you
your yours they them their there here what which who whom whose when where why how if so not no
just then up out off over about some any all very too also such only own both each few more most other
yeah yes oh uh um like i'm you're we're they're he's she's that's there's i've you've i'll you'll
don't can't won't isn't aren't wasn't weren't doesn't didn't""".split())

# Exercise sounds, not words ("da da da", "ah", "ooh"). They are said differently from the same vowel in
# a word: in the first live sessions they sat a median 2.3 Bark from that speaker's real-word tokens of
# the same vowel, real words 0.9 ("ah" far more open, "ooh" far more back). Kept off the vowel chart.
VOCABLES = set("""ah aah ahh ahhh eh ehh ee eee oo ooh oooh ooo ohh ohhh hm hmm mm mmm
da ba la ma na pa ta ga ka ha fa sa za wa ya dah bah lah mah nah pah tah hah doo dee""".split())


@dataclass(frozen=True)
class DictSyllable:
    phones: tuple
    vowel: str         # ARPAbet vowel without the stress digit, e.g. "IH"
    stress: int        # 0 unstressed, 1 primary, 2 secondary
    ipa: str           # e.g. "ˈkɪdz"
    prev_phone: str    # phone before the vowel inside the word ("" = word start)
    next_phone: str    # phone after the vowel inside the word ("" = word end)

    @property
    def vowel_ipa(self):
        return IPA_UNSTRESSED.get(self.vowel, IPA[self.vowel]) if self.stress == 0 else IPA[self.vowel]


@dataclass(frozen=True)
class Pron:
    word: str
    phones: tuple
    syllables: tuple

    @property
    def ipa(self):
        return ".".join(s.ipa for s in self.syllables)


_DICT = None


def _cmu():
    global _DICT
    if _DICT is None:
        import cmudict
        _DICT = cmudict.dict()
    return _DICT


def preload():
    """Parse CMUdict now (~1 s) instead of on the first word, which would freeze the live window."""
    _cmu()


def normalize(word):
    w = word.lower().strip().replace("’", "'")
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w).strip("'")


def _is_vowel(p):
    return p[:-1] in ARPA_VOWELS and p[-1].isdigit()


def syllabify(phones):
    """ARPAbet phones -> list of phone tuples, one per syllable (maximal onset)."""
    phones = tuple(phones)
    vidx = [i for i, p in enumerate(phones) if _is_vowel(p)]
    if not vidx:
        return []
    cuts = [0]
    for a, b in zip(vidx, vidx[1:]):
        cluster = tuple(p for p in phones[a + 1: b])
        k = next(k for k in range(len(cluster) + 1) if cluster[k:] in LEGAL_ONSETS)
        cuts.append(a + 1 + k)
    cuts.append(len(phones))
    return [phones[cuts[i]: cuts[i + 1]] for i in range(len(vidx))]


def _syllable(phones, prev_phone, next_phone):
    v = next(p for p in phones if _is_vowel(p))
    base, stress = v[:-1], int(v[-1])
    mark = {1: "ˈ", 2: "ˌ"}.get(stress, "")
    body = "".join((IPA_UNSTRESSED.get(base, IPA[base]) if stress == 0 else IPA[base]) if p == v
                   else IPA.get(p[:-1] if p[-1].isdigit() else p, p.lower()) for p in phones)
    return DictSyllable(tuple(phones), base, stress, mark + body, prev_phone, next_phone)


def make_pron(word, phones):
    phones = tuple(phones)
    sylls = []
    pos = 0
    for sp in syllabify(phones):
        vi = pos + next(i for i, p in enumerate(sp) if _is_vowel(p))
        prev_p = phones[vi - 1] if vi > 0 else ""
        next_p = phones[vi + 1] if vi + 1 < len(phones) else ""
        sylls.append(_syllable(sp, prev_p, next_p))
        pos += len(sp)
    return Pron(word, phones, tuple(sylls))


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    """0..999999 -> list of English words (for when Whisper writes digits)."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    if n < 1000000:
        return number_words(n // 1000) + ["thousand"] + (number_words(n % 1000) if n % 1000 else [])
    return []


def _concat(word, parts):
    d = _cmu()
    if not parts or any(p not in d for p in parts):
        return ()
    return (make_pron(word, sum((tuple(d[p][0]) for p in parts), ())),)


@lru_cache(maxsize=1)
def _letter_names():
    """CMUdict's own entries for the letters ('b' -> B IY), stress marks removed."""
    d = _cmu()
    return {ch: [tuple(p.rstrip("012") for p in v) for v in d.get(ch, ())] for ch in "abcdefghijklmnopqrstuvwxyz"}


def spelled_out(word, phones):
    """True if `phones` is `word` read letter by letter. CMUdict lists many short words that way as well
    ("da" also as D.A. /di.eɪ/, "ba" as B.A., "am" as A.M., "us" as U.S.), which is not what someone
    saying the word means."""
    letters = [ch for ch in normalize(word) if ch.isalpha()]
    ph = tuple(p.rstrip("012") for p in phones)
    names = _letter_names()
    ends = {0}
    for ch in letters:
        ends = {e + len(n) for e in ends for n in names.get(ch, ()) if ph[e:e + len(n)] == n}
        if not ends:
            return False
    return len(ph) in ends


@lru_cache(maxsize=50000)
def lookup(word):
    """All dictionary pronunciations of `word` (tuple of Pron), () if it is not in the dictionary.
    Letter-by-letter variants are dropped when the word also has an ordinary pronunciation."""
    w = normalize(word)
    if not w:
        return ()
    d = _cmu()
    if w in d:
        prons = tuple(make_pron(w, p) for p in d[w])
        return tuple(p for p in prons if not spelled_out(w, p.phones)) or prons
    if w.isdigit() and len(w) <= 6:
        return _concat(w, number_words(int(w)))
    if "-" in w:
        return _concat(w, [x for x in w.split("-") if x])
    for suffix in ("'s", "s'", "s"):
        base = w[: -len(suffix)]
        if w.endswith(suffix) and base in d:
            ph = tuple(d[base][0])
            last = ph[-1].rstrip("012")
            add = ("IH0", "Z") if last in {"S", "Z", "SH", "ZH", "CH", "JH"} else \
                (("S",) if last in {"P", "T", "K", "F", "TH"} else ("Z",))
            return (make_pron(w, ph + add),)
    return ()


def guess_syllable_count(word):
    """Spelling-based guess for words that are not in the dictionary (only used to keep the
    alignment of the other words sensible; such words are never scored or plotted)."""
    w = re.sub(r"[^a-z]", "", normalize(word))
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def chart_eligibility(syl, word):
    """Should this syllable's vowel go on the vowel chart? Returns (ok, reason)."""
    if syl.vowel not in CHART_VOWELS:
        return False, "diphthong"
    if syl.stress == 0:
        return False, "unstressed"
    w = normalize(word)
    if w in FUNCTION_WORDS:
        return False, "function word"
    if w in VOCABLES:
        return False, "exercise sound"
    if len(w) == 1 and w.isalpha():
        return False, "single letter"   # Whisper writes "S" for a hissed /s/, "E" for "eee"
    if syl.vowel != "ER" and syl.next_phone in ("R", "L"):
        return False, "before r/l"
    if syl.vowel == "AE" and syl.next_phone in ("M", "N", "NG"):
        return False, "ae before nasal"
    return True, ""
