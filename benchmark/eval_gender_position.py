"""python benchmark/eval_gender_position.py   (~3 min: measures 4 x 150 ARCTIC sentences)

Does 'position between the men's and women's averages' (syllabic.vowelspace.gender_position, the
"men → women" option of the live window) separate speakers it was not built from? And how far to the
side of those lines do real speakers sit (to read the 'off' number)?

1. Hillenbrand et al. 1995 published steady-state formants (139 speakers, 12 vowels, hand-checked): every
   speaker against men/women averages built WITHOUT them (leave-one-out); children against all adults.
2. CMU ARCTIC read sentences measured by this pipeline (running speech, true word times), each speaker
   against (a) Hillenbrand men/women measured by this pipeline (another corpus) and (b) the other ARCTIC
   man and woman only (leave-one-out).
3. Your own saved sessions (audio_logs/syllables_*), for orientation only (no ground truth).

Results on 2026-11-10 are in syllabic/README.md ("Men → women position").
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (PROJ, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from syllabic import pronounce, vowelspace  # noqa: E402
from syllabic.pronounce import HILLENBRAND_CODE  # noqa: E402

F = ("F1", "F2", "F3")
GROUPS = {"m": "men", "w": "women", "b": "boys", "g": "girls"}


def ref(df):
    return {v: {k: float(g[k].mean()) for k in F} for v, g in df.groupby("vowel")}


def means_of(df):
    return {v: {**{k: float(g[k].mean()) for k in F}, "n": len(g)} for v, g in df.groupby("vowel")}


def hillenbrand():
    import bench_datasets as ds
    rows = []
    for line in open(os.path.join(ds.hillenbrand_dir(), "vowdata.dat")):
        parts = line.split()
        if len(parts) != 16 or len(parts[0]) != 5 or parts[0][0] not in GROUPS:
            continue
        v = HILLENBRAND_CODE.get(parts[0][3:5])
        f = [float(x) for x in parts[3:7]]
        if v and min(f[:3]) > 0:
            rows.append({"group": GROUPS[parts[0][0]], "spk": parts[0][:3], "vowel": v, "F1": f[0], "F2": f[1],
                         "F3": f[2]})
    h = pd.DataFrame(rows)
    out = []
    for spk, g in h.groupby("spk"):
        men = ref(h[(h.group == "men") & (h.spk != spk)])
        women = ref(h[(h.group == "women") & (h.spk != spk)])
        r = vowelspace.gender_position(means_of(g), men, women, min_n=1)
        out.append({"group": g.group.iloc[0], "spk": spk, "pos": r["position"], "off": r["off"],
                    "pv_sd": float(np.std(list(r["per_vowel"].values())))})
    o = pd.DataFrame(out)
    print("1. Hillenbrand 1995 published formants, each speaker vs men/women averages without them")
    print(o.groupby("group").agg(n=("pos", "size"), pos_median=("pos", "median"),
                                 pos_p10=("pos", lambda s: s.quantile(.1)), pos_p90=("pos", lambda s: s.quantile(.9)),
                                 off_median=("off", "median"), off_p90=("off", lambda s: s.quantile(.9)),
                                 per_vowel_sd=("pv_sd", "median")).round(2).to_string())
    ad = o[o.group.isin(["men", "women"])]
    print(f"   men vs women split at 50%: {100 * np.mean((ad.pos > 0.5) == (ad.group == 'women')):.1f}% of "
          f"{len(ad)} adults on the right side")
    rng = np.random.default_rng(0)
    for k in (3, 5):   # fewer vowels, as early in a session: random subsets of k vowels per speaker
        hits = []
        for spk, g in h[h.group.isin(["men", "women"])].groupby("spk"):
            men = ref(h[(h.group == "men") & (h.spk != spk)])
            women = ref(h[(h.group == "women") & (h.spk != spk)])
            for _ in range(20):
                sub = g[g.vowel.isin(rng.choice(g.vowel.unique(), k, replace=False))]
                r = vowelspace.gender_position(means_of(sub), men, women, min_n=1)
                hits.append((r["position"] > 0.5) == (g.group.iloc[0] == "women"))
        print(f"   with only {k} random vowels per speaker: {100 * np.mean(hits):.1f}% right")


def arctic():
    import bench_datasets as ds
    from eval_syllables import TruthWords, build_stream
    from syllabic.frontend import SR, to_16k
    from syllabic.pipeline import Pipeline, PipelineConfig
    per = {}
    for spk in ("slt", "clb", "bdl", "rms"):
        items = ds.load_arctic(speakers=(spk,), max_per_speaker=150, seed=11)
        raw, words = build_stream(items)
        y = to_16k(raw, SR)
        p = Pipeline(SR, PipelineConfig(), whisper=TruthWords(words), store_seconds=len(y) / SR + 5)
        for i in range(0, len(y), 16000):
            p.feed16(y[i: i + 16000])
        p.finish()
        tgt = vowelspace.target_from_tokens(p.tokens, spk)
        per[spk] = {v: {**{k: m[k] for k in F}, "n": m["n"]} for v, m in tgt["vowels"].items()}
        print(f"   measured {spk}: {sum(m['n'] for m in per[spk].values())} chart tokens")
    hm = vowelspace.load_target("hillenbrand_men_measured")["vowels"]
    hw = vowelspace.load_target("hillenbrand_women_measured")["vowels"]
    print("\n2. ARCTIC read sentences (running speech, this pipeline), per speaker")
    sex = {"slt": "woman", "clb": "woman", "bdl": "man", "rms": "man"}
    for spk in per:
        a = vowelspace.gender_position(per[spk], hm, hw, min_n=5)
        om = [s for s in ("bdl", "rms") if s != spk][0]
        ow = [s for s in ("slt", "clb") if s != spk][0]
        b = vowelspace.gender_position(per[spk], per[om], per[ow], min_n=5)
        print(f"   {spk} ({sex[spk]}): vs Hillenbrand words {100 * a['position']:.0f}% (off {100 * a['off']:.0f}%, "
              f"{a['n_vowels']} vowels) | vs other ARCTIC man+woman ({om}, {ow}) {100 * b['position']:.0f}% "
              f"(off {100 * b['off']:.0f}%)")


def sessions():
    print("\n3. Your saved sessions (no ground truth; for orientation)")
    am = vowelspace.load_target("arctic_men_measured")["vowels"]
    aw = vowelspace.load_target("arctic_women_measured")["vowels"]
    hm = vowelspace.load_target("hillenbrand_men_measured")["vowels"]
    hw = vowelspace.load_target("hillenbrand_women_measured")["vowels"]
    logs = os.path.join(PROJ, "audio_logs")
    for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
        base = os.path.join(logs, name)
        f = os.path.join(base, "reprocessed", "tokens.csv")
        f = f if os.path.exists(f) else os.path.join(base, "tokens.csv")
        if not name.startswith("syllables_") or not os.path.exists(f):
            continue
        toks = [t for t in vowelspace.load_tokens(f) if t["chart"] and pronounce.normalize(t["word"]) not in
                pronounce.VOCABLES and len(pronounce.normalize(t["word"])) > 1]   # today's chart rules
        tg = vowelspace.target_from_tokens(toks, name)["vowels"]
        means = {v: {**{k: m[k] for k in F}, "n": m["n"]} for v, m in tg.items()}
        a = vowelspace.gender_position(means, am, aw, min_n=5)
        b = vowelspace.gender_position(means, hm, hw, min_n=5)
        if a["position"] is None or b["position"] is None:
            print(f"   {name}: too few tokens ({len(toks)})")
            continue
        print(f"   {name}: {len(toks)} tokens | vs ARCTIC (speech) {100 * a['position']:.0f}% off "
              f"{100 * a['off']:.0f}% | vs Hillenbrand (words) {100 * b['position']:.0f}% off {100 * b['off']:.0f}%")


if __name__ == "__main__":
    hillenbrand()
    arctic()
    sessions()
