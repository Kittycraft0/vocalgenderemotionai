# Ideas and to-dos

A working list, last updated 2026-11-10. Sources:
* **notes**: your notes in `userprompts.txt` (11-05, 11-09, 11-10)
* **demo**: the 11-09 voice-chat demo transcript in `somestuffs.txt`
* **found**: things that came up while building and testing

Effort: **S** = an afternoon, **M** = a few days, **L** = a project.

## Done on 2026-11-10

| Idea (source) | What exists now | Where |
|---|---|---|
| See the whole session and go back to analyse it (notes) | The timeline keeps the whole session (about 3 MB per hour). Drag or scroll it to go back. **Live ▶▶** or End follows the newest audio again, and so does dragging back to the right edge. | `syllabic/gui.py` |
| The window should have memory when paused, like main11's file mode (notes) | Pause, then scroll anywhere in the session. The transcript then lists the phrases in view, and clicking a phrase's time jumps there. | same |
| main11's red hover line (notes) | A dashed line follows the mouse. The row under the timeline shows the time, the word (with its IPA and ✓/✗), the syllable's F1–F4 and pitch, and why it is or isn't on the chart. The syllable is also ringed on the vowel chart. Clicking a word plays it, but only while paused or reviewing, so playback can't leak into the mic. **Play view** plays the visible stretch. | same |
| Open an offline file and map it through Whisper and the graphs (notes) | `--open talk.wav` processes the file as fast as possible, with every phrase transcribed and the timeline filling as it goes, and saves it as a session. `--open <session folder>` re-opens a saved session with its saved words, no Whisper needed, at about 15–30x real time. There's also an **Open…** button. | `main14livesyllables.py` |
| Load audio "like main11's headless mode" (notes) | Already existed: `--input file --headless` and `--reprocess`. Now there's `--open` too. | same |
| The chart looked "backwards" / "inverted"; F3 and F4 looked inversely related (demo) | F3/F4 and F2/F3 now use plain axes, where both grow up and right. Before, a voice with higher F3 *and* F4 moved up-left. F1/F2 keeps the IPA-chart layout, with the axis labels saying "higher to the LEFT / DOWNWARD". Untick **vowel-chart layout** for plain axes. | `gui.py` |
| An extreme "eeee" ran off the chart: "F2 is getting so high that it's losing the range" (demo) | The chart grows when the live dot stays outside it for about 0.1 s. Implausible values, like a tracking error at F2 = 9000, don't count. | `gui.py` |
| Gender as distance to the male and female templates; being "on the line between them" sounds natural and being off it sounds weird (demo) | **men→women** draws a per-vowel line from the men's average to the women's, gives your position along them (0% men, 100% women) and how far you sit to the side. It was validated on held-out speakers (see below). | `vowelspace.gender_position`, `gui.py` |
| Save the vowel space (demo) | **Save mine as target** saves the whole session, or only the timeline view (for example your best reading), as a target you can compare a later session against. | `gui.py` |
| Faster re-runs (found) | The formant median used `np.nanmedian`, which on tiny arrays was ~14x slower than needed. The pipeline went from 26x to 33x real time with identical results (20,000 random checks). | `syllabic/formants.py` |
| If Whisper's process crashed while loading, the app waited forever (found while testing `--open`) | Now it reports the error, and the phrases that were waiting come back as errors instead of "transcribing…" forever. | `syllabic/whisper_words.py` |

**What the men→women number can and can't say.** Evidence is in `syllabic/README.md`; `python benchmark/eval_gender_position.py` re-runs it in about 3 minutes.
* On Hillenbrand's 139 speakers, each compared with averages built without them, 99% of adults landed on the correct side of 50%. The middle 80% of men spanned −28% to 19% and of women 60% to 128%. Children sat beyond 100%.
* Typical speakers sit 5–12% to the side of the lines. Your sessions sit at 8–9%, inside that range.
* In running speech measured by this pipeline, one ARCTIC woman sat at 45–50%, so single speakers spread widely.
* Per-vowel values scatter about ±25 points even for typical speakers.
* So it describes your formants. It is not a verdict on how you are heard: pitch and voice quality matter as well (Hillenbrand & Clark 2009, *Atten. Percept. Psychophys.* 71:1150).

## Next: small, clear value

1. **Make a target from a recording, in the window** (demo: "upload an audio file and train your voice to it"). **S.** `--make-target` already exists on the command line. Add a button that runs it in a separate process and adds the result to the target list.
2. **Targets anywhere along men→women** (demo: "go from a masculine one to a feminine one and then further"). **S.** Per vowel, the target's log formant = log men + λ·(log women − log men), so λ = 0.7 or 1.2 can be a target. Only show it with the spread caveat above.
3. **Drill mode for held vowels** (demo, and your sessions are full of "eeee", "ba da da", "ooh"). **S–M.** Held vowels give no words, so Whisper can't label them and they never reach the chart. Add a "drilling: /i/" selector that labels the exercise phrases with that vowel and plots them as a separate layer, never mixed into your speech averages.
4. **Diphthongs as arrows** (demo: the "I" sound /aɪ/ didn't show up). **M.** Measure /aɪ aʊ ɔɪ eɪ oʊ/ at 20% and 80% of the vowel and draw an arrow, which is how sociophonetic plots show them.
5. **Point in the transcript ↔ point in the timeline.** **S.** Hovering a transcript word could put the same line on the timeline.
6. **Faster `--open` for long sessions.** **M.** A 2-hour session takes 5–10 min because the window draws while it processes. Running the pipeline in a worker process (like Whisper) would get close to the 33x headless speed.
7. **main11's live mode still forgets after 10 s.** **M.** Its file mode has the red playhead and whole-file view, but live mode rolls fixed-size arrays. Either give main11 growing histories plus a pause, or move its voice-weight and spectral-slope measures into main14 (see 12) and keep main11 for file analysis.

## Training features (the voice-training goal)

8. **Darts** (demo). **S–M.** Say a vowel cold (no sliding into it) and score the distance from the target ring: one throw per prompt, a score per round, a history. It trains landing on the target without the live dot. A party version (friends taking turns) needs nothing new.
9. **Background monitor with a sound cue** (demo: "if you can tell within a second whether someone is conforming… it could have it on in the background and correct them"). **M.** A quiet tone when, for example, the running VTL or the men→women position drifts past a threshold for N seconds. Pitch already gets a cue in many apps; resonance rarely does. Needs a slow, reliable measure: a few seconds of averaging, never single frames.
10. **Voice as a controller** (demo: F1 up/down and F2 left/right steering a car or character). **M.** It trains the dimensions separately ("decouple the different sounds in your voice"). Use the live cursor (40 ms updates) with smoothing; a simple Pygame or pyqtgraph scene is enough to try it.
11. **Match the spread, not only the averages** (demo: perfectly averaged vowels would "sound like early text-to-speech"). **M.** Show your per-vowel spread against the target's (targets already store `sd`), and add prosody: f0 range and syllable timing (see 15).
12. **Weight and resonance together** (demo: "a soft voice with the masculine vowel space"). **M.** A 2-D view with voice weight or spectral slope (main11's measures) on one axis and VTL or men→women on the other, so you can see the two are independent and practise them separately.
13. **F3/F4 steadiness, and nasality as its own axis** (demo: "F1/F2 is great but F3/F4 is not stable at all"; "nasality is a separate axis"). **M.** First check how much F3/F4 jitter is measurement (bandwidths, ceiling) and how much is you, by measuring sustained vowels from a reference corpus. Then add a nasality estimate such as A1−P0 (Chen 1997, *JASA* 102:2360) as its own read-out.
14. **"What would I sound like?"** (11-05 note from a friend; earlier offer). **L.** Resynthesise your own recording with your formants moved to the target's, using LPC or WORLD-style source–filter resynthesis, so you hear the goal in your own voice. A first version only needs per-vowel formant shifts. An estimate of progress after X hours would need longitudinal data, which the session logs are starting to collect (see 19).

## Analysis tool and research

15. **Syllable-duration and stress norms from real data** (notes, 11-10). **L.** The question is how long each syllable of each word is on average, and whether real speech confirms the stress rules the books state.
    1. Align LibriSpeech (~1,000 h, ~2,400 speakers; Panayotov et al. 2015) with the Montreal Forced Aligner (McAuliffe et al. 2017). Published LibriSpeech alignments exist, so look before aligning it yourself.
    2. For every word, store each syllable's duration divided by the word's (or the speaker's speaking rate), plus intensity and f0 at its nucleus.
    3. For words with ≥ 20 tokens, the medians of those proportions are the "how people actually say it" dictionary.
    4. Compare with CMUdict stress. Classic work says stressed syllables are longer, louder and higher (Fry 1955, *JASA* 27:765; Klatt 1976, *JASA* 59:1208); count where real data disagrees.
    5. In the app, score a new speaker's syllable proportions against those norms, which helps with accent work.

    Starting small: this pipeline already gives syllable timing plus CMUdict stress, and ARCTIC has true word times, so steps 2–4 can be tried on ARCTIC (1,132 sentences per speaker) in a day.
16. **Remove vocal-tract size to compare accents** (notes: "normalize by vocal tract length to analyze other factors such as dialect"). **M.** Lobanov (1971, *JASA* 49:606) and Nearey's log-mean (1978) normalisation removed the most speaker-size variation while keeping dialect differences in Adank, Smits & van Hout (2004, *JASA* 116:3099). In the app this would be a "compare shapes" view: the target rescaled to your size, so the remaining gaps are vowel quality (accent) rather than size. `gender_position` already works in log space, so most of the maths is there.
17. **Test the "off the line sounds weird" idea with listeners** (demo). **M.** The measurement is in place. What's missing is perception data: record short phrases at different positions and off-line distances, then have friends rate "natural/odd" and "masculine/feminine" blind. You mentioned friends offered to try it. Even 10 listeners × 40 clips would show whether off-line distance predicts "odd".
18. **More validation of men→women in running speech.** **S–M.** RAVDESS has 24 actors (12 women, 12 men) and is already on disk. Run them through `--open` and check the split, as was done for Hillenbrand. More ARCTIC voices (jmk, awb, ksp…) would show accent effects.
19. **Progress across sessions.** **M.** Per session, plot the per-vowel distance to the target, the men→women position and VTL against date. That's the start of "after X hours of practice" (14). It needs `--reprocess` on old sessions so they're all measured by the same code.
20. **Formant tracking at high pitch and extreme vowels** (found; demo). **M.** Tokens said above ~350 Hz pitch sit further from their vowel's centre (1.4 vs 0.9 Bark), because LPC pulls F1 toward a harmonic. Options: pitch-adaptive analysis windows, or flagging tokens above a pitch threshold. Also check that extreme /i/ (F2 near 3 kHz) isn't merged with F3 at the speaker ceiling.

## Product, later (demo: "friends said they'd pay $7/month")

21. **Packaging.** **L.** One app with live practice, review, targets and progress. It should run on a normal laptop, which means CPU Whisper (`base.en`) or a hosted transcription option.
22. **Things to get right first:**
    * Recordings are sensitive personal data: store them locally by default and make deletion easy.
    * Licences of everything shipped: Whisper is MIT, CMUdict is BSD-style; check Hillenbrand and ARCTIC before shipping their numbers as targets.
    * No promises the evidence doesn't support: for example, the men→women number is not a "passing" score.
    * A short "how to read the charts" page, because the inverted vowel-chart axes confused people at the demo.
