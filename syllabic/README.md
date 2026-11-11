# syllabic: live syllables, checked against Whisper, feeding a live vowel space

Run it with `python main14livesyllables.py` (live mic), or `python main14livesyllables.py --open <recording
or session folder>` to look through a recording or an earlier session. See the top of
[main14livesyllables.py](../main14livesyllables.py) for file mode, headless mode, `--reprocess` and making
targets. For running speech, pick the target `arctic_women_measured` or `arctic_men_measured`.

## What happens to the audio

```
mic (any rate) -> 16 kHz -> one frame every 10 ms (centred on its time stamp)
   |-> vowel-band level (300-2500 Hz), broadband level, periodicity, pitch      frontend.py
   |-> syllable detector: hill in the vowel band + voiced = a syllable nucleus   syllables.py
   |      reported ~0.1 s after the vowel's loudest point; start = the valley before it
   |-> F1-F4 at each nucleus (Praat Burg, median of +-20 ms, ceiling fitted     formants.py
   |      to your voice)
   '-> phrase cutter: 0.45 s of silence ends a phrase -> Whisper (own process)    whisper_words.py
          words + start/end times ~0.5-1 s after you pause
words -> CMUdict syllables + IPA (maximal onset)                                pronounce.py
dictionary syllables <-> detected nuclei, order-keeping DP alignment             align.py
   |-> live scores: syllable count per word, word start vs first-syllable start
   '-> each matched nucleus gets its intended vowel -> vowel-space token         vowelspace.py
```

Why each choice:
* **Vowel-band peaks** are the classic automatic syllable-nucleus method (de Jong & Wempe 2009, the
  Praat "syllable nuclei" script), made causal: a hill only counts once the level has dropped
  `min_dip_db` below it, which is what lets it run live.
* **Periodicity gate**: the same normalised-autocorrelation check that halved get_pitch's voicing
  errors in the earlier experiment. It stops /s/, breaths and noise from becoming "syllables".
* **Dictionary vowels, not recognised vowels**: the chart has to know which vowel you *meant* so it can
  show how you *said* it. A phoneme recogniser would label the vowel by how it sounded, hiding exactly
  that difference. FAVE-style sociophonetic tools label vowels the same way (transcript + CMUdict).
* **Only words whose syllable count came out right** give vowel tokens, and only stressed
  monophthongs outside function words, not before /r/ or /l/, and no /æ/ before nasals (the usual
  vowel-plot exclusions, since those contexts shift the vowel). Exercise sounds ("da", "ah", "ooh") and
  single letters ("S" for a hissed /s/) are left out too: see 2026-11-10 below.
* **CMUdict's letter-by-letter variants are dropped** when a word has an ordinary pronunciation ("da"
  is also listed as D.A. /di.eɪ/, "am" as A.M.), and the detected syllable count may only choose
  between variants with the same stressed vowels ("ev(e)ry", "fam(i)ly").
* **One formant ceiling per speaker** (Escudero et al. 2009): the candidate that makes tokens of the same
  vowel most alike, re-estimated as you talk and moved one step at a time. See below.
* **Targets measured the same way as you**: every built-in `_measured` target runs its recordings
  through this exact pipeline, so a gap between you and the target is not a difference between two
  measuring methods.

## What the first live sessions showed, and what changed (2026-11-09)

From `audio_logs/syllables_live_20261105_*` and `..._20261109_*`:

| Problem in the logs | Cause | Fix |
|---|---|---|
| Whisper "stopped": 41 and 216 phrases never answered, from minute 7 and 47 | In a thread it ran 3-10x slower than alone (the window's drawing), and held vowels ("eeee", "OHHH") made it retry for 2-4 min per phrase | Own process; temperature 0 only; token budget from the syllable count; at most 2 phrases waiting; watchdog restart. Same audio replayed: median 0.9 s, worst 4.9 s (was 142 s) |
| Words split across phrases ("...at one end. P") | Forced cut every 12 s at the quietest frame | Phrases up to 25 s, cut at the start of a syllable |
| "Word starts after a pause" only 16-27% | Whisper pins each phrase's first word to the clip start (median exactly -250 ms = the padding) | That word is no longer scored |
| A session's files only existed once the window closed (the 11-09 one looked broken while it was still recording: no CSVs, header saying "no audio") | Saving happened only on a clean close, so a crash or a killed terminal would lose the whole session | Autosave every 2 min, header refreshed every 30 s, Ctrl+C closes cleanly; `--reprocess` rebuilds a session from its audio |
| Clipped input (up to 0.3% of samples at full scale) | Gain too high | Warning in the window |
| Vowel space "clustered" | Mostly real: running speech has fronted back vowels. The Hillenbrand target is careful isolated words. Also the per-vowel "auto" ceiling picked 4500-4750 Hz for ~20% of tokens | Running-speech targets (ARCTIC); speaker-fitted ceiling |

The window size of the formant analysis was **not** the problem: 12-35 ms windows all scored within
~1 point of each other on Hillenbrand, and your space barely changed with them (F2 spread 922-933 Hz).

## A twisted vowel chart six minutes into a session (2026-11-10)

A screenshot of `syllables_live_20261110_084650` showed /i/ to the right of /u/ and /ɑ/ near the top.
Replaying the saved tokens through the chart's averaging reproduced the screenshot to within ~50 Hz at
t = 380 s, so the chart drew its data correctly. The data was wrong:

| Problem | Evidence | Fix |
|---|---|---|
| "ba"/"da" from a "ba da da" exercise were labelled /i/ and /eɪ/ | CMUdict lists them as B.A. /bi.eɪ/ and D.A. /di.eɪ/ too. When the detector split a vowel in two, the aligner picked that variant because its syllable count fit. 4 of /i/'s 5 tokens at 6 min were these, putting /i/ at F2 ≈ 1,520 Hz | Letter-by-letter variants dropped (183 CMUdict words); count may only pick a variant with the same stressed vowels |
| Exercise sounds and single letters on the chart | Across 4 sessions they sat a median 2.3 Bark ("da", "ah", "ooh") and 1.2 Bark (single letters) from the speaker's real-word tokens of the same vowel, vs 0.9 for real words | Left off the chart ("exercise sound", "single letter") |
| Averages of 1-4 tokens drawn as if settled | Replaying 23 vowels from 3 sessions: an average of 1-3 tokens was a median ~0.8 Bark from where that vowel settled later (1 in 10: 1.3-1.7 Bark); 5 tokens 0.56, 10 tokens 0.39. A windowed median was no better than the running mean | Under 5 tokens: hollow ring, grey label, not in the outline, spread or target links |
| Pitch in tooltips/CSVs an octave too high on 18% of that session's vowel tokens | Checked against Praat's pitch at the nuclei. The f0 rule took the shortest lag within 90% of the best peak, so the 2nd harmonic won whenever F1 boosted it | Praat's octave cost (0.01/octave): 5% that session, 1-1.5% on two earlier ones. Periodicity, which the detector uses, is unchanged |

With the new rules the same session at 6 minutes has /eɪ/ at F2 2,116 Hz (was 1,881) and /ɑ/ at F1 702
(was 498), and /i/, /æ/, /ɑ/, /ʊ/, /u/ show as not settled yet. Over the whole session, dropping the
"da" tokens moved /ɑ/ from F1 595 to 672. On ARCTIC only 104 of 40,144 words (0.26%: am, us, cod,
doctrine, fine, prayers) can get a different pronunciation, so the measured accuracy below stands.

## Going back, listening, reviewing (2026-11-10)

The window used to show only the last 8 seconds. Now it keeps the whole session:

* **Scroll back.** Drag the timeline or turn the mouse wheel over it. **Live ▶▶** (or End, or dragging back to
  the right edge) follows the newest audio again. While you are scrolled back, the transcript lists the
  phrases in view, and clicking a phrase's time jumps there; so does clicking the vocal-tract-length plot,
  which doubles as an overview of the session.
* **Point at it.** A dashed line follows the mouse. The row under the timeline gives the time, the word
  (IPA, syllable count ✓/✗), the syllable's F1-F4 and pitch, and whether it is on the chart (or why not:
  function word, exercise sound, outlier...). The same syllable is ringed on the vowel chart.
* **Listen.** Clicking a word plays it; **Play view** plays what the timeline shows (Esc stops). Live, this only
  works while paused: with VoiceMeeter or speakers the sound would otherwise go back into the microphone
  and into your own session. Audio older than what is kept in memory is read back from the session's
  `audio_16k.wav`.
* **Chart for one stretch.** "timeline view only" averages just the vowels in view (e.g. one reading
  passage); **Save mine as target** saves what the chart shows, so a later session can be compared with it.
* **Review a file or an old session.** `--open talk.wav` runs a recording through the same pipeline as fast as
  the computer allows, with Whisper on every phrase (it waits for Whisper rather than skipping), and saves
  it as a session. `--open <session folder>` re-runs a saved session with the words it already has
  (`reprocessed/words.csv` if there is one, else `words.csv`), no Whisper needed: on today's first 400 s it
  ran at 18x real time with the window drawing and gave the same 270 tokens, word for word and Hz for Hz,
  as the earlier full `--reprocess`. The **Open…** button starts a review in a new window.

Memory: the level trace costs 8 bytes per 10 ms (~3 MB per hour); words, syllables and tokens were already
kept for the whole session.

## Men → women position: what it can and can't say (2026-11-10)

From the 11-09 demo: "if you imagine the line from the male template to the female template, you should be
lying on that line". The chart can now draw, per vowel, the line from a men's average to a women's average
and place your settled vowels on them (`vowelspace.gender_position`, log F1-F3, so a uniform shift of all
formants, as a shorter vocal tract gives, moves you straight along the lines). It reports one position
(0% = men's average, 100% = women's) and how far your vowels sit to the side of the lines.

Checked on speakers the references were not built from (leave-one-out; `python
benchmark/eval_gender_position.py` re-runs all of this in ~3 min):

| Speakers | Position: median (middle 80%) | Off the lines, median |
|---|---|---|
| Hillenbrand men (45), each vs averages without them | 1% (-28% to 19%) | 5% |
| Hillenbrand women (48) | 93% (60% to 128%) | 6% |
| Hillenbrand boys (27) / girls (19) vs all adults | 136% / 169% | 7% / 9% |
| ARCTIC readers measured by this pipeline vs Hillenbrand (other corpus) | slt 102%, clb 45% (women); bdl 3%, rms -64% (men) | 9-17% |
| ARCTIC vs the other ARCTIC man + woman | slt 63%, clb 50%; bdl 38%, rms -41% | 6-12% |
| Your sessions 11-05 / 11-09 / 11-10 vs ARCTIC (running speech) | 46% / 45% / 54% | 8-9% |

* With clean, hand-checked formants it separates adults well: 98.9% of 93 on the right side of 50%, still
  96.7% with only 3 random vowels per speaker.
* In running speech, single speakers spread widely (one woman at 45-50%), so the number describes formants;
  it is not a verdict on how a voice is heard. Pitch, voice quality and intonation also carry the cue
  (Hillenbrand & Clark 2009, Atten. Percept. Psychophys. 71:1150).
* Per-vowel positions scatter about ±25 points even for typical speakers (SD 0.22-0.27 on Hillenbrand), so
  read the overall number, not single vowels.
* "Off the line" for real speakers is 5-12%; yours (8-9%) is in that range. Whether a bigger value sounds
  odd needs listening tests (see IDEAS.md).

## Measured accuracy

From `benchmark/eval_syllables.py` (results in `benchmark_reports/syllables_*`). Settings were tuned on
ARCTIC speakers clb + rms and the numbers below are for slt + bdl, which tuning never saw.

**ARCTIC (true word/syllable boundaries), speakers slt / bdl:**

| What | slt | bdl |
|---|---|---|
| Detected nuclei that sit inside a real vowel (precision) | 93% | 95% |
| Real syllables that were found (recall) | 80% | 81% |
| Syllable starts within 50 ms / 100 ms of the true start | 65% / 90% | 68% / 92% |
| First syllable after a pause: start within 100 ms | 94% | 98% |
| Whisper's own word starts within 100 ms: mid-phrase / first word after a pause | 82% / 75% | 84% / 84% |
| Live "word start" score vs the same detected starts judged against the truth (same words) | 65% vs 74% | 71% vs 77% |
| Live "syllable count right" score vs the truth | 77% vs 77% | 79% vs 74% |
| Vowel-chart points inside a real vowel / labelled with the right vowel | 98% / 97% | 98% / 94% |

(Re-measured 2026-11-09 on the final code, `benchmark_reports/syllables_20261109_230454`.)

Noise (white noise, both test speakers): nucleus F1 86.6 clean, 83.2 at 10 dB SNR, 72.2 at 5 dB.

**RAVDESS** (1,440 clips, 24 actors, every sentence has 7 syllables): exactly 7 in 44% of clips (neutral
81%, the emotions 34-52%), within one in 84%. Emotional speech over-counts (mean 7.5): long, shaky
vowels can dip twice.

**Hillenbrand 1995** (1,668 vowels, share within 10% of the hand-checked formants):

| Formant ceiling | F1 | F2 | F3 |
|---|---|---|---|
| 5000 Hz | 80% | 63% | 67% |
| 5500 Hz | 81% | 75% | 88% |
| from pitch | 81% | 76% | 90% |
| auto (per vowel) | 81% | 78% | 93% |
| **speaker-fitted (default)**, per group: men 5250, women 5750, girls 6000, boys 6500 Hz | **81%** | **77%** | **91%** |
| auto, measured at Hillenbrand's own steady-state time | 95-96% | 94-96% | 94% |

* The last row shows the formant method itself is fine. The live measuring point (the vowel's loudest
  moment, median 50 ms after their steady state) costs the rest, the same for you and the
  `_measured` targets.
* On isolated words, per-vowel "auto" and the speaker-fitted ceiling are about equal. In running speech
  the speaker-fitted one makes same-vowel tokens more consistent for all 5 voices tested (4 ARCTIC
  speakers and you), and it lands where Praat's advice says without being told who is talking:
  4750 Hz for both ARCTIC men, 5500-6000 Hz for the women, 5750 Hz for you.

**Running speech vs isolated words.** Measured exactly like you, ARCTIC readers have much more fronted
back vowels than Hillenbrand's isolated words: /u/ F2 1480-1490 Hz (men) and 1650-1920 Hz (women) in
sentences, vs about 1000-1100 Hz in "who'd". Your /u/ (~1610), /oʊ/ (~1330) and /i/ (~2250) sit between
the ARCTIC men and women: a normal running-speech space. Use the `arctic_*_measured` targets for talking
and reading; use the Hillenbrand ones only when you say isolated words.

## Limits worth knowing

* Whisper is not a streaming model: labels arrive after each pause. Syllables (timing, formants)
  are live; their IPA labels are ~0.5-1 s late. Held vowels ("eeee") are not sent to Whisper at all.
* Whisper's word times are not ground truth. Mid-phrase they are good to roughly +-60 ms; the first word
  of each phrase is unusable and left out. So the live "word start" score understates the detector
  (on ARCTIC it read 5-10 points lower than the truth on the same words).
* Syllables without a clear loudness dip (fast speech, vowel-vowel sequences such as "the eyes",
  reduced function words) get merged: about 1 in 5 dictionary syllables is missed.
* The higher the pitch, the fewer harmonics there are to outline the formants, and LPC (Praat's method
  too) pulls F1 toward the nearest harmonic (e.g. Shadle, Nam & Whalen 2016, JASA 139:713). In the
  first live sessions, chart tokens sat a median 0.9 Bark from their vowel's centre below 250 Hz
  (Praat's pitch), 1.2 at 250-350 Hz and 1.4 above 350 Hz (45% beyond 1.5 Bark). They are kept, since
  their labels are right and the averages absorb some noise, but the chart is less precise up there.
* ARCTIC's reference boundaries are themselves from automatic forced alignment (accurate to roughly
  10-30 ms), not hand labelling. The running-speech targets are 2 readers per group: good for shape and
  direction, too few to call population norms.
* For a target to train toward, record the target voice reading ordinary text and use `--make-target`:
  about 2.5 minutes of read sentences gave 5-18 tokens for each of the 12 vowels.
