# Vocal analysis benchmark

Scores `get_audio_data` (main10, run exactly the way main11's headless logger runs it) against datasets with known
answers, and against Praat and pYIN on the same audio. It writes an HTML report of what works, what doesn't and
what to change.

## Run it

From the `vocalgenderemotionai` folder:

```
python benchmark/run_benchmark.py             # full run, about 15 minutes, downloads ~170 MB the first time
python benchmark/run_benchmark.py --quick     # small subset, a few minutes
python benchmark/run_benchmark.py --help      # all options (subset sizes, workers, skipping parts)
```

Each run writes `benchmark_reports/<date_time>/report.html` and `metrics.json`. The report compares itself with the
previous run, so after you change main10 you can rerun and see what got better or worse.

Results are cached per file in `benchmark_data/cache`. The cache key includes the source of main10 and the benchmark,
so only work affected by a code change is redone.

## What it tests

| Dataset | What it checks | Reference |
|---|---|---|
| PTDB-TUG (subset) | pitch, voiced/unvoiced, timing, noise robustness | laryngograph (throat sensor) pitch every 10 ms |
| Hillenbrand 1995 | formants F1-F4 and pitch on 12 vowels from men, women, children | hand-checked measurements at the vowel steady state |
| Synthetic vowels | pitch range, voice-quality features, onset/offset timing | exactly known generator settings |

"Candidate fix" in the report is a quick prototype of a suggested change (Praat pitch on the last 100 ms, and a
pitch-based formant ceiling). It is there to test whether a suggestion actually helps before you build it.

## Files

- `bench_datasets.py`: downloads and loads the datasets, and generates the synthetic vowels
- `bench_extractors.py`: your analysis (`run_yours`), Praat, pYIN and the candidate fixes
- `bench_metrics.py`: scoring (standard pitch-tracking and formant error measures)
- `run_benchmark.py`: runs everything in parallel and computes the results
- `bench_report.py`: verdicts, rule-based suggestions, charts and the HTML page

## Adding another dataset

1. In `bench_datasets.py`, write a loader that returns a list of dicts, each with at least `path` (audio file) and the
   reference values with their times.
2. In `run_benchmark.py`, add a `task_<name>` function that loads the audio and calls `ex.run_yours(y, sr, end_times)`
   at the times you need. Remember that `run_yours` takes window END times, like main11's logs.
3. Add an `analyse_<name>` function using `bench_metrics`, store its output in `results["<name>"]`, and add a section in
   `bench_report.write_report`.
