# Assessment — EN↔ES SmolLM2-360M

**Start here:** [`REPORT.md`](REPORT.md) — full write-up organised by the three questions
(catastrophic forgetting, error attribution, metric quality), with figures and tables.
All translations were produced locally on CPU.

## Layout

| path | what |
| --- | --- |
| `REPORT.md` | the report (read this) |
| `figures/*.png` | forgetting, English-PPL, scaling, attribution, metric-scatter plots |
| `build_train_freq.py` | E0 — per-tier word-frequency tables (data-gap evidence) |
| `run_teacher_forced.py` | E1a — teacher-forced word recall across snapshots |
| `run_english_ppl.py` | E2 — held-out English perplexity per checkpoint |
| `run_generation.py` | E1b — greedy generation (`--selection best/trail/indomain`) |
| `error_attribution.py` | E3 — classify best-ckpt misses into data / forgetting / decoding / capability |
| `metric_quality.py` | E4 — BLEU vs chrF vs word-recall, single-ref tax, domain gap, asymmetry |
| `make_report.py` | E5 — render figures + assemble `REPORT.md` |
| `common.py` | shared helpers (tiers, prompt format, recall, checkpoint selection) |
| `data/*.json` | machine-readable results for every experiment |
| `data/attribution/*.csv` | every missed reference word with its assigned cause |

## Reproduce

See the "Reproduce" section at the bottom of `REPORT.md`. All commands use `uv run python`.
The two `data/_*.ps1` drivers run the whole pipeline end-to-end (phase 1 then phase 2).
