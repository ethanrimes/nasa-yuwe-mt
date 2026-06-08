# spanish-nasa-mt-experiment (`snmt`)

NLLB-200 fine-tuning on the **real** Spanish↔Nasa-Yuwe parallel corpus
(`nasa_yuwe_parallel_dataset.jsonl`, ~24k pairs). These runs are packed onto the
**same H100** as the En↔Zh ablation matrix to fill spare GPU capacity — one VM
session, one shared auto-teardown.

## Why this exists
The En↔Zh experiment is a *proxy* ablation. The H100 has spare VRAM during it, so
we opportunistically fine-tune NLLB (600M / 1.3B / 3.3B) on the actual target-language
data. The on-VM runner (`english-chinese-mt-experiment/scripts/run_matrix_on_vm.py`)
packs these runs into the same VRAM-bounded waves as the En↔Zh runs and mirrors their
outputs to Blob under `experiments/es-nasa/`. If this package is absent the En↔Zh
matrix runs exactly as before — the integration is robust-if-absent.

## Nasa-Yuwe is not an NLLB language
Nasa-Yuwe (Páez, ISO 639-3 `pbb`) isn't one of NLLB's 200 languages. `snmt.lang`:
1. adds `pbb_Latn` as a special token,
2. resizes the model's embeddings, and
3. warm-starts the new row from `spa_Latn`'s embedding.

Sequences are framed manually (`[lang_id] + content + [eos]`) for stability across
`transformers` versions, paired with `DataCollatorForSeq2Seq` to reproduce NLLB's
"decoder emits the target language code first" scheme. We train bidirectionally
(spa→pbb and pbb→spa).

## Layout
- `src/snmt/data.py` — deterministic, content-hashed train/dev/test split + dedup.
- `src/snmt/lang.py` — `pbb_Latn` token handling + manual NLLB framing (pure, tested).
- `src/snmt/train.py` — `Seq2SeqTrainer` fine-tune (loss-only) + post-hoc `evaluate_bleu`.
- `src/snmt/runs.py` — turns `configs/nllb.yaml` into `schedule.Run`s + launch argv.
- `configs/` — `nllb.yaml` (run list) + `model_{600m,1.3b,3.3b}.yaml` (per-size hparams).
- `scripts/prepare_data.py` — build splits (source: explicit → local → Blob).
- `scripts/train_nllb.py` — launch one fine-tune.

## Install
```bash
pip install -e .          # light deps only (pyarrow / pyyaml / tqdm) — for planning/tests
pip install -e '.[gpu]'   # adds torch / transformers / datasets / sacrebleu (GPU/VM)
```

## Local validation (no GPU spend)
```bash
pytest tests                                   # pure-logic tests, all green
python scripts/prepare_data.py --source <jsonl> --out-dir /tmp/splits
```
From the En↔Zh experiment, `python scripts/11_run_ablations.py plan` shows the
combined estimate (En↔Zh + the 3 NLLB co-runs). Pass `--no-nllb` to exclude them.

## Run (on the H100, billed)
Driven end-to-end by the En↔Zh orchestrator — there is no separate provisioning:
```bash
cd ../english-chinese-mt-experiment
python scripts/11_run_ablations.py run            # packs En↔Zh + NLLB, single teardown
python scripts/11_run_ablations.py run --no-nllb  # En↔Zh only
```
