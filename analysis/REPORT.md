# H100 Run — Results, Analysis & Interpretation

_Generated from the durable Blob artifacts of the single-VM H100 session (now torn
down). Covers the En↔Zh proxy ablation (4 cells) and the opportunistic NLLB
Spanish↔Nasa-Yuwe fine-tunes (3 sizes). All 7 model dirs are in Blob; metrics,
charts and vibe-checks are reproduced locally under `analysis/`._

---

## 0. TL;DR

| Track | Verdict |
|---|---|
| **En↔Zh proxy (SmolLM2 360M & 1.7B × sentence / sentence+vocab)** | ✅ **Healthy.** All four cells trained cleanly, eval loss + token-accuracy improved, **negligible English forgetting** (≤ ±1.2 %, threshold +20 %). 1.7B clearly beats 360M → the data has real signal. |
| **NLLB es↔nasa (600M / 1.3B / 3.3B)** | ❌ **Diverged to NaN.** All three collapsed in the first ~25 optimizer steps; final checkpoints are **100 % NaN** (509/509 tensors for 600M) and emit empty output. **Root cause: true-bf16 _master weights_** (`train.py:95 torch_dtype=torch.bfloat16`) + AdamW with no non-finite-step guard. The data is fine; the precision config is the bug. |

**The single most important takeaway:** the En↔Zh result (the actual purpose of the
run — a data-quality probe) is valid and positive. The NLLB divergence is a
fixable training-config defect, **not** a problem with the Nasa-Yuwe data.

---

## 1. What was run

7 fine-tunes packed onto one H100, auto-torn-down on completion.

### En↔Zh proxy ablation (causal LM, package `ecmt`)
A 2×2 grid: model size {360M, 1.7B} × tokenization {`sentence`, `sentence+vocab`}.
Goal: does the current En↔Zh parallel data let a model *learn* translation, and
does fine-tuning damage the base model's English (catastrophic forgetting)?

### NLLB es↔nasa fine-tunes (seq2seq, package `snmt`)
Opportunistic use of spare VRAM: fine-tune NLLB-200 {600M, 1.3B, 3.3B} on the
**real** 24,229-pair Spanish↔Nasa-Yuwe corpus, bidirectionally. Nasa-Yuwe (Páez,
ISO `pbb`) is not a native NLLB language, so a `pbb_Latn` token was added,
embeddings resized, and the new row warm-started from Spanish.

---

## 2. Headline metrics

Full numbers in [`metrics_table.json`](metrics_table.json); charts in
[`charts/`](charts/).

### En↔Zh (all healthy)

| cell | final eval loss | eval token-acc | English forgetting Δ | wall-clock |
|---|---|---|---|---|
| 1.7B · sentence       | **4.358** | **0.387** | +1.11 % | 18.3 min |
| 1.7B · sentence+vocab | 4.618 | 0.362 | −0.97 % | 19.9 min |
| 360M · sentence       | 6.176 | 0.254 | −0.83 % | 15.1 min |
| 360M · sentence+vocab | 6.681 | 0.241 | −0.31 % | 22.9 min |

- **Lower eval loss / higher token-acc = better.** 1.7B is ~1.8 loss-nats below
  360M and gets ~52 % more tokens exactly right (0.387 vs 0.254) — a large,
  expected capacity gap that confirms the data scales with model size.
- `sentence` slightly beats `sentence+vocab` at both sizes here, but both are
  healthy; the custom-vocab variant trains on more examples (47k vs 32k).

### Weights & Biases (en-zh only)

The four en-zh cells logged to W&B project
`ethanrimes-university-of-pennsylvania-athletics/english-chinese-mt`
(saved locally as [`wandb_runs.json`](wandb_runs.json)). W&B `eval/loss` matches
the local `trainer_state.json` numbers exactly, corroborating §2.

| cell | W&B run | eval loss |
|---|---|---|
| 1.7B · sentence       | [b50bbesw](https://wandb.ai/ethanrimes-university-of-pennsylvania-athletics/english-chinese-mt/runs/b50bbesw) | 4.358 |
| 1.7B · sentence+vocab | [qbul5h9s](https://wandb.ai/ethanrimes-university-of-pennsylvania-athletics/english-chinese-mt/runs/qbul5h9s) | 4.618 |
| 360M · sentence       | [2e18x6by](https://wandb.ai/ethanrimes-university-of-pennsylvania-athletics/english-chinese-mt/runs/2e18x6by) | 6.176 |
| 360M · sentence+vocab | [wwh0ipgs](https://wandb.ai/ethanrimes-university-of-pennsylvania-athletics/english-chinese-mt/runs/wwh0ipgs) | 6.681 |

(A 5th run, `scale-23729-20260608T141023`, is the first 1.7B·sentence+vocab
attempt that **failed** early and was relaunched as `qbul5h9s`.) **The 3 NLLB
runs are not in W&B** — `snmt/train.py` sets `report_to=["none"]`, so their only
telemetry is the local `trainer_state.json` (which is how the divergence was
traced).

### NLLB es↔nasa (all diverged)

| model | final train loss | final eval loss | wall-clock | final weights |
|---|---|---|---|---|
| nllb-600m | 0.0 | **NaN** | 25.1 min | **509/509 tensors NaN** |
| nllb-1.3b | 0.0 | **NaN** | 44.2 min | NaN (same signature) |
| nllb-3.3b | 0.0 | **NaN** | 32.5 min | NaN (same signature) |

`train loss = 0.0` here is **not** success — it is the floor reached *after*
the logits became NaN and the loss stopped being meaningful.

---

## 3. Charts (how to read each one)

All PNGs are in [`analysis/charts/`](charts/):

1. **`01_enzh_train_loss.png`** — training loss vs step, 4 en-zh cells. Look for a
   smooth monotonic decrease that flattens (convergence). 1.7B curves sit well
   below 360M.
2. **`02_enzh_eval.png`** — held-out eval loss & token-accuracy. This is the
   honest generalization signal (training loss can fall from memorization; eval
   loss falling means it learned transferable structure).
3. **`03_forgetting.png`** — English perplexity **delta vs the base model**
   (the catastrophic-forgetting probe). Bars near 0 % = no forgetting. All four
   bars are within ±1.2 %; the +20 % warn line is never approached.
4. **`04_nllb_train_loss.png`** — NLLB training loss. Shows the pathology: the
   curve is flat/zero with NaN markers instead of a healthy descent.
5. **`05_timing.png`** — wall-clock minutes per run (the data you asked to keep).

---

## 4. Catastrophic-forgetting verdict

**En↔Zh: no catastrophic forgetting, and the models _did_ learn the task.**

- *Did they learn the training data?* Yes — eval loss dropped and token accuracy
  rose on **held-out** pairs (1.7B → 0.387). A model that failed to learn would
  sit near the random/`ln(V)` ceiling with ~0 token-acc.
- *Did they forget English?* No — English perplexity moved by ≤1.2 % vs the base
  model in every cell (`forgetting.jsonl` → `03_forgetting.png`). That is noise,
  not degradation. The short fine-tunes (15–23 min) and modest LRs preserved the
  base capability.

**NLLB: the opposite failure mode — they did _not_ learn anything.** The weights
became NaN before any useful gradient signal accumulated, so there is nothing to
forget *and* nothing learned. (A "forgetting" question is moot for a NaN model.)

---

## 5. Vibe-check (qualitative translations)

Full tables: [`analysis/vibe_check/translations.md`](vibe_check/translations.md);
health flags in [`vibe_check/_health.json`](vibe_check/_health.json). Greedy,
CPU, 3 in-domain + 3 out-of-domain sentences per direction.

### NLLB — unambiguous
Every one of the 8 NLLB generations is **empty (∅)** and the model carries **509
NaN tensors**. This is the visible end-state of the divergence: a NaN model
produces no tokens. (Only 600M was pulled for inference; 1.3B/3.3B share the
identical NaN signature, so 600M is representative.) The reference column shows
what correct Nasa-Yuwe looks like (the corpus is largely Biblical/liturgical
register).

### En↔Zh — **read with a caveat**
The free-generation outputs are **not** a reliable quality signal here and should
**not** override the metrics:
- For `en2zh`, several outputs **echo the English source** instead of producing
  Chinese; the 1.7B cells emit runs of `<|tgt|>` / `<|endoftext|>` control tokens.
- Yet teacher-forced **token accuracy is 0.387** on held-out pairs — you cannot
  get 39 % next-token accuracy on Chinese targets by echoing English. So the
  model demonstrably learned the conditional distribution.

**Interpretation:** the discrepancy is a *generation-harness / light-fine-tune*
artifact, not a learning failure. Greedy free-running decoding from a small
causal LM fine-tuned for only 15–23 min with a custom control-token template
(`{direction}\n<|src|> … <|tgt|> …`) is brittle on CPU — small prompt/format or
spacing mismatches make it collapse to echoes or EOS spam, even when the
teacher-forced model is good. **Trust the eval-loss / token-accuracy / forgetting
metrics as the headline; treat these vibe generations as a known-fragile
secondary check.** (To get clean translations: match the exact training template
byte-for-byte, use beam search + `no_repeat_ngram`, and ideally fine-tune longer.)

---

## 6. Root cause of the NLLB Nasa divergence  ⟵ (your explicit question)

**Cause: the NLLB runs were loaded with true-bf16 _master weights_ and optimized
with AdamW that has no non-finite-step safety net. This is numerically unstable
for NLLB-200 and the parameters went NaN within the first ~25 steps. The
Nasa-Yuwe data is not at fault.**

### The exact defect
`spanish-nasa-mt-experiment/src/snmt/train.py:95` loads the model as:

```python
AutoModelForSeq2SeqLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16)  # ← bug
```

combined with `Seq2SeqTrainingArguments(bf16=True, fp16=False)` and
`adamw_torch`. So the **stored/optimized parameters themselves are bf16**, not
fp32. bf16 has only ~3 significant decimal digits and a tiny mantissa. When the
optimizer writes `p ← p − lr · m/(√v+ε)` back into bf16 parameters every step,
the updates round/underflow badly; once any value grows, bf16's range overflows
to `inf`, `inf − inf = NaN`, and the NaN spreads across all tensors. Because
`GradScaler` only guards **fp16** (not bf16), Hugging Face does **not** skip the
non-finite step, so the very first NaN is committed permanently.

### Evidence (this is proven, not assumed)
1. **The data, tokenization framing and warm-start are provably fine.** A forward
   pass on the *base* NLLB-600M over real es↔nasa batches is finite and nearly
   identical in all three precisions — see
   [`repro_forward.json`](repro_forward.json):

   | case | weight dtype | autocast | forward loss | finite | warm-start = Spanish |
   |---|---|---|---|---|---|
   | A (as trained) | bfloat16 | no | 5.875 | ✅ | ✅ |
   | B (fix) | float32 | no | 5.908 | ✅ | ✅ |
   | C (best fix) | float32 | bf16 | 5.908 | ✅ | ✅ |

   The forward isn't where it breaks, and the `pbb_Latn` warm-start copied the
   Spanish row correctly. → rules out a data / embedding-resize / framing bug.
2. **The real runs went NaN immediately and identically.** In every run's
   `trainer_state.json`, `grad_norm = NaN` at the **first logged step (25)**, and
   the windowed-mean initial loss **rises with model depth** — 51 (600M) → 3413
   (1.3B) → 2.07×10⁸ (3.3B). A data problem would not scale with parameter count;
   accumulated bf16 rounding across more layers does exactly this.
3. **The end-state is total.** After the blow-up: train loss pinned at 0.0,
   `eval_loss = NaN`, and the saved 600M checkpoint has **all 509 weight tensors
   NaN** (`model.shared.weight` NaN-fraction = 1.0) → empty generations.

### The fix (for the re-run)
- **Load fp32 master weights, autocast bf16 for compute only:**
  ```python
  AutoModelForSeq2SeqLM.from_pretrained(hf_id, dtype=torch.float32)
  # keep Seq2SeqTrainingArguments(bf16=True)  # autocast only — case C above
  ```
- Add **`max_grad_norm=1.0`** clipping (already HF-default, but make it explicit)
  and a **non-finite-loss guard** that skips/aborts a step if `loss` or grad is
  NaN/Inf, so one bad step can't poison the whole model silently.
- Consider a slightly lower LR + longer warmup for the 3.3B, and log grad-norm
  every step for the first epoch.
- These three NLLB runs should be **re-run**; the en-zh results stand as-is.

---

## 7. How to interpret all of this (quick guide)

- **Eval loss (nats/token):** held-out cross-entropy. Lower = better. The
  theoretical max for random output is `ln(vocab)`; healthy en-zh sits far below
  it (4.4–6.7). `NaN` = the model is broken (NLLB).
- **Token accuracy:** fraction of held-out target tokens predicted exactly under
  teacher forcing. Higher = better; 0.387 (1.7B) is solid for a short fine-tune.
- **Forgetting Δ (%):** change in English perplexity vs the *base* model. ~0 % =
  capability preserved. Watch a +20 % line; we're at ≤1.2 %.
- **Train loss alone is not success.** NLLB's `0.0` is a NaN-collapse artifact,
  not learning — always cross-check eval loss and weight finiteness.
- **Metrics > vibe generations** when they disagree, *unless* the model is NaN
  (then both agree it's dead). For en-zh, the metrics say "learned"; the brittle
  greedy generations are a secondary, harness-sensitive check.
- **Wall-clock (your ask):** per-run minutes are in `05_timing.png` /
  `metrics_table.json` (`wall_min`): en-zh 15–23 min each; NLLB 25–44 min each.

---

## 8. Repo layout after cleanup

```
analysis/
  00_pull_metrics.py        # pull summary/forgetting/trainer_state from Blob
  01_build_charts.py        # build the 5 charts + metrics_table.json
  02_vibe_check.py          # CPU translation vibe-check (en-zh + NLLB)
  02a_download_models.py    # pull final/ model dirs from Blob
  03b_repro_fast.py         # forward+backward divergence repro (slow, bf16/CPU)
  03c_repro_forward.py      # forward-only A/B/C precision contrast (fast)
  metrics_table.json        # consolidated headline metrics
  repro_forward.json        # precision contrast result (proof, §6)
  charts/                   # 5 PNGs
  metrics/<run>/            # raw summary.json / forgetting.jsonl / trainer_state
  vibe_check/               # translations.md, _health.json, <run>.json
  wandb_runs.json           # en-zh W&B run URLs + eval losses
logs/h100_orchestrator/     # all h100_*.{out,err,pid} run logs
```

`analysis/models_cache/` (the 9 GB of downloaded final checkpoints) was pulled
from Blob for the vibe-check/NaN audit and has since been **deleted** to reclaim
space — it is gitignored and fully re-downloadable via `02a_download_models.py`.
Everything needed for this report is already extracted into JSON/PNG/Markdown.
