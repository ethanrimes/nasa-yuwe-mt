# Data-Ablation Experiment Plan — Nasa Yuwe MT (via English↔Chinese proxy)

> **Authoritative plan** for the data-ablation × model-size training sweep.
> Scope: code + dry-run only — **NO GPU spend until explicitly approved.**
> (The older `english-chinese-mt-experiment/PLAN.md` is the *generic* data-volume
> scaling study and is superseded by this document for the Nasa-Yuwe ablation work.)

---

## 1. Goal

Train a **Spanish → Nasa Yuwe (Páez)** MT model where **data, not compute, is the
bottleneck**. Because the only sizable base LMs (SmolLM2) are English-centric, we cannot
train directly on Nasa Yuwe yet. Instead we build an **English↔Chinese proxy** that mirrors
the existing Nasa-Spanish parallel data **row-for-row**, run cheap GPU ablations on the
proxy, and use the results to decide **which real datasets are worth commissioning for
human Nasa translation** (the expensive resource).

**Why En→Zh as the proxy:** Chinese is acquired from near-zero by an English base model
(isolating script, very different grammar, no lexical transfer) — the closest available
analogue to Spanish→Nasa Yuwe. Spanish→English would be far too easy and over-optimistic.

---

## 2. Two axes of the core sweep

| Axis | Values |
|---|---|
| **Model size** | `SmolLM2-360M`, `SmolLM2-1.7B` |
| **Data ablation** | `sentence_only`, `sentence_plus_vocab` |

**Core matrix = {360M, 1.7B} × {sentence_only, sentence_plus_vocab} = 4 runs**
(defined in `english-chinese-mt-experiment/configs/ablations.yaml`).

### Bidirectional training (Es↔Nasa, mirrored as En↔Zh)
The model is trained **bidirectionally** — both directions, not Es→Nasa only. This is
already the implemented design and does **not change the ablation axes**; it is orthogonal
to (and applies equally to) every run:
- Each pair is expanded to **two** direction-tagged training examples
  (`expand_to_bidirectional` → `>>en2zh<<` / `>>zh2en<<`; `formatting.py`). Example count is
  therefore `2 × pairs × epochs` — **already** baked into the cost model (`schedule.py`).
- Vocabulary/dictionary pairs train both directions too (a `house=casa` entry teaches
  lookup *and* generation), which is exactly why the `sentence_plus_vocab` lift is worth
  isolating.
- **Evaluation runs both directions** (`eval.py` scores `en2zh` *and* `zh2en`), so every
  reported number is a 2-direction × {deterministic, LLM-judge} grid (see §6, Evaluation).

The only practical implication: report and gate on **both directions** (a checkpoint can be
strong Nasa→Es but weak Es→Nasa); the matrix structure is unchanged.

### Two-stage design (baseline → checkpoint-resume signal probes)
Per the research goal ("how much signal does each *additional* dataset give, on top of what
we already have?"), the sweep is **two stages**:

- **Stage A — baselines, trained from scratch (non-negotiable base = the data we already
  have):** the 4 core runs above. The two anchor baselines are `sentence_only` and
  `sentence_plus_vocab` (= *everything currently in hand*).
- **Stage B — incremental data-signal probes, warm-started from the best Stage-A
  checkpoint:** each candidate dataset (and combinations) is stacked **on top of
  `sentence_plus_vocab`** and the best checkpoint is **resumed** (not retrained from
  scratch), measuring **ΔBLEU / Δchrf++ / ΔCOMET / Δjudge**. Warm-start makes each probe a
  fraction of a from-scratch run, so we can afford many combinations cheaply. This directly
  ranks **which 30–40k sentences are worth commissioning for human Nasa translation** (§6).

### The two core data ablations (exactly what was requested)
Derived from the reconstructed En-Zh mirror of the Nasa-Spanish data, split by `level`:

| Ablation | Pairs (full) | Contents |
|---|---|---|
| `sentence_only` | **16,446** | every `level == sentence` pair |
| `sentence_plus_vocab` | **24,229** | sentence pairs **+** all `vocabulary` pairs |

> The marginal lift of `sentence_plus_vocab` over `sentence_only` measures **how much the
> Nasa-Yuwe dictionary/vocabulary data is worth** as training signal.

> **Current on-disk state:** the built subsets are *partial* (7,983 / 8,157 rows) because
> only the real Bible (7,902) + 1 Gemini smoke-test batch (200) + external community pairs
> (55) are translated so far. They grow to the full 16,446 / 24,229 once the deferred
> 81-batch Gemini translation job runs. See `english-chinese-mt-experiment/data-en-zh/`.

---

## 3. Underlying data (the structure being mirrored)

`nasa_yuwe_parallel_dataset.jsonl` — **24,229 pairs** (was 24,174; +55 from external
community-corpus ingestion).

| Source | Pairs | Level | En-Zh reconstruction |
|---|---|---|---|
| bible_nt | 7,906 | sentence | **Fetch real** En (WEB) + Zh (CUV) by verse ref |
| broomva_instruct | 7,076 | 4,364 sent / 2,712 vocab | Gemini (Es→En, Es→Zh) |
| americasnlp_2024 | 3,847 | sentence | Gemini |
| living_dict | 3,500 | vocabulary | Gemini |
| broomva_translation | 1,023 | vocabulary | Gemini |
| kwesxyuwe_thematic | 548 | vocabulary | Gemini |
| territorios_narrados | 164 | sentence | Gemini |
| procuraduria_institutional | 57 | sentence | Gemini |
| bible_ot_creation | 46 | sentence | **Fetch real** En + Zh by verse ref |
| adres_health | 7 | sentence | Gemini |
| **external_repos** (darcygith/darcysem RASA+cultural) | **55** | sentence | Gemini |

Totals: **sentence = 16,446**, **vocabulary = 7,783** → **24,229**.

---

## 4. H100 orchestration (single GPU, max parallelism)

One rented **Azure `Standard_NC40ads_H100_v5`** (1× H100 80GB). Design goal: **kill the GPU
the instant the matrix finishes.** Implemented in `shared/nymt_shared` +
`english-chinese-mt-experiment/scripts`:

- One `az vm create` (spot, cloud-init installs CUDA/uv/repo) → SSH → train → **auto
  `az vm delete`** (idempotent teardown also fires on error / SIGINT / `--max-budget-hours`).
- **Pack multiple runs concurrently** on the one 80 GB GPU (usable 72 GB): 360M ≈ 8–10 GB
  each, 1.7B ≈ 30–35 GB each. `shared/schedule.py` packs runs into VRAM-bounded waves
  (first-fit-decreasing by VRAM, so the long 1.7B runs co-locate in one wave and overlap).
- **NVIDIA MPS** is started on the VM before the wave loop (`run_matrix_on_vm.py:start_mps`)
  so co-resident CUDA processes share SMs **spatially** instead of context-switching; each
  run gets a `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` budget weighted by its model's VRAM share
  (homogeneous waves split evenly, e.g. 50/50 for two 1.7B runs). Best-effort: if the MPS
  binary is absent the runner falls back to plain concurrent processes.
- Frequent `save_steps`; a background mirror loops every ~5 min: rsync new checkpoints +
  `metrics.jsonl` → Azure Blob, so an early kill loses ≤ 5 min of work.
- **Resume path:** any Blob snapshot can be re-pulled to continue training as new (real
  Nasa, or newly-translated proxy) data arrives — directly enabling the "promising snapshot
  → continue training" workflow.

Metrics collected per run: see **§6 (Evaluation)** — deterministic (BLEU, chrF++, COMET,
both directions, every eval step) **+ LLM-as-judge** + an **English-perplexity
catastrophic-forgetting probe** + the full checkpoint time-series.

---

## 5. GPU-hour & cost estimate

`NC40ads_H100_v5` ≈ **$6.98/hr on-demand**, **~$6.28/hr spot**. Datasets are tiny
(16K–30K pairs ⇒ short runs). Estimates assume **max parallel packing** (bf16 +
grad-checkpointing, 3 epochs), computed by `shared/schedule.py`
(throughput 360M ≈ 38 ex/s, 1.7B ≈ 11 ex/s).

| Matrix | Runs | Serial GPU-h | **Packed GPU-h** | Cost (spot) |
|---|---|---|---|---|
| **Core** {360M,1.7B}×{sentence, sentence+vocab} | 4 | ~8.1 | **~4.8** | **~$30** |
| **Extended** (4×360M proxy ablations + 1×1.7B `+seed`) | 5 | ~9.7 | **~5.9** | **~$37** |
| **Core + Extended combined** (one VM session) | 9 | ~17.8 | **~9.6** | **~$61** |

Packing detail (core): **Wave 1** = both 1.7B (~68 GB, ~3.7 h), **Wave 2** = both 360M
(~20 GB, ~1.1 h). Provision/teardown adds ~10–15 min wallclock (VM uptime, not compute).
`--max-budget-hours` (default 6) is a hard tripwire that flushes snapshots to Blob and
auto-deletes the VM.

**Recommended phasing:** run **Core first (~4.8 GPU-h / ~$30)**; only after inspecting Core
results, build proxy data for the winning proposed datasets and run the **Extended**
ablations (~+5.9 GPU-h / ~+$37).

---

## 6. Proposed-dataset prioritization + Evaluation

Research objective restated: **we can afford to manually translate ~30–40k more sentences
(Spanish→Nasa). Which are the highest-value 30–40k?** The proxy answers this cheaply: for
every candidate we build an **En↔Zh proxy**, stack it on the `sentence_plus_vocab` baseline
via **Stage-B checkpoint-resume** (§2), and rank datasets by **marginal** ΔBLEU/Δchrf++/
ΔCOMET/Δjudge. We spend cheap GPU dollars to avoid wasting expensive human-translation
dollars. The data we already hold (`sentence_plus_vocab`) is the **non-negotiable base** of
every Stage-B run.

### 6.1 Which proposed sources already have En↔Zh vs. need our translation
This decides Gemini cost for the *proxy* (the real Es→Nasa side is always manual). Sources
with a **vetted Spanish source side** (FLORES/TICO/XNLI) also let human translators start
immediately for the real target.

| Proposed source | En↔Zh already public? | Action for the proxy |
|---|---|---|
| **FLORES+ / FLORES-200** | ✅ **Yes** — pro-translated into 200 langs incl. `eng` + `zho_Hans/Hant` (also `spa`) | Pull directly. **EVAL-ONLY — never train.** |
| **TICO-19** | ✅ **Yes** — 36 langs incl. `en`, `zh`, `es` | Pull directly (health train domain) |
| **AmericasNLI → XNLI proxy** | ✅ **Yes** — XNLI has `en`+`zh`+`es` (AmericasNLI itself has **no** Nasa/zh) | Pull XNLI directly (semantic-inference eval) |
| **SIL Sem-Domains / RWC dict, Swadesh-207** | ✅ **Mostly** — CC-CEDICT (free en↔zh dict) + Wiktionary Swadesh | Pull CC-CEDICT/Wiktionary (folds into vocab tier) |
| **OLDI / NLLB-Seed** | ⚠️ **En source only** (Seed = En→low-resource; zh is *not* a seed target) | **Gemini** translate zh |
| **NLLB Multi-Domain** | ❌ targets are low-resource (Aymara, Bhojpuri…), not zh | **Gemini** (or use only as a domain guide) |
| **Gamayun kits (TWB)** | ❌ English source; zh not standard | **Gemini** translate zh |
| **AVENUE elicitation** | ❌ English minimal-pair sentences only | **Gemini** translate zh |
| **Dahl TMA / PAWS / Lingua / Max-Planck Q's** | ❌ *methodologies* (prompts, not parallel corpora) | **Generate** sentences, then Gemini zh |
| **Living Tongues Master List** (4,932 vocab + 1,959 phrases) | ❌ English-based list | **Gemini** translate zh |
| **LORELEI / data.org / Masakhane** | ❌ frameworks/process, not corpora | N/A (methodology only) |

**Takeaway:** FLORES+, TICO-19, XNLI and CC-CEDICT/Swadesh are **free En↔Zh** (no Gemini) —
build their proxies first at zero translation cost. Everything else needs Gemini for the zh
side (same resumable agent-fulfilled queue as the base data).

### 6.2 Stage-B proxy ablations (measure marginal value, on top of `sentence_plus_vocab`)
Each stacks **on the `sentence_plus_vocab` base** and warm-starts the **best Stage-A 360M
checkpoint** (top picks also re-checked on 1.7B). Combinations probe whether signal is
additive or redundant. `+grammar_synth` is **net-new, Gemini-built** (§6.3).

| Stage-B ablation | + adds (proxy pairs) | zh cost | Probes |
|---|---|---|---|
| `+seed` | OLDI/NLLB-Seed ~6.2k | Gemini | Domain-balanced backbone — expected top single source |
| `+morph` | AVENUE + Dahl-TMA stress ~5–7k | Gemini | Person/tense/aspect/evidentiality/negation (where polysynthetic MT breaks) |
| `+grammar_phrases` | Living Tongues 1,959 | Gemini | Broad construction coverage |
| `+grammar_synth` | **synthetic feature-coverage ~2–3k** | **Gemini-built** | Do we *need* explicit coverage of every main En+Zh grammatical feature? (§6.3) |
| `+health` | TICO-19 ~3k | free (zh exists) | Real-world utility, high BLEU/$ |
| `+domain` | NLLB-MD + Gamayun ~6–9k | Gemini | Domain transfer (news/informal/gov/agri) |
| `+dict_ext` | CC-CEDICT/Swadesh slice ~5k | free | Extra lexical/OOV signal beyond current vocab |
| **best-combo** | top-2/3 stacked | mixed | Additive vs. redundant signal |
| **FLORES+ / XNLI** | — | free | **Held-out eval only** (never trained on) |

Recommended **~30–40k commission packet** (the proxy winners → human Nasa translation):
`+seed 6k · +morph 6k · +grammar(phrases+synth) 5k · +health 3k · +domain 8k · +dict_ext 5k`
≈ **33k**, holding FLORES+ (~3k) + XNLI out for evaluation. Commission order follows the
Stage-B ΔCOMET/Δjudge ranking; default prior is `seed → morph → grammar → health → domain`.

### 6.3 Synthetic grammatical-feature-coverage set (`+grammar_synth`, Gemini-built)
A deliberately-constructed corpus that **exercises every main grammatical feature of English
and Chinese at least N times**, to test how much signal explicit grammatical coverage adds
vs. equal-size random sentences. Spec: `configs/grammar_coverage_spec.yaml` (drives Gemini
generation via the same batch queue). Feature inventory (abbrev.):
- **English:** tense×aspect (simple/progressive/perfect/perfect-progressive), modality,
  articles & definiteness, plural/possessive morphology, comparatives/superlatives,
  passive, relative clauses, conditionals (0–3), wh-/polar questions, subject-aux inversion,
  gerund vs. infinitive, phrasal verbs, negation scope.
- **Chinese:** aspect particles 了/着/过, classifiers/measure words, 把-construction,
  被-passive, topic–comment, serial-verb & resultative complements, 的/得/地, 是…的 cleft,
  comparatives 比, polar 吗 / A-not-A, time-via-adverb (no tense morphology), 了₂ change-of-state.
For the **real Es↔Nasa**, the analogue targets Nasa Yuwe's polysynthetic features (verbal
suffix chains, evidentiality, number/person agreement) — the AVENUE/TMA philosophy applied
to the actual target language.

### 6.4 Evaluation (deterministic + LLM-as-judge, both directions)
Because many high-fidelity translations exist for one input, single-reference n-gram metrics
under-credit valid outputs — so we run **two complementary tracks** at every eval step, for
**both** `en2zh` and `zh2en`:
- **Deterministic (wired, `eval.py`):** **BLEU** (zh tokenizer for the zh side), **chrF++**
  (word_order=2; robust to morphology — important for the polysynthetic target), **COMET**
  (learned, reference-based) + an **English-perplexity forgetting probe**. Cheap, on-GPU,
  fast feedback during training.
- **LLM-as-judge (new, `shared/llm_judge.py`):** an agent-fulfilled judge queue (mirrors the
  Gemini batch pattern — no API key needed) that scores each hypothesis on **adequacy /
  fluency / terminology (1–5)** against source + reference(s), and does **pairwise A/B**
  between candidate checkpoints to rank them. Run **off the GPU** on the held-out
  FLORES+/XNLI hypotheses *after* a run (so it never holds the H100). For the real
  low-resource Es↔Nasa, the LLM judge is weak → there we lean on chrF++/COMET + human spot
  checks; the proxy is precisely where we **validate the judge methodology** before trusting
  it.

Gate metric for "promising checkpoint → resume with more data": **ΔCOMET + Δjudge (both
directions)** on the held-out set, not BLEU alone.

### 6.5 Headline "train-on-everything" → deficiency diagnosis → checkpoint curves
Complementary to the Stage-B *per-source* probes (§6.2): train **one upper-bound proxy
model on `everything`** (= `sentence_plus_vocab` + every built candidate proxy; the
`everything` subset + `*-everything` headline runs in `ablations.yaml`), then **diagnose
where it is still deficient**. Implemented in `src/ecmt/training/eval_stratified.py`:
- **Stratified eval** groups per-segment BLEU/chrF++ by **topic/domain** (free FLORES
  metadata), **length bucket**, **grammatical feature** (grammar-spec tags) and **direction**.
- `find_deficiencies()` flags strata (n ≥ 15) scoring < 0.85× overall → **the capabilities
  no current source covers → what NEW Nasa data to commission** (over-sample those buckets).
- `evaluate_checkpoints_over_time()` re-evals **every retained checkpoint** → `eval_curve.jsonl`,
  so we watch *which capabilities emerge early/late/never* and pick the checkpoint to resume
  for the next data wave.
- **Attribution split:** "everything + diagnose" finds *gaps*; Stage-B add-on-top gives
  *per-source credit*. Use both — diagnosis says *what kind* of data is missing, Stage-B
  says *which specific source* supplies it cheapest.

**What FLORES+ actually contains (the eval gold):** 3,001 sentences sampled from English
Wikipedia across **Wikinews / WikiJunior / WikiVoyage**, ~21 words/sentence, ~10 topics
(Travel, Politics, Science, Crime, Sports, Health, Geography, Entertainment, Nature,
Disasters), professionally translated into 200+ langs incl. `eng`/`zho`/`spa`; splits dev
(997) + devtest (1012) + hidden test (992). Held out because training on the published
benchmark breaks comparability. **Similar gold alternatives** (so we are not single-benchmark):
**NTREX-128** (Microsoft; 1,997 WMT19-**news** sentences → 128 langs incl. zh+es; news-domain
complement to FLORES's Wikipedia domain — now wired via `loaders.load_ntrex128()`, eval-only),
plus WMT news test sets, TICO-19 test split (health), Tatoeba, IWSLT/TED (en-zh).

### 6.6 Foundational human-translation commission spec (the real Es→Nasa target)
Canonical blueprint: **`english-chinese-mt-experiment/configs/commission_spec.yaml`** — the
answer to *"what must humans translate for a well-rounded model?"* It buys **capabilities**,
not just sentences; the proxy Stage-B/headline passes above **rank these before any human
spend**. The existing 24,229-pair base skews **formal/written legal + biblical + bare vocab**;
these buckets fill the gaps. (**AmericasNLP-2025 added a Nasa side but it is
constitution-only → low diversity; it does not close these gaps** — use its test split only as
a secondary eval.)

| # | Bucket | Capability bought | Spanish source | Target | Proxy probe |
|---|---|---|---|---|---|
| 1 | **general_backbone** | domain-balanced training spine | OLDI/NLLB-Seed | 6,000 | `+seed` |
| 1 | **grammar_morphology** ★ | systematic morphology via minimal pairs (verbal suffix chains, TAM, agreement, valence, evidentiality) | AVENUE+Dahl-TMA+PAWS+Swadesh-in-frames | 5,000 | `+morph`/`+grammar_synth` |
| 1 | **health_crisis** | health/crisis register, high real-world value | TICO-19 + Gamayun | 4,000 | `+health` |
| 2 | **daily_conversational** | spoken/informal register (base has ~none) | NLLB-MD informal + Tatoeba-style | 4,000 | `+domain` |
| 2 | **news_contemporary** | named entities/numbers (anti-hallucination) | NLLB-MD News / WMT-es | 3,000 | `+domain` |
| 2 | **education_pedagogical** | graded-reader/classroom register | **CRIC cartillas** (EIB) | 3,000 | `+domain` |
| 3 | **core_lexicon_in_frames** | core vocab in frames (seeds BPE, cuts OOV) | Living Tongues/RWC in frames | 2,000 | `+dict_ext` |
|   | *method:* **back-translation** | amplify scarce data w/ monolingual target | monolingual Nasa (radio/social/oral) | ~0 human | `bt_mono_zh` |

**≈ 27k commissioned (incl. lexicon) + FLORES+ 3,001 eval quarantine**, headroom to the ~30k
budget reserved for whichever buckets the headline-run deficiency diagnosis (§6.5) flags.
**Three rules that matter more than the exact mix:** (1) **reserve + quarantine eval FIRST**
(FLORES+ Spanish; dedup all train buckets against it) — otherwise we cannot tell if more data
helps; (2) **★ grammar_morphology is the highest-leverage spend** — 5k systematic minimal
pairs teach more morphology than 15k random sentences and is exactly what constitution/
AmericasNLP data cannot fake; (3) **diversity > volume** — cap any single domain at ≤ 25%
(the base is already over-indexed on legal/biblical). Commission **order** follows the
Stage-B ΔCOMET/Δjudge ranking; default prior `seed → grammar → health → conversational →
news/education`.

---

## 7. Status & remaining gates

- [x] Monorepo, shared infra, En-Zh reconstruction pipeline, H100 orchestration (dry-run
  verified, no spend), Azure Blob backfill of en-es results, external-corpus ingestion.
- [x] Core ablation matrix + model configs + cost model wired; bidirectional training +
  both-direction deterministic eval already implemented.
- [x] Two-stage design (Stage-A baselines → Stage-B checkpoint-resume signal probes) + the
  source En↔Zh availability table + ~30–40k commission packet (this doc, §6).
- [x] LLM-as-judge scaffold (`shared/llm_judge.py`, agent-fulfilled queue) + synthetic
  grammar-coverage spec (`configs/grammar_coverage_spec.yaml`).
- [x] Headline "train-on-everything" → **deficiency-diagnosis** + checkpoint-over-time eval
  (`src/ecmt/training/eval_stratified.py`, unit-tested; `everything` subset + `*-everything`
  runs + stratified `eval` block in `ablations.yaml`); FLORES topic/domain metadata preserved
  + **NTREX-128** wired as a second news-domain gold (`loaders.load_ntrex128`, eval-only).
- [x] **Foundational commission spec** (`configs/commission_spec.yaml`) — capability-based
  Spanish→Nasa human-translation blueprint (~27k train + FLORES+ 3k eval), each bucket linked
  to its proxy probe; back-translation method ablation registered.
- [ ] **Deferred (awaiting "go"):** (a) full 81-batch Gemini translation → grows base subsets
  to 16,446 / 24,229; (b) build the free En↔Zh proxies (FLORES+/TICO-19/XNLI/CC-CEDICT) +
  Gemini-translate the rest + generate `+grammar_synth` + build `everything.jsonl` + register
  NTREX in the downloader (`scripts/01_download_data.py` / `configs/data.yaml`); (c) **provision
  H100 → Stage-A Core (~4.8 GPU-h / ~$30) + headline `everything` run → resume best checkpoint
  for Stage-B probes + deficiency diagnosis**.

## 8. Out of scope (for now)
- Renting the H100 / any GPU spend (code + dry-run only until approved).
- Real Nasa-Spanish human-translation commissioning (we only pick the list).
- LoRA, RL/preference tuning, inference quantization.
