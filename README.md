# Nasa Yuwe MT — monorepo

Umbrella repository for all machine-translation work supporting a **Spanish → Nasa Yuwe
(Páez)** MT model. Data — not compute — is the bottleneck for the real Nasa Yuwe model, so
this repo is built to (a) reconstruct an **English↔Chinese proxy** that mirrors the existing
Nasa-Spanish parallel data, (b) run cheap, heavily-parallelized **data-ablation × model-size**
sweeps on a single rented Azure **H100**, and (c) decide which proposed datasets are
highest-value to commission for real human translation.

## Why English↔Chinese as a proxy
The base model family (SmolLM2) is English-centric, so we can only realistically *train from*
English. Chinese is the held-out target whose acquisition-from-near-zero behavior best
approximates Nasa Yuwe: different script, very different (isolating vs. polysynthetic)
grammar, and minimal lexical transfer. En→Zh distance ≫ En→Es distance, so En↔Zh results
generalize to Es↔Nasa far better than the older En↔Es experiment did.

## Layout
```
nasa-yuwe-mt/
├── shared/                          installable infra package (nymt_shared)
│   ├── config.py                    Azure IDs, storage layout, model registry
│   ├── blob.py                      Blob upload / download / mirror
│   ├── h100.py                      H100 provision / deprovision (idempotent teardown)
│   ├── mirror.py                    background checkpoint+metrics → Blob loop
│   ├── bible.py                     fetch real WEB(en)+CUV(zh) by book/chapter/verse
│   └── translate.py                 Gemini-subagent batch translation driver
├── english-chinese-mt-experiment/   PRIMARY: En↔Zh proxy training pipeline (pkg: ecmt)
│   └── data-en-zh/                  reconstructed mirror dataset + ablation subsets
├── english-spanish-mt-experiment/   prior En↔Es work (pkg: en_es_mt)
│   └── experiment-results/          completed En↔Es run output (synced to Blob)
├── pyproject.toml                   workspace; installs nymt_shared
├── .env.example
└── README.md
```

## Azure resources
| Thing | Value |
|---|---|
| Subscription | `AIERP_DevPlayground_CorporateFunctions` (`5423a68c-2f22-4512-a82c-eb6ed44d9aaf`) |
| Resource group | `nasa-yuwe-mt-rg` (westus2) |
| Storage account | `nasayuwemtdata` |
| GPU SKU | `Standard_NC40ads_H100_v5` (1× H100 80 GB) — quota 0/360 vCPUs in westus2 |

### Blob container/prefix layout
```
training-data/            existing Nasa-Spanish source data (untouched)
reconstructed-en-zh/      raw/ processed/ subsets/      (the proxy dataset)
experiments/en-es/        results/{checkpoints,metrics,results,logs}
experiments/en-zh/<run>/  checkpoints/ metrics/ logs/ samples/
```

## Quick start
```bash
# 1. install the shared infra package + (per-experiment) its own package
pip install -e .                                   # installs nymt_shared
pip install -e english-chinese-mt-experiment       # installs ecmt

# 2. reconstruct the En↔Zh proxy dataset (Bible fetched real, rest via Gemini subagents)
python english-chinese-mt-experiment/scripts/10_reconstruct_from_nasa.py --dry-run

# 3. provision + train + auto-teardown (NOTHING runs on GPU until you pass --go)
python shared/h100.py up                           # provision H100 (spot)
python english-chinese-mt-experiment/scripts/11_run_ablations.py --matrix core
python shared/h100.py down                          # idempotent teardown
```

## Cost discipline
The H100 is expensive (~$6.28/hr spot). All orchestration:
- packs multiple small runs onto one GPU concurrently,
- mirrors checkpoints+metrics to Blob every ~5 min (so an early kill loses ≤5 min),
- auto-deprovisions on completion / budget tripwire / error / SIGINT.

See `plan.md` (session) and `docs/` for the full ablation matrix and GPU-hour estimates.
