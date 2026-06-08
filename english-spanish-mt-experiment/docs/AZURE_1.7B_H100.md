# SmolLM2-1.7B size-sweep on Azure H100 (AIERP subscription)

The assessment (`assessment/REPORT.md`) shows the 360M model hits a **capability
ceiling**: high-frequency Spanish words are never produced under teacher forcing,
and the plateau does not lift with 100× more data. The decisive test is to retrain
the **identical recipe** with a ~4.7× larger model and see whether the plateau
lifts. This doc provisions that run on the H100 quota you already have.

> **TL;DR cost & time:** the full 5-tier 1.7B sweep is **~4.9 H100-hours of pure
> compute (~7 h wall-clock with eval/checkpointing)**, i.e. **~$35–55 on-demand**
> or **~$30–45 spot** at westus2 H100 prices. T1M alone is ~4 wall-clock hours
> (~$28 on-demand). See the cost table at the bottom.

Everything here is *the same pipeline* as the 360M runs — only the **model config**
(`configs/model_1.7b.yaml`) and the **compute target** (H100) change.

---

## 0. Resources you already have (from the access audit)

| Thing | Value |
| --- | --- |
| Subscription | `AIERP_DevPlayground_CorporateFunctions` = `5423a68c-2f22-4512-a82c-eb6ed44d9aaf` |
| Region | **westus2** |
| H100 quota | **360 vCPU** on `Standard_NCadsH100v5` → up to **9× `Standard_NC40ads_H100_v5`** (1× H100 NVL 94 GB each) |
| Storage account | `nasayuwemtdata` (ADLS Gen2, **Entra-auth only**, container `training-data`) |
| Resource group | `nasa-yuwe-mt-rg` |
| Your data role | Storage Blob Data Contributor |

A single `NC40ads_H100_v5` (1× H100, 94 GB) is plenty: SmolLM2-1.7B full
fine-tune at seq 384 needs ~25–30 GB (bf16 weights + fp32 AdamW states +
grad-checkpointed activations). No multi-GPU sharding required.

`.env` is already filled in with these values.

---

## 1. One-time provisioning (CLI, PowerShell)

```powershell
az login
az account set --subscription 5423a68c-2f22-4512-a82c-eb6ed44d9aaf

$RG  = "nasa-yuwe-mt-rg"      # reuse the RG that already holds the storage account
$LOC = "westus2"             # MUST match the H100 quota + storage region (no egress)
$WS  = "aml-en-es-mt"

# 1. AML workspace (co-located with the storage account in westus2).
#    Attach the existing storage account so data + compute + tracking share a region.
az ml workspace create --name $WS --resource-group $RG --location $LOC `
    --storage-account "/subscriptions/5423a68c-2f22-4512-a82c-eb6ed44d9aaf/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/nasayuwemtdata"

# 2. H100 compute cluster. min=0 so it scales to zero (no idle burn).
#    Bump --max-instances up to 5 to run several tiers in parallel (quota allows 9).
az ml compute create `
    --name gpu-h100-1x `
    --type AmlCompute `
    --size Standard_NC40ads_H100_v5 `
    --min-instances 0 --max-instances 1 `
    --idle-time-before-scale-down 1800 `
    --workspace-name $WS --resource-group $RG

# 3. Register the training environment (curated PyTorch image + our package).
az ml environment create -f azure/environment.yml `
    --workspace-name $WS --resource-group $RG
```

> **Spot (optional, ~10% cheaper, evictable):** add
> `--tier low_priority` to the `az ml compute create` call. Training is
> checkpoint-resumable (`save_total_limit`, `load_best_model_at_end`), so an
> eviction only costs the steps since the last snapshot.

---

## 2. Stage the data (once)

The 1.7B run mounts the **same processed corpus** the 360M runs used. Upload the
local `data/processed` + `data/eval` to the `training-data` container (Entra auth —
no account keys), then register a versioned data asset.

```powershell
# Upload processed tiers + eval sets to the ADLS Gen2 container.
az storage blob upload-batch `
    --account-name nasayuwemtdata --destination training-data `
    --source ./data/processed --destination-path en-es-mt/data/processed `
    --auth-mode login
az storage blob upload-batch `
    --account-name nasayuwemtdata --destination training-data `
    --source ./data/eval --destination-path en-es-mt/data/eval `
    --auth-mode login

# Register as an AML data asset (mounted read-only into each job).
az ml data create `
    --name en-es-parallel-processed `
    --type uri_folder `
    --path "https://nasayuwemtdata.blob.core.windows.net/training-data/en-es-mt/data/processed" `
    --workspace-name $WS --resource-group $RG
```

The trainer expects, under the mounted root: `T<tier>/train.jsonl` and
`T<tier>/val.jsonl`. The FLORES + English-holdout eval sets live in `data/eval/`
**inside the repo** and are uploaded automatically as part of the job `code`
(the trainer reads `eval_dir: data/eval` relative to the working dir), so you only
need to stage the per-tier `train`/`val` JSONL under `data/processed`.

---

## 3. Submit the size-sweep

`azure/submit_job.py` now takes `--model-config` and `--model-name`, so the only
difference from a 360M submission is pointing it at `configs/model_1.7b.yaml` and
the H100 cluster:

```powershell
# Smoke-test first: cheapest tier, stream logs end-to-end.
uv run python azure/submit_job.py --tiers 10000 `
    --model-config configs/model_1.7b.yaml --model-name SmolLM2-1.7B `
    --compute gpu-h100-1x --wait

# Then the rest of the sweep (each job auto-scales the cluster up and back to 0).
uv run python azure/submit_job.py --tiers 50000 100000 500000 1000000 `
    --model-config configs/model_1.7b.yaml --model-name SmolLM2-1.7B `
    --compute gpu-h100-1x
```

Each submission prints a Studio URL for live logs + MLflow/W&B metrics. Jobs are
tagged `model=SmolLM2-1.7B` so they sort separately from the 360M runs.
`per_device_train_batch_size=32, grad_accum=1` (in `configs/train.yaml`) gives the
**same effective batch 32** and therefore the **same optimizer-step counts** as the
360M runs — a clean controlled comparison.

Pull a finished run's artifacts (checkpoints + `runs/registry.jsonl`):

```powershell
az ml job download --name <job_name> --download-path ./runs/T<tier>-1.7b-<job_name> `
    --workspace-name $WS --resource-group $RG
```

---

## 4. Analyze (close the loop on the capacity question)

Re-run the assessment against the 1.7B checkpoints to see whether the
"taught-but-not-learned" plateau lifts:

```powershell
# Point the assessment at the 1.7B checkpoints, then:
uv run python assessment/taught_not_learned.py   # E6: do entre/parte/gran/... now recall?
uv run python assessment/make_report.py          # regenerates REPORT.md
```

**Decision rule:** if the 1.7B model recalls the high-frequency words that 360M
never produced (under teacher forcing) at matched data tiers, the bottleneck was
**model capacity**. If the same words stay at 0% recall, capacity is *not* the
lever and the limit is the data/objective.

---

## 5. Cost & GPU-hours (westus2 H100, live retail prices)

`Standard_NC40ads_H100_v5` = 1× H100 NVL (94 GB): **$6.98/hr on-demand**,
**$6.28/hr spot**. Hours below scale the measured 360M `total_flos` by the
parameter ratio (1.711B / 0.362B = 4.73×) at ~40% H100 MFU, then add ~1.5× for
eval/checkpoint/IO overhead.

| Tier | compute h | wall-clock h | on-demand | spot |
| --- | --: | --: | --: | --: |
| T10k | 0.11 | ~0.2 | ~$1 | ~$1 |
| T50k | 0.33 | ~0.5 | ~$3 | ~$3 |
| T100k | 0.52 | ~0.8 | ~$5 | ~$5 |
| T500k | 1.30 | ~2.0 | ~$14 | ~$12 |
| T1M | 2.58 | ~3.9 | ~$27 | ~$24 |
| **All 5** | **4.85** | **~7.3** | **~$51** | **~$46** |

Notes:
- Numbers move with **MFU (±25%)**; range for the full sweep is ~6–11 wall-clock h
  (~$40–75 on-demand). Anchored to your real `total_flos`, so the relative scaling
  is solid.
- A 1.7B checkpoint with optimizer state is ~20 GB; `save_total_limit` 8 ⇒ up to
  ~160 GB of transient checkpoints. The NVMe on `NC40ads_H100_v5` handles this, but
  lower `save_total_limit` per tier if you want to cut blob storage.
- **Storage** of the corpus is negligible (~$0.02/GB-mo; a 500 GB corpus ≈
  $9.80/mo) and there is **no egress** because compute and storage are both in
  westus2.

---

## 6. Cleanup

```powershell
# Scale the cluster to zero (idle scale-down also handles this automatically).
az ml compute update --name gpu-h100-1x --min-instances 0 `
    --workspace-name $WS --resource-group $RG

# Or delete the cluster entirely when the sweep is done.
az ml compute delete --name gpu-h100-1x --yes `
    --workspace-name $WS --resource-group $RG
```

The H100 VM bills only while a job is running; with `min-instances 0` an idle
cluster costs nothing but disk.
