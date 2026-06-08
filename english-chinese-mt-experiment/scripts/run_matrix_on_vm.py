#!/usr/bin/env python3
"""Run the ablation matrix ON the H100 VM, packing runs concurrently on one GPU.

Invoked remotely by ``11_run_ablations.py`` (never run this locally — it needs the
GPU). Sequence:

  1. Build parquet splits from the synced reconstructed subsets (13_prepare...).
  2. For each distinct model size, build the vocab-extended model once (03 + 04).
  3. Pack the runs into VRAM-bounded waves (nymt_shared.schedule).
  4. Start an NVIDIA MPS (Multi-Process Service) daemon so the concurrently-packed
     processes share the GPU's SMs *spatially* instead of time-slicing via context
     switches -- this is what makes co-locating runs on one H100 actually faster
     rather than just VRAM-efficient. Each process is given an SM budget
     (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) weighted by its model size.
  5. For each wave, launch the runs concurrently (one process each, distinct
     output_root), capping fragmentation with expandable_segments.
  6. A background thread mirrors models/runs/ -> Azure Blob every --mirror-interval
     seconds, so an early teardown loses at most one interval of work.
  7. A --max-budget-hours tripwire force-kills everything and returns, guaranteeing
     we never blow the GPU budget.

Snapshots + metrics for every run land in Blob at experiments/en-zh/<run_id>/...
which is exactly what a later "resume the promising snapshot" workflow re-pulls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]      # english-chinese-mt-experiment
sys.path.insert(0, str(REPO / "src"))

from nymt_shared import blob, config, schedule  # noqa: E402

ABLATIONS = REPO / "configs" / "ablations.yaml"
RUNS_ROOT = REPO / "models" / "runs"
SPLITS_MANIFEST = REPO / "data" / "splits" / "splits_manifest.json"
PY = sys.executable

# ---- Optional NLLB es<->nasa fine-tunes packed onto the same H100 ------------ #
# These fill spare VRAM during the En<->Zh matrix (max GPU utilization, one shared
# teardown). Wired in robustly: if the `snmt` package isn't installed/shipped the
# runner silently runs the En<->Zh matrix exactly as before.
NLLB_RUNS_ROOT = REPO / "models" / "nllb_runs"      # mirrored -> experiments/es-nasa/
SNMT_DIR = config.ES_NASA_DIR                        # spanish-nasa-mt-experiment/
SNMT_PREPARE = SNMT_DIR / "scripts" / "prepare_data.py"
NLLB_SPLITS_DIR = SNMT_DIR / "data" / "splits"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


# --------------------------------------------------------------------------- #
# NVIDIA MPS — spatial GPU sharing for concurrently-packed runs
# --------------------------------------------------------------------------- #
MPS_PIPE = "/tmp/nymt-mps/pipe"
MPS_LOG = "/tmp/nymt-mps/log"


def start_mps() -> dict | None:
    """Best-effort start of the CUDA MPS control daemon.

    With MPS, the multiple training processes we pack onto one H100 submit work to a
    single shared CUDA context, so the GPU schedules their kernels *spatially* across
    SMs instead of expensively context-switching between separate contexts. This is
    the difference between packing being merely VRAM-efficient and it being genuinely
    faster. Returns the env vars to inject into each run, or None if MPS is
    unavailable (we then fall back to plain concurrent processes — still correct,
    just less efficient on the compute-bound 1.7B wave).
    """
    import shutil

    if shutil.which("nvidia-cuda-mps-control") is None:
        print("[mps] nvidia-cuda-mps-control not found; running without MPS.", flush=True)
        return None
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_MPS_PIPE_DIRECTORY"] = MPS_PIPE
    env["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG
    try:
        # Daemonizes; returns immediately. Idempotent enough — a second start is a no-op.
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True,
                       text=True, timeout=30)
        print(f"[mps] control daemon started (pipe={MPS_PIPE}).", flush=True)
        return {
            "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE,
            "CUDA_MPS_LOG_DIRECTORY": MPS_LOG,
        }
    except Exception as e:  # pragma: no cover - VM-only path
        print(f"[mps] WARN: could not start MPS ({e}); running without it.", flush=True)
        return None


def stop_mps() -> None:
    """Best-effort shutdown of the MPS daemon (so teardown leaves nothing running)."""
    import shutil

    if shutil.which("nvidia-cuda-mps-control") is None:
        return
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True,
                       timeout=30, check=False)
        print("[mps] control daemon stopped.", flush=True)
    except Exception as e:  # pragma: no cover - VM-only path
        print(f"[mps] WARN: stop failed ({e}).", flush=True)


def sm_budget(wave) -> dict[str, int]:
    """Per-run CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for one wave.

    Partition the GPU's SMs across the wave's processes weighted by VRAM footprint
    (a stand-in for compute demand): a 34 GB 1.7B run gets a bigger SM slice than a
    10 GB 360M run sharing the same wave. A single-run wave gets the whole GPU (no
    cap). Floored at 10% so nothing is starved; the cap is advisory so a small sum
    is fine and a slightly-over sum just means soft oversubscription.
    """
    if len(wave.runs) <= 1:
        return {wave.runs[0].run_id: 100} if wave.runs else {}
    total = sum(r.vram_gb for r in wave.runs) or 1.0
    return {r.run_id: max(10, int(round(100.0 * r.vram_gb / total))) for r in wave.runs}


def prepare_splits() -> dict:
    sh([PY, "scripts/13_prepare_ablation_subsets.py"], cwd=str(REPO))
    return json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))


def extend_model(model_config: str, source_parquet: str, zh_vocab: int) -> None:
    """Build merged tokenizer + extended model for one size from the proxy zh text."""
    sh([PY, "scripts/03_train_tokenizer_extension.py", "--model-config", model_config,
        "--source-parquet", source_parquet, "--vocab-size", str(zh_vocab)], cwd=str(REPO))
    sh([PY, "scripts/04_extend_model.py", "--model-config", model_config], cwd=str(REPO))


def mirror_loop(stop: threading.Event, interval: int) -> None:
    """Mirror both the En<->Zh runs and (if present) the NLLB runs to Blob."""
    targets = []
    if RUNS_ROOT.exists():
        targets.append((RUNS_ROOT, config.BLOB_LAYOUT["exp_en_zh"]))
    if NLLB_RUNS_ROOT.exists():
        targets.append((NLLB_RUNS_ROOT, config.BLOB_LAYOUT["exp_es_nasa"]))
    while not stop.is_set():
        # Re-discover roots each tick: under --only-nllb the en-zh root never
        # appears, and the NLLB root may be created slightly after the loop starts.
        live = []
        if RUNS_ROOT.exists():
            live.append((RUNS_ROOT, config.BLOB_LAYOUT["exp_en_zh"]))
        if NLLB_RUNS_ROOT.exists():
            live.append((NLLB_RUNS_ROOT, config.BLOB_LAYOUT["exp_es_nasa"]))
        for root, prefix in (live or targets):
            try:
                n = blob.mirror_dir(root, prefix, only_new=True, verbose=False)
                print(f"[mirror] flushed {n} file(s) -> {prefix}", flush=True)
            except Exception as e:
                print(f"[mirror] WARN: {e}", flush=True)
        stop.wait(interval)


def prepare_nllb(enabled: bool) -> tuple[list, dict]:
    """Prepare NLLB data + run list, robustly. Returns (schedule.Runs, launch_map).

    On any failure (package absent, blob/auth hiccup, etc.) this logs a warning and
    returns empties so the En<->Zh matrix proceeds untouched — the NLLB packing is
    strictly opportunistic.
    """
    if not enabled:
        return [], {}
    try:
        sys.path.insert(0, str(SNMT_DIR / "src"))
        from snmt import runs as nllb_runs  # noqa: PLC0415

        # Build BOTH subset split dirs on the VM (downloads the jsonl from Blob;
        # the en-zh source copy isn't rsynced). Per-subset dirs land under
        # <NLLB_SPLITS_DIR>/<subset>/ with a top-level subsets_index.json.
        sh([PY, str(SNMT_PREPARE), "--out-dir", str(NLLB_SPLITS_DIR), "--subset", "all"],
           cwd=str(SNMT_DIR))
        index = json.loads(
            (NLLB_SPLITS_DIR / "subsets_index.json").read_text(encoding="utf-8")
        )
        train_pairs_by_subset = {
            name: int(info["train_pairs"]) for name, info in index["subsets"].items()
        }

        cfg = nllb_runs.load_config()
        runs = nllb_runs.load_nllb_runs(train_pairs_by_subset, cfg)
        launch = nllb_runs.launch_map(
            nllb_runs.run_specs(cfg),
            splits_root=NLLB_SPLITS_DIR,
            output_root=NLLB_RUNS_ROOT,
            python=PY,
        )
        pairs_desc = ", ".join(f"{k}={v:,}" for k, v in sorted(train_pairs_by_subset.items()))
        print(f"[nllb] prepared {len(runs)} run(s); train_pairs[{pairs_desc}]", flush=True)
        return runs, launch
    except Exception as e:
        print(f"[nllb] WARN: NLLB co-runs disabled ({e}); running En<->Zh only.", flush=True)
        return [], {}


def launch_nllb_run(argv: list[str], extra_env: dict | None = None) -> subprocess.Popen:
    """Launch one NLLB fine-tune (one process), mirroring launch_run's env/log setup."""
    run_id = argv[argv.index("--run-id") + 1]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    pct = env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "-")
    print("+", " ".join(argv), f"(nllb run_id={run_id} sm%={pct})", flush=True)
    NLLB_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    log = NLLB_RUNS_ROOT / f"{run_id}.log"
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen(argv, cwd=str(SNMT_DIR), env=env, stdout=fh, stderr=subprocess.STDOUT)


def launch_run(run_cfg: dict, scale: int, extra_env: dict | None = None) -> subprocess.Popen:
    out_root = f"models/runs/{run_cfg['id']}"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    # Weights & Biases on the headless VM. configs/training.yaml enables wandb, so HF
    # Trainer calls wandb.init() at on_train_begin. If a WANDB_API_KEY is present in the
    # environment (propagated from the orchestrator's run_all.sh) we run wandb ONLINE and
    # pin the project; otherwise we hard-disable it via a real boolean override so the
    # run can't die with "UsageError: No API key configured". AML behaviour is untouched.
    # MLflow defaults to a *shared* local store rooted at the common CWD. When two runs are
    # packed onto the GPU via MPS they start in the same instant and race to create the same
    # MLflow experiment -> sqlite "OperationalError: database is locked", killing one run at
    # step 0 (observed for 1.7b-sentvocab). MLflow is only used in AML ("# used in AML
    # automatically"); on the VM we already have wandb (online) + per-run TensorBoard dirs,
    # which have no shared state. So hard-disable MLflow on the VM path to make packing safe.
    overrides = [f"output_root={out_root}", "tracking.mlflow.enabled=false"]
    if env.get("WANDB_API_KEY", "").strip():
        env["WANDB_MODE"] = "online"
        env.setdefault("WANDB_PROJECT", "english-chinese-mt")
        env.setdefault("WANDB_SILENT", "true")
        env.pop("WANDB_DISABLED", None)
    else:
        env["WANDB_DISABLED"] = "true"
        env["WANDB_MODE"] = "offline"
        overrides.append("tracking.wandb.enabled=false")
    cmd = [PY, "scripts/06_train.py", "--scale", str(scale),
           "--model-config", run_cfg["model_config"]]
    for o in overrides:
        cmd += ["--override", o]
    cmd.append("--yes")
    pct = env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "-")
    print("+", " ".join(cmd), f"(run_id={run_cfg['id']} sm%={pct})", flush=True)
    log = RUNS_ROOT / f"{run_cfg['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--max-budget-hours", type=float, default=config.MAX_BUDGET_HOURS)
    ap.add_argument("--mirror-interval", type=int, default=300)
    ap.add_argument("--no-nllb", action="store_true",
                    help="do NOT pack the es<->nasa NLLB fine-tunes onto the GPU")
    ap.add_argument("--only-nllb", action="store_true",
                    help="run ONLY the es<->nasa NLLB fine-tunes (skip the En<->Zh "
                         "matrix entirely). Used to re-run the NLLB axis without "
                         "touching the already-complete En<->Zh runs.")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated en-zh run id(s) to run; others skipped. "
                         "Re-runs a single failed matrix cell without redoing the rest.")
    args = ap.parse_args()

    if args.only_nllb and args.no_nllb:
        raise SystemExit("--only-nllb and --no-nllb are mutually exclusive")

    only_nllb = args.only_nllb
    cfg = yaml.safe_load(ABLATIONS.read_text(encoding="utf-8"))

    if only_nllb:
        # NLLB-only path: skip all En<->Zh prep (splits, vocab-extend, runs).
        print("[matrix] --only-nllb: skipping En<->Zh matrix; NLLB es<->nasa only.",
              flush=True)
        run_cfgs: list = []
        runs = []
        by_id = {}
        scale_for = {}
    else:
        run_cfgs = cfg["runs"]
        if args.only:
            wanted = {s.strip() for s in args.only.split(",") if s.strip()}
            run_cfgs = [r for r in run_cfgs if r["id"] in wanted]
            if not run_cfgs:
                raise SystemExit(f"--only {args.only!r} matched no run ids in {[r['id'] for r in cfg['runs']]}")
            print(f"[matrix] --only filter -> {[r['id'] for r in run_cfgs]}", flush=True)

        manifest = prepare_splits()
        scale_for = {
            "sentence_only": manifest["scale_sentence_only"],
            "sentence_plus_vocab": manifest["scale_sentence_plus_vocab"],
        }

        # Extend each distinct model size once, learning Chinese BPE from the proxy text.
        full_parquet = f"data/splits/{manifest['scale_filename_sentence_plus_vocab']}"
        zh_vocab = int(cfg.get("zh_vocab_size", 8000))
        for mc in sorted({r["model_config"] for r in run_cfgs}):
            extend_model(mc, full_parquet, zh_vocab)

        # Build the En<->Zh schedule.
        runs = [
            schedule.Run(
                run_id=r["id"], model_key=r["model_key"], subset=r["subset"],
                pairs=int(scale_for[r["subset"]]),
                epochs=float(cfg.get("epochs", 3)),
                effective_batch=int(cfg.get("effective_batch", 64)),
            )
            for r in run_cfgs
        ]
        by_id = {r["id"]: r for r in run_cfgs}

    # Opportunistically pack the NLLB es<->nasa fine-tunes into the SAME waves to
    # fill spare H100 VRAM (or run them alone under --only-nllb). Robust: if snmt is
    # absent these are empty and the En<->Zh matrix runs exactly as before.
    nllb_enabled = not args.no_nllb  # always on under --only-nllb (mutual-excl above)
    nllb_runs, nllb_launch = prepare_nllb(enabled=nllb_enabled)
    runs = runs + nllb_runs

    if not runs:
        raise SystemExit("[matrix] no runs to execute (en-zh skipped and no NLLB runs prepared)")

    waves = schedule.pack(runs, usable_vram_gb=float(cfg.get("usable_vram_gb", 72)),
                          max_concurrent=int(cfg.get("max_concurrent", 6)))
    print(f"[matrix] {len(runs)} runs ({len(by_id)} en-zh + {len(nllb_runs)} nllb) "
          f"-> {len(waves)} wave(s)", flush=True)

    stop = threading.Event()
    mt = threading.Thread(target=mirror_loop, args=(stop, args.mirror_interval), daemon=True)
    mt.start()

    mps_env = start_mps()  # None if MPS unavailable -> plain concurrent processes

    t0 = time.time()
    budget_s = args.max_budget_hours * 3600.0
    aborted = False
    try:
        for wi, wave in enumerate(waves):
            budgets = sm_budget(wave) if mps_env else {}
            print(f"[matrix] wave {wi+1}/{len(waves)} "
                  f"({wave.vram_gb:.0f} GB): {[r.run_id for r in wave.runs]} "
                  f"sm%={budgets or 'n/a'}", flush=True)
            procs = []
            for r in wave.runs:
                run_env = dict(mps_env) if mps_env else None
                if mps_env and r.run_id in budgets:
                    run_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = budgets[r.run_id]
                if r.run_id in by_id:  # En<->Zh causal-LM run (06_train.py)
                    p = launch_run(by_id[r.run_id], scale_for[r.subset], run_env)
                else:                  # NLLB es<->nasa seq2seq run (train_nllb.py)
                    p = launch_nllb_run(nllb_launch[r.run_id], run_env)
                procs.append((r.run_id, p))
            while any(p.poll() is None for _, p in procs):
                if time.time() - t0 > budget_s:
                    print("[matrix] BUDGET EXCEEDED — killing wave", flush=True)
                    for _, p in procs:
                        if p.poll() is None:
                            p.terminate()
                    aborted = True
                    break
                time.sleep(15)
            for rid, p in procs:
                rc = p.wait()
                print(f"[matrix] run {rid} exit={rc}", flush=True)
            if aborted:
                break
    finally:
        print("[matrix] final mirror flush ...", flush=True)
        try:
            if RUNS_ROOT.exists():
                blob.mirror_dir(RUNS_ROOT, config.BLOB_LAYOUT["exp_en_zh"], only_new=True, verbose=False)
            if NLLB_RUNS_ROOT.exists():
                blob.mirror_dir(NLLB_RUNS_ROOT, config.BLOB_LAYOUT["exp_es_nasa"], only_new=True, verbose=False)
        finally:
            stop.set()
            stop_mps()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (RUNS_ROOT / "MATRIX_DONE.json").write_text(
        json.dumps({"aborted": aborted, "elapsed_h": round((time.time() - t0) / 3600.0, 3)}, indent=2),
        encoding="utf-8",
    )
    print(f"[matrix] done aborted={aborted} elapsed_h={(time.time()-t0)/3600.0:.2f}", flush=True)
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
