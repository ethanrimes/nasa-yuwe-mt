"""Quick vibe-check translator for the EN<->ES SmolLM2-360M checkpoints.

Self-contained: only needs `transformers` + `torch` and a checkpoint dir.

Examples:
    # Interactive REPL (default = T50k best ckpt, auto-detected direction)
    python translate.py

    # Specific checkpoint via shortcut (one of T10k/T50k/T100k/T500k/T1M, or "best")
    python translate.py --ckpt T1M
    python translate.py --ckpt best     # picks T50k (highest FLORES BLEU)

    # One-off translation from CLI
    python translate.py --ckpt T100k --text "The quick brown fox jumps over the lazy dog."
    python translate.py --ckpt T100k --text "Hola, ¿cómo estás?" --direction es2en

    # Compare all 5 tiers side-by-side on the same sentence
    python translate.py --compare --text "Machine learning is changing the world."

    # Run the built-in sample suite across all checkpoints (no input needed)
    python translate.py --compare --suite

Direction auto-detection is naive (looks for Spanish-only chars / common Spanish
words). Override with --direction en2es | es2en when in doubt.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows so we can print arrows / accents safely.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Project layout: this file lives at the repo root (Q:\experiment-results)
ROOT = Path(__file__).resolve().parent
CKPT_ROOT = ROOT / "checkpoints"

# Best ckpts per tier (chosen by highest avg BLEU on the 100-sample training
# gen-eval subset; see plot_per_tier.png / summarize_metrics.py).
BEST_CKPT = {
    "T10k":  ("T10000",   600),
    "T50k":  ("T50000",   7250),
    "T100k": ("T100000",  10500),
    "T500k": ("T500000",  30000),
    "T1M":   ("T1000000", 36000),
}
# Highest FLORES-200 devtest avg BLEU was T50k (9.54).
DEFAULT_TIER = "T50k"

LANG_NAMES = {"en": "English", "es": "Spanish"}

_SPANISH_HINTS = re.compile(
    r"[ñÑáéíóúÁÉÍÓÚ¿¡üÜ]|\b(que|hola|gracias|por|para|estoy|estás|el|los|las|una|también|"
    r"español|mañana|ayer|señor|señora|cómo|qué|sí|usted|nosotros|tú)\b",
    re.IGNORECASE,
)


def detect_direction(text: str) -> str:
    """Return 'es2en' if the text looks Spanish, else 'en2es'."""
    return "es2en" if _SPANISH_HINTS.search(text) else "en2es"


def make_prompt(text: str, direction: str) -> str:
    if direction == "en2es":
        return f"{LANG_NAMES['en']}: {text}\n{LANG_NAMES['es']}: "
    return f"{LANG_NAMES['es']}: {text}\n{LANG_NAMES['en']}: "


def resolve_ckpt(spec: str) -> Path:
    """Accepts: a tier shortcut (T10k/T50k/...), 'best', or a raw path."""
    if spec.lower() == "best":
        spec = DEFAULT_TIER
    if spec in BEST_CKPT:
        tier_dir, step = BEST_CKPT[spec]
        return CKPT_ROOT / tier_dir / f"checkpoint-{step}"
    p = Path(spec)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_model(ckpt_path: Path, dtype_str: str = "auto"):
    """Returns (model, tokenizer, device, dtype). Heavy import is lazy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[load] {ckpt_path}  ...", end=" ", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only LM → left padding for batched generation.
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    if dtype_str != "auto":
        dtype = getattr(torch, dtype_str)

    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"loaded on {device} ({str(dtype).split('.')[-1]}) in {time.time()-t0:.1f}s")
    return model, tokenizer, device, dtype


def translate(
    text: str,
    direction: str,
    *,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 192,
    num_beams: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    import torch

    prompt = make_prompt(text, direction)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["num_beams"] = 1
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    new_ids = out[0, enc["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    # Stop at first newline — the model is trained to emit one line of target.
    return decoded.split("\n", 1)[0].strip()


SAMPLE_SUITE = [
    ("en2es", "The quick brown fox jumps over the lazy dog."),
    ("en2es", "Machine learning is changing the world."),
    ("en2es", "Could you please pass the salt?"),
    ("en2es", "I have lived in Barcelona for five years."),
    ("en2es", "She doesn't know whether to laugh or cry."),
    ("es2en", "Hola, ¿cómo estás hoy?"),
    ("es2en", "El gato negro saltó sobre la mesa."),
    ("es2en", "No tengo ni idea de lo que estás diciendo."),
    ("es2en", "Mañana iremos al mercado a comprar frutas."),
    ("es2en", "Aunque llueva, saldremos a caminar por el parque."),
]


def run_interactive(tier_spec: str, num_beams: int, do_sample: bool, temperature: float):
    ckpt = resolve_ckpt(tier_spec)
    model, tok, dev, _ = load_model(ckpt)
    direction_override = None
    print()
    print("=" * 72)
    print(f"Vibe-check REPL  |  {tier_spec}  |  beams={num_beams}"
          + (f" sample T={temperature}" if do_sample else ""))
    print("Type a sentence and press Enter. Direction is auto-detected.")
    print("Commands:  :en2es   :es2en   :auto   :ckpt <tier>   :beams <n>   :sample on|off   :q")
    print("=" * 72)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return
        if line.startswith(":ckpt"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    new_ckpt = resolve_ckpt(parts[1])
                    model, tok, dev, _ = load_model(new_ckpt)
                    tier_spec = parts[1]
                except Exception as e:
                    print(f"  ! {e}")
            continue
        if line.startswith(":beams"):
            try:
                num_beams = int(line.split()[1]); print(f"  beams = {num_beams}")
            except Exception:
                print("  usage: :beams <int>")
            continue
        if line.startswith(":sample"):
            parts = line.split()
            do_sample = (len(parts) >= 2 and parts[1].lower() in ("on", "true", "1"))
            print(f"  sample = {do_sample}")
            continue
        if line == ":en2es":
            direction_override = "en2es"; print("  forced direction = en2es"); continue
        if line == ":es2en":
            direction_override = "es2en"; print("  forced direction = es2en"); continue
        if line == ":auto":
            direction_override = None; print("  direction = auto"); continue

        direction = direction_override or detect_direction(line)
        t0 = time.time()
        out = translate(line, direction, model=model, tokenizer=tok, device=dev,
                        num_beams=num_beams, do_sample=do_sample, temperature=temperature)
        dt = time.time() - t0
        arrow = "->" if direction == "en2es" else "<-"
        print(f"  [{direction} {arrow} {dt:.2f}s]  {out}")


def run_oneoff(tier_spec: str, text: str, direction: str | None, num_beams: int,
               do_sample: bool, temperature: float):
    ckpt = resolve_ckpt(tier_spec)
    model, tok, dev, _ = load_model(ckpt)
    direction = direction or detect_direction(text)
    out = translate(text, direction, model=model, tokenizer=tok, device=dev,
                    num_beams=num_beams, do_sample=do_sample, temperature=temperature)
    print(f"\n[{tier_spec}  {direction}]")
    print(f"  src: {text}")
    print(f"  tgt: {out}")


def run_compare(tiers: list[str], texts_with_dirs: list[tuple[str, str]], num_beams: int):
    """Load each checkpoint in turn (saves VRAM) and translate the whole batch."""
    import gc, torch
    results: dict[str, list[str]] = {t: [] for t in tiers}
    for tier in tiers:
        ckpt = resolve_ckpt(tier)
        model, tok, dev, _ = load_model(ckpt)
        for direction, text in texts_with_dirs:
            out = translate(text, direction, model=model, tokenizer=tok, device=dev,
                            num_beams=num_beams)
            results[tier].append(out)
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    for i, (direction, text) in enumerate(texts_with_dirs):
        arrow = "EN->ES" if direction == "en2es" else "ES->EN"
        print(f"\n[{i+1}] {arrow}  {text}")
        for tier in tiers:
            print(f"    {tier:<6}  {results[tier][i]}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_TIER,
                   help="Tier shortcut (T10k/T50k/T100k/T500k/T1M), 'best', or raw checkpoint path.")
    p.add_argument("--text", default=None, help="Sentence to translate (one-off mode).")
    p.add_argument("--direction", choices=["en2es", "es2en"], default=None,
                   help="Translation direction. Auto-detected if omitted.")
    p.add_argument("--beams", type=int, default=4, help="Beam search width (default 4).")
    p.add_argument("--sample", action="store_true", help="Use temperature sampling instead of beam search.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--compare", action="store_true",
                   help="Run the SAME text through every tier (or --tiers list).")
    p.add_argument("--tiers", default="T10k,T50k,T100k,T500k,T1M",
                   help="Comma-separated tiers for --compare mode.")
    p.add_argument("--suite", action="store_true",
                   help="With --compare: run the built-in 10-sentence sample suite.")
    args = p.parse_args()

    if args.compare:
        tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
        if args.suite:
            texts = SAMPLE_SUITE
        elif args.text:
            direction = args.direction or detect_direction(args.text)
            texts = [(direction, args.text)]
        else:
            print("ERROR: --compare needs either --text or --suite.")
            sys.exit(2)
        run_compare(tiers, texts, args.beams)
        return

    if args.text:
        run_oneoff(args.ckpt, args.text, args.direction, args.beams, args.sample, args.temperature)
    else:
        run_interactive(args.ckpt, args.beams, args.sample, args.temperature)


if __name__ == "__main__":
    main()
