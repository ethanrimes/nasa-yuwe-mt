"""NLLB language-token handling for the Nasa-Yuwe fine-tune.

NLLB-200 is a many-to-many model whose tokenizer prefixes each sequence with a
language code token (e.g. ``spa_Latn``) and terminates it with EOS (id 2). The
decoder is *forced* to start by emitting the target language code, which is how a
single model translates between any of its 200 languages.

Nasa-Yuwe (Páez, ISO 639-3 ``pbb``) is **not** one of those 200 languages, so we:

  1. add ``pbb_Latn`` as a new special token,
  2. resize the model's token embeddings to match, and
  3. (optionally) warm-start the new code's embedding from a related language so
     it isn't pure noise.

We build ``input_ids`` / ``labels`` *manually* rather than relying on the
``NllbTokenizer`` source/target-language internals, because those internals shift
between ``transformers`` versions. The manual recipe (lang-code + tokens + EOS for
both sides) is stable and matches the model's pre-training scheme. Pairing it with
``DataCollatorForSeq2Seq`` — which prepends ``decoder_start_token_id`` (EOS) to form
``decoder_input_ids`` — reproduces NLLB's "decoder emits target lang code first"
behaviour exactly.

Most of this module is pure (no torch / no transformers) so it is unit-testable;
the few functions that touch a live tokenizer/model take them as arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical tags used throughout the package.
NASA_LANG = "pbb_Latn"
SPANISH_LANG = "spa_Latn"

# NLLB / M2M100 end-of-sequence id. Constant across the NLLB-200 checkpoints.
NLLB_EOS_ID = 2

# When warm-starting the new Nasa-Yuwe language embedding, copy from this NLLB
# language. Spanish is the natural donor: Nasa-Yuwe's parallel data is Spanish and
# shares the Latin script + much religious-register vocabulary/orthographic shape.
WARMSTART_DONOR_LANG = "spa_Latn"


@dataclass(frozen=True)
class Direction:
    """One translation direction for a parallel pair."""

    src_lang: str
    tgt_lang: str
    # Keys into the dataset row for the source / target text of this direction.
    src_field: str
    tgt_field: str


# The two directions we train bidirectionally. Field names match the parallel
# jsonl schema ({"spanish": ..., "nasa_yuwe": ...}).
DIRECTIONS: tuple[Direction, Direction] = (
    Direction(SPANISH_LANG, NASA_LANG, "spanish", "nasa_yuwe"),
    Direction(NASA_LANG, SPANISH_LANG, "nasa_yuwe", "spanish"),
)


def both_directions() -> tuple[Direction, Direction]:
    """Return the (spa->pbb, pbb->spa) direction pair."""
    return DIRECTIONS


def ensure_nasa_lang_token(tokenizer, model=None) -> int:
    """Make ``pbb_Latn`` a real token; resize + warm-start the model if given.

    Idempotent: if the token already exists (re-loading a checkpoint that was
    already extended) nothing is added. Returns the token id of ``pbb_Latn``.

    When ``model`` is provided and the vocabulary grew, the embedding matrix is
    resized and the new Nasa-Yuwe row is initialised from the Spanish language
    embedding (a far better start than the random init ``resize_token_embeddings``
    would otherwise leave it with).
    """
    added = 0
    if NASA_LANG not in tokenizer.get_vocab():
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [NASA_LANG]}
        )

    new_id = tokenizer.convert_tokens_to_ids(NASA_LANG)

    if model is not None and added > 0:
        donor_id = tokenizer.convert_tokens_to_ids(WARMSTART_DONOR_LANG)
        model.resize_token_embeddings(len(tokenizer))
        if donor_id is not None and donor_id >= 0:
            _warmstart_embedding(model, new_id=new_id, donor_id=donor_id)
    return new_id


def _warmstart_embedding(model, *, new_id: int, donor_id: int) -> None:
    """Copy the donor language's embedding (and tied LM-head row) into the new id."""
    import torch  # local import: only needed on the GPU/train path

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[new_id].copy_(emb[donor_id])
        head = model.get_output_embeddings()
        if head is not None and head.weight.data_ptr() != emb.data_ptr():
            head.weight[new_id].copy_(head.weight[donor_id])


def lang_token_id(tokenizer, lang: str) -> int:
    """Resolve a language code to its token id (after ``ensure_nasa_lang_token``)."""
    tid = tokenizer.convert_tokens_to_ids(lang)
    if tid is None or tid < 0:
        raise KeyError(f"language token {lang!r} is not in the tokenizer vocabulary")
    return tid


def build_input_ids(token_ids: list[int], lang_id: int, eos_id: int = NLLB_EOS_ID) -> list[int]:
    """``[lang_id] + token_ids + [eos]`` — NLLB's per-sequence framing.

    Pure list logic so it can be unit-tested without a tokenizer. ``token_ids`` are
    the *content* tokens (no special tokens added by the tokenizer).
    """
    return [lang_id, *token_ids, eos_id]


def encode_example(
    tokenizer,
    *,
    src_text: str,
    tgt_text: str,
    src_lang: str,
    tgt_lang: str,
    max_source_len: int,
    max_target_len: int,
    eos_id: int = NLLB_EOS_ID,
) -> dict[str, list[int]]:
    """Tokenize one directed pair into NLLB-framed ``input_ids`` / ``labels``.

    The collator (``DataCollatorForSeq2Seq``) later pads these, converts label pad
    to ``-100``, and builds ``decoder_input_ids`` by shifting ``labels`` right and
    prepending ``decoder_start_token_id`` (EOS), so the model is trained to emit the
    target language code as its first generated token — exactly NLLB's scheme.
    """
    src_ids = tokenizer(
        src_text, add_special_tokens=False, truncation=True,
        max_length=max(1, max_source_len - 2),
    )["input_ids"]
    tgt_ids = tokenizer(
        tgt_text, add_special_tokens=False, truncation=True,
        max_length=max(1, max_target_len - 2),
    )["input_ids"]

    input_ids = build_input_ids(src_ids, lang_token_id(tokenizer, src_lang), eos_id)
    labels = build_input_ids(tgt_ids, lang_token_id(tokenizer, tgt_lang), eos_id)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
