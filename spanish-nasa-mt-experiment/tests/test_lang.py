"""Pure-logic tests for snmt.lang — no torch, no transformers."""

from __future__ import annotations

import pytest

from snmt import lang as L


def test_directions_are_spanish_and_nasa_bidirectional():
    assert len(L.DIRECTIONS) == 2
    d0, d1 = L.DIRECTIONS
    assert (d0.src_lang, d0.tgt_lang) == (L.SPANISH_LANG, L.NASA_LANG)
    assert (d1.src_lang, d1.tgt_lang) == (L.NASA_LANG, L.SPANISH_LANG)
    assert (d0.src_field, d0.tgt_field) == ("spanish", "nasa_yuwe")
    assert (d1.src_field, d1.tgt_field) == ("nasa_yuwe", "spanish")


def test_build_input_ids_frames_with_lang_and_eos():
    out = L.build_input_ids([10, 11, 12], lang_id=99, eos_id=2)
    assert out == [99, 10, 11, 12, 2]


def test_build_input_ids_default_eos_is_nllb_eos():
    out = L.build_input_ids([5], lang_id=7)
    assert out[0] == 7 and out[-1] == L.NLLB_EOS_ID


class _FakeTok:
    """Minimal tokenizer stub: vocab lookups + content tokenization."""

    def __init__(self, vocab):
        self._vocab = dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, -1)

    def add_special_tokens(self, d):
        toks = d["additional_special_tokens"]
        n = 0
        for t in toks:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
                n += 1
        return n

    def __call__(self, text, **kw):
        # one id per whitespace word, offset so they don't collide with lang ids
        return {"input_ids": [1000 + len(w) for w in text.split()]}


def test_ensure_nasa_lang_token_is_idempotent():
    tok = _FakeTok({"spa_Latn": 0, "</s>": 2})
    first = L.ensure_nasa_lang_token(tok)
    assert tok.convert_tokens_to_ids(L.NASA_LANG) == first
    second = L.ensure_nasa_lang_token(tok)
    assert first == second  # no duplicate add


def test_lang_token_id_raises_for_unknown():
    tok = _FakeTok({"spa_Latn": 0})
    with pytest.raises(KeyError):
        L.lang_token_id(tok, "xxx_Yyyy")


def test_encode_example_round_trips_framing():
    tok = _FakeTok({"spa_Latn": 0, "pbb_Latn": 1, "</s>": 2})
    enc = L.encode_example(
        tok,
        src_text="hola mundo",
        tgt_text="aa",
        src_lang="spa_Latn",
        tgt_lang="pbb_Latn",
        max_source_len=64,
        max_target_len=64,
    )
    assert enc["input_ids"][0] == 0  # src lang first
    assert enc["input_ids"][-1] == L.NLLB_EOS_ID
    assert enc["labels"][0] == 1  # tgt lang first
    assert enc["labels"][-1] == L.NLLB_EOS_ID
    assert enc["attention_mask"] == [1] * len(enc["input_ids"])
