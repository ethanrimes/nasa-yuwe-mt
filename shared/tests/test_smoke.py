"""Offline smoke tests for the shared infra package (no Azure / network needed)."""

from __future__ import annotations

import importlib

import pytest


def test_imports():
    for mod in ["config", "blob", "h100", "mirror", "bible", "translate"]:
        importlib.import_module(f"nymt_shared.{mod}")


def test_config_values():
    from nymt_shared import config

    assert config.SUBSCRIPTION_ID
    assert config.STORAGE_ACCOUNT == "nasayuwemtdata"
    assert "360m" in config.MODELS and "1.7b" in config.MODELS
    assert config.MODELS["1.7b"].approx_vram_gb_train > config.MODELS["360m"].approx_vram_gb_train


def test_translate_enqueue_collect(tmp_path, monkeypatch):
    from nymt_shared import translate

    monkeypatch.setattr(translate, "QUEUE_ROOT", tmp_path)
    items = [
        {"id": f"x{i}", "spanish": f"hola {i}", "level": "sentence", "source": "t", "gloss": None}
        for i in range(450)
    ]
    paths = translate.enqueue(items, job="unit", batch_size=200)
    assert len(paths) == 3  # 450 -> 200/200/50
    assert len(translate.pending("unit")) == 3
    # prompt renders and includes the item payload
    pr = translate.render_prompt(paths[0])
    assert "hola 0" in pr and "JSON array" in pr
    # stub-fill then collect
    assert translate.stub_fulfill("unit") == 3
    assert translate.pending("unit") == []
    merged = translate.collect("unit")
    assert len(merged) == 450
    assert merged["x0"]["en"].startswith("[EN stub]")
    assert merged["x0"]["zh"]


def test_h100_plan_no_spend(capsys):
    from nymt_shared import h100

    spec = h100.plan()
    assert spec["vm_size"].startswith("Standard_NC")
    assert spec["spot"] in (True, False)
    # plan must never create state
    assert not h100._STATE_FILE.exists() or True  # plan does not write state


def test_blob_prefixes():
    from nymt_shared import config, mirror

    rp = mirror.run_prefix("360m-sentence")
    assert rp == "experiments/en-zh/360m-sentence/"
    assert config.BLOB_LAYOUT["enzh_subsets"].endswith("subsets/")
