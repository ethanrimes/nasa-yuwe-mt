import json
import shutil

from nymt_shared import llm_judge


def _clean(job: str):
    d = llm_judge._job_dir(job)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_score_roundtrip():
    job = "_pytest_judge_score"
    _clean(job)
    items = [
        {"id": "s1", "direction": "en2zh", "source": "A.", "reference": "甲。", "hypothesis": "甲。"},
        {"id": "s2", "direction": "zh2en", "source": "乙。", "reference": "B.", "hypothesis": "B?"},
    ]
    llm_judge.enqueue_score(items, job=job, batch_size=1)
    assert len(llm_judge.pending(job)) == 2
    prompt = llm_judge.render_prompt(llm_judge.input_files(job)[0])
    assert "adequacy" in prompt and "overall" in prompt
    assert llm_judge.stub_fulfill(job) == 2
    assert llm_judge.pending(job) == []
    agg = llm_judge.aggregate(job)
    assert agg["mode"] == "score" and agg["n"] == 2
    assert set(agg["means"]) == {"adequacy", "fluency", "terminology", "overall"}
    _clean(job)


def test_pairwise_roundtrip():
    job = "_pytest_judge_pw"
    _clean(job)
    items = [{"id": "p1", "direction": "en2zh", "source": "Hi.",
              "reference": "你好。", "hyp_a": "你好。", "hyp_b": "您好。"}]
    llm_judge.enqueue_pairwise(items, job=job, batch_size=10)
    prompt = llm_judge.render_prompt(llm_judge.input_files(job)[0])
    assert "winner" in prompt
    llm_judge.stub_fulfill(job)
    agg = llm_judge.aggregate(job)
    assert agg["mode"] == "pairwise" and agg["n"] == 1
    assert "a_win_rate" in agg and "b_win_rate" in agg
    _clean(job)
