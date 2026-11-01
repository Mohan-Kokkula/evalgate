import pytest

from evalgate.dataset import DatasetError, GoldenCase, load_golden
from evalgate.judge import Judge, JudgeParseError
from evalgate.metrics import percentile, retrieval_quality


# --- retrieval -------------------------------------------------------------


def test_perfect_retrieval():
    r = retrieval_quality(retrieved_ids=["a", "b"], expected_ids=["a", "b"])
    assert r.hit_rate == 1.0
    assert r.precision == 1.0
    assert r.rank_of_first_hit == 1


def test_missing_document_lowers_hit_rate():
    r = retrieval_quality(retrieved_ids=["a", "x"], expected_ids=["a", "b"])
    assert r.hit_rate == 0.5
    assert r.precision == 0.5


def test_noise_lowers_precision_but_not_recall():
    r = retrieval_quality(retrieved_ids=["x", "y", "a"], expected_ids=["a"])
    assert r.hit_rate == 1.0
    assert r.precision == pytest.approx(1 / 3)
    assert r.rank_of_first_hit == 3


def test_no_expected_docs_cannot_fail():
    r = retrieval_quality(retrieved_ids=[], expected_ids=[])
    assert r.hit_rate == 1.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0], 101)


# --- dataset ---------------------------------------------------------------


def test_answerable_case_requires_expected_answer():
    with pytest.raises(DatasetError, match="expected_answer"):
        GoldenCase.from_dict({"id": "c1", "question": "what?"})


def test_refusal_case_needs_no_expected_answer():
    case = GoldenCase.from_dict(
        {"id": "c1", "question": "what is my balance?", "must_refuse": True}
    )
    assert case.must_refuse


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"a","question":"q","expected_answer":"x"}\n'
        '{"id":"a","question":"q2","expected_answer":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate"):
        load_golden(path)


# --- judge -----------------------------------------------------------------


def test_judge_parses_verdict_and_rationale():
    judge = Judge(lambda _: "SUPPORTED\nEvery claim appears in the context.")
    v = judge.groundedness(answer="a", context="c")
    assert v.passed
    assert v.score == 1.0
    assert "context" in v.rationale


def test_partial_does_not_pass_a_strict_gate():
    judge = Judge(lambda _: "PARTIAL\nThe date is not in the context.")
    assert not judge.groundedness(answer="a", context="c").passed


def test_unparseable_judge_output_raises_rather_than_scoring_zero():
    """A broken judge is a harness bug. Silently scoring zero would fail the
    build for the wrong reason and send someone debugging the model."""
    judge = Judge(lambda _: "I think it looks fine overall!")
    with pytest.raises(JudgeParseError):
        judge.groundedness(answer="a", context="c")
