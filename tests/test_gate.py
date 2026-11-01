from evalgate.gate import RunSummary, Thresholds, evaluate_gate


def _good(**over) -> RunSummary:
    base = dict(
        groundedness=0.97,
        faithfulness=0.94,
        retrieval_hit_rate=0.90,
        refusal_accuracy=1.0,
        p95_latency_s=1.2,
        n_cases=50,
    )
    base.update(over)
    return RunSummary(**base)


def test_clean_run_passes():
    result = evaluate_gate(_good(), Thresholds())
    assert result.passed
    assert result.failures == []


def test_below_threshold_fails_with_named_metric():
    result = evaluate_gate(_good(groundedness=0.80), Thresholds())
    assert not result.passed
    assert [f.metric for f in result.failures] == ["groundedness"]
    assert result.failures[0].kind == "threshold"


def test_regression_blocks_even_when_thresholds_pass():
    """The case that ships quietly: still above the bar, but worse than before."""
    baseline = _good(groundedness=0.99)
    current = _good(groundedness=0.96)  # clears the 0.95 threshold

    result = evaluate_gate(current, Thresholds(max_regression=0.02), baseline)

    assert not result.passed
    assert result.failures[0].kind == "regression"
    assert result.failures[0].metric == "groundedness"


def test_small_drop_within_noise_is_allowed():
    baseline = _good(groundedness=0.98)
    current = _good(groundedness=0.97)
    assert evaluate_gate(current, Thresholds(max_regression=0.02), baseline).passed


def test_latency_threshold_is_optional():
    slow = _good(p95_latency_s=9.0)
    assert evaluate_gate(slow, Thresholds()).passed
    assert not evaluate_gate(slow, Thresholds(max_p95_latency_s=3.0)).passed


def test_report_lists_every_failure():
    result = evaluate_gate(
        _good(groundedness=0.10, faithfulness=0.10), Thresholds()
    )
    report = result.report()
    assert "groundedness" in report
    assert "faithfulness" in report
    assert report.startswith("FAIL")


def test_summary_roundtrips_through_json(tmp_path):
    path = tmp_path / "baseline.json"
    original = _good(commit="abc123")
    original.to_json(path)
    assert RunSummary.from_json(path) == original
