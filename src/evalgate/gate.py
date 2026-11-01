"""Release gate: absolute thresholds plus regression against a baseline.

Two checks, because they catch different failures.

Absolute thresholds encode what the business agreed to before rollout -- the
accuracy bar, the tolerated error rate. They answer "is this good enough to
ship at all".

Regression checks compare against the last shipped run. They answer "did this
change make anything worse". A build can clear every absolute threshold and
still be a regression worth blocking, which is exactly the case that ships
quietly and gets discovered in production.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    """Minimum acceptable values, agreed with the people who own the outcome.

    ``max_regression`` is the slack allowed against the baseline before a
    drop counts as a regression. Set it from run-to-run noise measured on a
    fixed system, not from a round number -- too tight and the gate cries
    wolf, too loose and it never fires.
    """

    groundedness: float = 0.95
    faithfulness: float = 0.90
    retrieval_hit_rate: float = 0.85
    refusal_accuracy: float = 0.95
    max_p95_latency_s: float | None = None
    max_regression: float = 0.02


@dataclass
class RunSummary:
    """Aggregate scores for one evaluation run."""

    groundedness: float = 0.0
    faithfulness: float = 0.0
    retrieval_hit_rate: float = 0.0
    refusal_accuracy: float = 0.0
    p95_latency_s: float | None = None
    n_cases: int = 0
    commit: str | None = None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunSummary":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Failure:
    metric: str
    kind: str  # "threshold" | "regression"
    actual: float
    limit: float

    def __str__(self) -> str:
        if self.kind == "threshold":
            return f"{self.metric}: {self.actual:.3f} below threshold {self.limit:.3f}"
        return (
            f"{self.metric}: {self.actual:.3f} is a regression against "
            f"baseline {self.limit:.3f}"
        )


@dataclass
class GateResult:
    passed: bool
    failures: list[Failure] = field(default_factory=list)

    def report(self) -> str:
        if self.passed:
            return "PASS - all thresholds met, no regressions"
        lines = ["FAIL"]
        lines += [f"  - {f}" for f in self.failures]
        return "\n".join(lines)


_QUALITY_METRICS = (
    "groundedness",
    "faithfulness",
    "retrieval_hit_rate",
    "refusal_accuracy",
)


def evaluate_gate(
    summary: RunSummary,
    thresholds: Thresholds,
    baseline: RunSummary | None = None,
) -> GateResult:
    """Apply thresholds and, when a baseline exists, regression checks."""
    failures: list[Failure] = []

    for metric in _QUALITY_METRICS:
        actual = getattr(summary, metric)
        limit = getattr(thresholds, metric)
        if actual < limit:
            failures.append(Failure(metric, "threshold", actual, limit))

    if thresholds.max_p95_latency_s is not None and summary.p95_latency_s is not None:
        if summary.p95_latency_s > thresholds.max_p95_latency_s:
            failures.append(
                Failure(
                    "p95_latency_s",
                    "threshold",
                    summary.p95_latency_s,
                    thresholds.max_p95_latency_s,
                )
            )

    if baseline is not None:
        for metric in _QUALITY_METRICS:
            actual = getattr(summary, metric)
            previous = getattr(baseline, metric)
            if actual < previous - thresholds.max_regression:
                failures.append(Failure(metric, "regression", actual, previous))

    return GateResult(passed=not failures, failures=failures)
