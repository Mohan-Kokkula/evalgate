"""Run a golden dataset against a system under test and summarise the result."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .dataset import GoldenCase
from .judge import Judge
from .gate import RunSummary
from .metrics import mean, percentile, retrieval_quality


class SystemUnderTest(Protocol):
    """Whatever you are evaluating: a RAG chain, an agent, a bare model."""

    def __call__(self, question: str) -> "Response": ...


@dataclass
class Response:
    """What the system returned for one question."""

    answer: str
    context: str = ""
    retrieved_ids: Sequence[str] = ()


@dataclass
class CaseResult:
    case_id: str
    groundedness: float
    faithfulness: float
    retrieval_hit_rate: float
    refusal_correct: float | None
    latency_s: float
    notes: str = ""


def run_eval(
    cases: Sequence[GoldenCase],
    system: SystemUnderTest,
    judge: Judge,
    *,
    commit: str | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> tuple[RunSummary, list[CaseResult]]:
    """Evaluate every case and return the aggregate plus per-case detail.

    Per-case detail matters as much as the summary. When the gate fails, the
    first question is always "which cases", and a mean on its own cannot
    answer it.
    """
    results: list[CaseResult] = []

    for case in cases:
        started = time.perf_counter()
        response = system(case.question)
        latency = time.perf_counter() - started

        retrieval = retrieval_quality(
            retrieved_ids=response.retrieved_ids,
            expected_ids=case.expected_context_ids,
        )

        if case.must_refuse:
            verdict = judge.refusal(question=case.question, answer=response.answer)
            result = CaseResult(
                case_id=case.id,
                # A correct refusal is trivially grounded and on-topic; scoring
                # it against the answer rubrics would punish correct behaviour.
                groundedness=1.0 if verdict.passed else 0.0,
                faithfulness=1.0 if verdict.passed else 0.0,
                retrieval_hit_rate=retrieval.hit_rate,
                refusal_correct=1.0 if verdict.passed else 0.0,
                latency_s=latency,
                notes=verdict.rationale,
            )
        else:
            grounded = judge.groundedness(
                answer=response.answer, context=response.context
            )
            faithful = judge.faithfulness(
                question=case.question, answer=response.answer
            )
            result = CaseResult(
                case_id=case.id,
                groundedness=grounded.score,
                faithfulness=faithful.score,
                retrieval_hit_rate=retrieval.hit_rate,
                refusal_correct=None,
                latency_s=latency,
                notes="; ".join(filter(None, [grounded.rationale, faithful.rationale])),
            )

        results.append(result)
        if on_case:
            on_case(result)

    refusals = [r.refusal_correct for r in results if r.refusal_correct is not None]

    summary = RunSummary(
        groundedness=mean([r.groundedness for r in results]),
        faithfulness=mean([r.faithfulness for r in results]),
        retrieval_hit_rate=mean([r.retrieval_hit_rate for r in results]),
        # No refusal cases means nothing to get wrong, so score it clean rather
        # than zero -- otherwise the gate blocks on a metric that was never run.
        refusal_accuracy=mean(refusals) if refusals else 1.0,
        p95_latency_s=percentile([r.latency_s for r in results], 95),
        n_cases=len(results),
        commit=commit,
    )

    return summary, results
