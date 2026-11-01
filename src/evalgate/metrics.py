"""Deterministic metrics that do not need a judge.

Keep as much as possible out of the judge. Retrieval quality is measurable
without an LLM, it is cheap, and it is the first thing to check when
groundedness drops -- if the right document was never retrieved, the
generator was never going to get it right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalResult:
    hit_rate: float          # fraction of expected docs that were retrieved
    precision: float         # fraction of retrieved docs that were expected
    rank_of_first_hit: int | None  # 1-indexed; None if nothing relevant found


def retrieval_quality(
    *,
    retrieved_ids: Sequence[str],
    expected_ids: Sequence[str],
) -> RetrievalResult:
    """Score a single retrieval against the documents the answer needs.

    ``hit_rate`` is recall over the expected set. For grounded answering that
    matters more than precision: a missing document guarantees a wrong or
    unsupported answer, whereas an extra one usually just costs tokens.
    """
    expected = set(expected_ids)
    retrieved = list(retrieved_ids)

    if not expected:
        # Nothing was required, so retrieval cannot have failed.
        return RetrievalResult(hit_rate=1.0, precision=1.0, rank_of_first_hit=None)

    hits = [doc for doc in retrieved if doc in expected]
    hit_rate = len(set(hits)) / len(expected)
    precision = len(hits) / len(retrieved) if retrieved else 0.0

    rank = None
    for i, doc in enumerate(retrieved, start=1):
        if doc in expected:
            rank = i
            break

    return RetrievalResult(
        hit_rate=hit_rate,
        precision=precision,
        rank_of_first_hit=rank,
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` in [0, 100].

    Latency belongs in the gate alongside quality. A model change that makes
    answers better and p95 twice as slow is not obviously an improvement, and
    the gate should force that conversation rather than hide it.
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"percentile out of range: {p}")

    ordered = sorted(values)
    if p == 0:
        return ordered[0]
    index = max(1, int(round(p / 100 * len(ordered)))) - 1
    return ordered[index]
