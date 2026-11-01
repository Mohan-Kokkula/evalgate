"""Golden dataset loading and validation.

A golden dataset is the contract between the people who own the business
outcome and the people who ship the model. It is deliberately boring: a
JSONL file in version control, reviewed like code, with one row per case
the system is expected to get right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


class DatasetError(ValueError):
    """Raised when a golden dataset row is malformed."""


@dataclass(frozen=True)
class GoldenCase:
    """One evaluation case.

    Attributes:
        id: Stable identifier. Never reuse an id for a different question --
            baselines are keyed on it.
        question: The user-facing input.
        expected_answer: Reference answer. Used by answer-correctness scorers.
        expected_context_ids: Document ids that *must* be retrieved for the
            answer to be groundable. Drives retrieval hit rate.
        must_refuse: True when the correct behaviour is to decline or escalate
            rather than answer. Cases like this are the ones teams forget, and
            they are where confident wrong answers come from.
        tags: Free-form labels (``billing``, ``regression``, ``adversarial``)
            used to slice results.
    """

    id: str
    question: str
    expected_answer: str | None = None
    expected_context_ids: tuple[str, ...] = ()
    must_refuse: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict, *, line_no: int | None = None) -> "GoldenCase":
        where = f" (line {line_no})" if line_no is not None else ""

        case_id = raw.get("id")
        if not case_id or not isinstance(case_id, str):
            raise DatasetError(f"row missing a string 'id'{where}")

        question = raw.get("question")
        if not question or not isinstance(question, str):
            raise DatasetError(f"case {case_id!r} missing a string 'question'{where}")

        must_refuse = bool(raw.get("must_refuse", False))
        expected_answer = raw.get("expected_answer")

        if not must_refuse and not expected_answer:
            raise DatasetError(
                f"case {case_id!r} needs an 'expected_answer' unless "
                f"'must_refuse' is true{where}"
            )

        return cls(
            id=case_id,
            question=question,
            expected_answer=expected_answer,
            expected_context_ids=tuple(raw.get("expected_context_ids", ())),
            must_refuse=must_refuse,
            tags=tuple(raw.get("tags", ())),
        )


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load and validate a JSONL golden dataset.

    Fails loudly on the first bad row. A dataset that silently drops cases is
    worse than no dataset, because the gate still goes green.
    """
    path = Path(path)
    cases: list[GoldenCase] = []
    seen: set[str] = set()

    for line_no, line in enumerate(_nonempty_lines(path), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON on line {line_no}: {exc}") from exc

        case = GoldenCase.from_dict(raw, line_no=line_no)
        if case.id in seen:
            raise DatasetError(f"duplicate case id {case.id!r} on line {line_no}")
        seen.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{path} contains no cases")

    return cases


def filter_by_tag(cases: Sequence[GoldenCase], tag: str) -> list[GoldenCase]:
    return [c for c in cases if tag in c.tags]


def _nonempty_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                yield stripped
