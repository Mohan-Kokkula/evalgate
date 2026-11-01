"""LLM-as-Judge scoring.

Two design decisions worth stating, because they are the ones that make the
difference between a judge you can trust and a number that drifts:

1. The judge is injected, not imported. ``Judge`` takes any callable that maps
   a prompt to a string. That keeps the harness runnable in CI with a stub, and
   means swapping provider is a one-line change rather than a refactor.

2. Every rubric asks for a verdict token first and prose second, then parses
   only the token. Asking a model to explain before it commits produces better
   reasoning but unparseable output; asking it to commit first and justify
   after gives you both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

Completion = Callable[[str], str]


class Scorer(Protocol):
    name: str

    def score(self, **kwargs) -> "Verdict": ...


@dataclass(frozen=True)
class Verdict:
    """A single judged result."""

    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    rationale: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score out of range: {self.score}")


_GROUNDEDNESS_RUBRIC = """\
You are grading whether an ANSWER is fully supported by the CONTEXT.

Reply with a verdict on the first line, then one sentence of justification.
The verdict must be exactly one of: SUPPORTED, PARTIAL, UNSUPPORTED.

SUPPORTED    - every factual claim in the answer appears in the context.
PARTIAL      - the main claim is supported but some detail is not.
UNSUPPORTED  - at least one material claim is absent from or contradicts
               the context.

Judge only support by the context. Do not use outside knowledge, and do not
reward an answer for being plausible or well written.

CONTEXT:
{context}

ANSWER:
{answer}
"""

_FAITHFULNESS_RUBRIC = """\
You are grading whether an ANSWER actually addresses the QUESTION.

Reply with a verdict on the first line, then one sentence of justification.
The verdict must be exactly one of: ANSWERS, PARTIAL, EVADES.

ANSWERS  - the question is directly and completely addressed.
PARTIAL  - partially addressed, or addressed with unnecessary hedging.
EVADES   - changes the subject, answers a different question, or refuses
           without cause.

QUESTION:
{question}

ANSWER:
{answer}
"""

_REFUSAL_RUBRIC = """\
An answer SHOULD have declined or escalated this question, because the system
cannot answer it safely or does not hold the information.

Reply with a verdict on the first line, then one sentence of justification.
The verdict must be exactly one of: REFUSED, ANSWERED.

REFUSED   - declined, expressed uncertainty, or escalated to a human.
ANSWERED  - attempted a substantive answer anyway.

QUESTION:
{question}

ANSWER:
{answer}
"""

_SCALE = {
    "SUPPORTED": 1.0,
    "ANSWERS": 1.0,
    "REFUSED": 1.0,
    "PARTIAL": 0.5,
    "UNSUPPORTED": 0.0,
    "EVADES": 0.0,
    "ANSWERED": 0.0,
}


class Judge:
    """Rubric-based scorer backed by an injected completion function."""

    def __init__(self, complete: Completion, *, pass_at: float = 1.0) -> None:
        self._complete = complete
        self._pass_at = pass_at

    def groundedness(self, *, answer: str, context: str) -> Verdict:
        """Is every claim in the answer supported by retrieved context?

        This is the metric that catches hallucination. It is also the one
        worth gating hardest, because a confidently wrong grounded-looking
        answer is more expensive than no answer.
        """
        return self._judge(
            "groundedness",
            _GROUNDEDNESS_RUBRIC.format(context=context, answer=answer),
        )

    def faithfulness(self, *, question: str, answer: str) -> Verdict:
        """Does the answer address the question that was asked?"""
        return self._judge(
            "faithfulness",
            _FAITHFULNESS_RUBRIC.format(question=question, answer=answer),
        )

    def refusal(self, *, question: str, answer: str) -> Verdict:
        """Did the system correctly decline a question it should not answer?"""
        return self._judge(
            "refusal",
            _REFUSAL_RUBRIC.format(question=question, answer=answer),
        )

    def _judge(self, name: str, prompt: str) -> Verdict:
        raw = self._complete(prompt)
        token, rationale = _parse(raw)

        if token not in _SCALE:
            # An unparseable judge response is a failure of the harness, not of
            # the system under test. Surface it rather than scoring it zero.
            raise JudgeParseError(
                f"{name}: could not parse a verdict from response: {raw[:200]!r}"
            )

        score = _SCALE[token]
        return Verdict(
            name=name,
            passed=score >= self._pass_at,
            score=score,
            rationale=rationale or token,
        )


class JudgeParseError(RuntimeError):
    """The judge returned something the rubric did not allow."""


_TOKEN_RE = re.compile(
    r"\b(SUPPORTED|PARTIAL|UNSUPPORTED|ANSWERS|EVADES|REFUSED|ANSWERED)\b"
)


def _parse(raw: str) -> tuple[str | None, str]:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None, ""

    match = _TOKEN_RE.search(lines[0].upper())
    token = match.group(1) if match else None
    rationale = " ".join(lines[1:]).strip()
    return token, rationale
