#!/usr/bin/env python3
"""Run the gate against a system under test.

Wire ``build_system()`` and ``build_judge()`` to your own stack. Everything
below that line is provider-agnostic on purpose: the harness should outlive
whichever model you are using this quarter.

    python examples/run_eval.py \\
        --golden datasets/golden.example.jsonl \\
        --baseline baselines/baseline.example.json \\
        --out results/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalgate import (  # noqa: E402
    Judge,
    Response,
    RunSummary,
    Thresholds,
    evaluate_gate,
    load_golden,
    run_eval,
)


# --- wire these two to your stack ------------------------------------------


def build_system():
    """Return a callable: question -> Response.

    Replace the body with a call into your RAG chain or agent. Return the
    retrieved document ids alongside the answer -- without them you cannot
    tell a generation failure from a retrieval failure, and those have
    completely different fixes.
    """

    def system(question: str) -> Response:
        return Response(
            answer="Open Billing > Payment methods and choose Replace card.",
            context="To change a card: Billing > Payment methods > Replace card.",
            retrieved_ids=["kb/billing/payment-methods"],
        )

    return system


def build_judge() -> Judge:
    """Return a Judge backed by your model provider.

    Use a different model from the one under test where you can. A model
    grading its own output is measurably more generous to itself, and the
    whole point of the gate is an opinion you did not write.
    """

    def complete(prompt: str) -> str:
        # from openai import OpenAI
        # return OpenAI().chat.completions.create(
        #     model="gpt-4o", temperature=0,
        #     messages=[{"role": "user", "content": prompt}],
        # ).choices[0].message.content
        return "SUPPORTED\nStub judge: replace build_judge() with a real model."

    return Judge(complete)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True, help="JSONL golden dataset")
    parser.add_argument("--baseline", help="previous run summary, for regression checks")
    parser.add_argument("--out", help="where to write this run's summary")
    parser.add_argument("--commit", help="commit sha, recorded in the summary")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite the baseline with this run (only after a human has "
             "looked at the diff)",
    )
    args = parser.parse_args()

    cases = load_golden(args.golden)
    print(f"loaded {len(cases)} cases from {args.golden}")

    summary, results = run_eval(
        cases,
        build_system(),
        build_judge(),
        commit=args.commit,
        on_case=lambda r: print(
            f"  {r.case_id:<16} grounded={r.groundedness:.2f} "
            f"faithful={r.faithfulness:.2f} retrieval={r.retrieval_hit_rate:.2f}"
        ),
    )

    baseline = RunSummary.from_json(args.baseline) if args.baseline else None
    gate = evaluate_gate(summary, Thresholds(), baseline)

    print()
    print(f"groundedness      {summary.groundedness:.3f}")
    print(f"faithfulness      {summary.faithfulness:.3f}")
    print(f"retrieval hit     {summary.retrieval_hit_rate:.3f}")
    print(f"refusal accuracy  {summary.refusal_accuracy:.3f}")
    if summary.p95_latency_s is not None:
        print(f"p95 latency       {summary.p95_latency_s:.2f}s")
    print()
    print(gate.report())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_json(args.out)

    if args.update_baseline and gate.passed and args.baseline:
        summary.to_json(args.baseline)
        print(f"\nbaseline updated: {args.baseline}")

    # Non-zero exit is what blocks the merge.
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
