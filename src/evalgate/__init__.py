"""evalgate - a release gate for retrieval-augmented and agentic systems."""

from .dataset import GoldenCase, load_golden
from .gate import GateResult, RunSummary, Thresholds, evaluate_gate
from .judge import Judge, Verdict
from .metrics import retrieval_quality
from .runner import Response, run_eval

__all__ = [
    "GoldenCase", "load_golden",
    "Judge", "Verdict",
    "retrieval_quality",
    "Response", "run_eval",
    "RunSummary", "Thresholds", "GateResult", "evaluate_gate",
]
