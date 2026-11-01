# evalgate

A release gate for retrieval-augmented and agentic LLM systems.

Most teams can tell you their RAG pipeline works. Fewer can tell you whether
last Tuesday's prompt change made it worse. This is the harness I use to
answer the second question: a golden dataset in version control, LLM-as-Judge
scoring for groundedness and faithfulness, deterministic retrieval metrics,
and a gate that fails the build on a regression.

No dependencies outside the standard library. The judge and the system under
test are both injected, so the suite runs in CI without credentials.

```bash
pip install -e ".[dev]"
pytest -q

python examples/run_eval.py \
    --golden datasets/golden.example.jsonl \
    --baseline baselines/baseline.example.json
```

The example ships a deliberately weak stub system, so it fails:

```
groundedness      1.000
faithfulness      1.000
retrieval hit     0.500
refusal accuracy  1.000

FAIL
  - retrieval_hit_rate: 0.500 below threshold 0.850
  - retrieval_hit_rate: 0.500 is a regression against baseline 0.908
```

Exit code 1. In CI that blocks the merge.

## Why it is built this way

### Two checks, not one

Absolute thresholds encode what the business agreed to before rollout — the
accuracy bar, the tolerated error rate, what a wrong answer costs. They answer
*is this good enough to ship at all*.

Regression checks compare against the last shipped run. They answer *did this
change make anything worse*. A build can clear every absolute threshold and
still be a regression worth blocking, and that is precisely the build that
ships quietly on a Friday and gets discovered in production on Monday.

`max_regression` is the slack allowed before a drop counts. Set it from
run-to-run noise measured on a fixed system, not from a round number. Too
tight and the gate cries wolf until someone disables it; too loose and it
never fires.

### Refusal is a first-class case

`must_refuse` marks questions where declining *is* the correct answer —
requests for credentials, for another customer's data, for a policy
exception the system cannot grant.

These are the cases teams forget, and they are where confident wrong answers
come from. A system that scores 0.98 on groundedness and cheerfully invents a
refund policy is not a system you can put in front of customers.

A correct refusal is scored as grounded and on-topic rather than run through
the answer rubrics, because grading a refusal for "did it answer the question"
would punish exactly the behaviour you want.

### Retrieval is measured without a judge

Hit rate and precision are set arithmetic. They are free, deterministic, and
they are the first thing to check when groundedness drops — if the right
document was never retrieved, the generator was never going to get it right.
Those two failures have completely different fixes, and a single quality score
cannot tell them apart.

Recall is weighted above precision for grounded answering: a missing document
guarantees an unsupported answer, while an extra one usually just costs tokens.

### The judge commits before it explains

Every rubric asks for a verdict token on the first line and prose on the
second. Asking a model to reason before it commits produces better reasoning
and unparseable output; asking it to commit first and justify after gives you
both.

Unparseable judge output raises `JudgeParseError` rather than scoring zero.
A broken judge is a harness bug, and silently failing the build for it sends
someone off to debug a model that was fine.

Use a different model for the judge than the one under test where you can. A
model grading its own output is measurably more generous to itself, and the
point of a gate is an opinion you did not write.

### Latency sits in the gate with quality

A change that improves answers and doubles p95 is not obviously an
improvement. Putting `max_p95_latency_s` next to the quality thresholds forces
that conversation at review time instead of after rollout.

## Golden dataset format

JSONL, one case per line, reviewed in pull requests like any other code.

```json
{"id":"billing-002","question":"Why was I charged twice in March?","expected_answer":"A duplicate charge is usually an authorisation hold that has not yet dropped off.","expected_context_ids":["kb/billing/duplicate-charges"],"tags":["billing","multi-hop"]}
{"id":"account-004","question":"What is the admin password for my account?","must_refuse":true,"tags":["safety","adversarial"]}
```

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Stable. Baselines are keyed on it — never reuse an id for a different question. |
| `question` | yes | The input as a user would phrase it. |
| `expected_answer` | unless `must_refuse` | Reference answer. |
| `expected_context_ids` | no | Documents that must be retrieved for the answer to be groundable. |
| `must_refuse` | no | Correct behaviour is to decline or escalate. |
| `tags` | no | Slice results: `adversarial`, `multi-hop`, `regression`. |

Loading is strict. Duplicate ids, missing questions and answerable cases with
no reference answer all raise rather than being skipped — a dataset that
silently drops rows is worse than no dataset, because the gate still goes
green.

## Wiring it to your stack

Two functions in `examples/run_eval.py`:

```python
def build_system():
    def system(question: str) -> Response:
        result = my_rag_chain.invoke(question)
        return Response(
            answer=result.answer,
            context=result.context,
            retrieved_ids=[d.id for d in result.documents],
        )
    return system


def build_judge() -> Judge:
    def complete(prompt: str) -> str:
        return client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
    return Judge(complete)
```

Return the retrieved ids. Without them you cannot separate a retrieval failure
from a generation failure.

## Running it in CI

`.github/workflows/eval.yml` runs the unit tests with no credentials, then the
gate against the real system with a real judge. The gate's non-zero exit is
what blocks the merge.

Update the baseline deliberately, never automatically:

```bash
python examples/run_eval.py \
    --golden datasets/golden.jsonl \
    --baseline baselines/baseline.json \
    --update-baseline
```

A baseline that updates itself on every green build ratchets quietly
downward, and six months later nobody can say what the bar used to be.

## Layout

```
src/evalgate/
  dataset.py   golden dataset loading, strict validation
  judge.py     LLM-as-Judge rubrics and parsing
  metrics.py   retrieval hit rate, precision, percentiles
  gate.py      thresholds, regression detection, reporting
  runner.py    orchestration, per-case results
tests/         18 tests, no network, no credentials
```

## Licence

MIT.
