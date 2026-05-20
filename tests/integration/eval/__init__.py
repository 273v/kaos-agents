"""kaos-agents-bench — multi-model live evaluation harness.

Runs a curated catalogue of agentic-AI pathologies (see
`pathologies.py`) against multiple frontier models in parallel and
audits the resulting event traces for expected signals.

The harness is **catalog-driven and model-adaptive**: every
pathology declares its expected event-trace shape rather than a
symptom-text regex. Tests assert what the agent DID (tool spans,
intent classifications, cost), not what its prose says.

Companion to `docs/plans/2026-05-19-agentic-failure-taxonomy.md`
and `docs/plans/2026-05-19-three-layer-test-protocol.md`.
"""

from tests.integration.eval.harness import (
    EvalReport,
    EvalResult,
    SignalOutcome,
    run_pathology,
    run_pathology_matrix,
)
from tests.integration.eval.pathologies import (
    PATHOLOGY_PACK,
    ExpectedSignal,
    Pathology,
)

__all__ = [
    "PATHOLOGY_PACK",
    "EvalReport",
    "EvalResult",
    "ExpectedSignal",
    "Pathology",
    "SignalOutcome",
    "run_pathology",
    "run_pathology_matrix",
]
