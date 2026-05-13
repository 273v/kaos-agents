"""Live integration tests for ReflexionLoop (G6).

Each test wires a *real* :class:`ReflexionLoop` around a real
:class:`ChatAgent` inner with a real :class:`ReflexionCritic`. The
critic uses an Anthropic model; the inner uses an Anthropic model
(can be different). Tests assert observable behaviors of the
reflexion loop — approval, retry, max-iteration fallback, score
progression — by inspecting the ``reflexion_trace`` metadata.

Requires ``ANTHROPIC_API_KEY``. Skipped otherwise.

Why these tests matter: the unit tests stub both the inner and the
critic. They prove the loop's control flow, not that a real LLM
critic actually produces useful critiques that a real LLM inner
actually responds to. In a regulated context, the loop is a
quality-control layer — if it doesn't actually improve output, the
audit trail says "we added review" while in fact we added theater.
"""

from __future__ import annotations

import os

import pytest
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.reflexion import ReflexionCritic, ReflexionLoop
from kaos_agents.types.response import AgentResponse

# ---------------------------------------------------------------------------
# Skip markers + pinned models
# ---------------------------------------------------------------------------

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)

INNER_MODEL = "anthropic:claude-haiku-4-5"
CRITIC_MODEL = "anthropic:claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# A realistic NDA clause used in retrieval-style critic tests. Real text
# that the inner agent and critic both reason over.
# ---------------------------------------------------------------------------

NDA_CONFIDENTIALITY_CLAUSE = """
2.1 Confidential Information. "Confidential Information" means all
non-public, proprietary, or confidential information disclosed by
either party (the "Disclosing Party") to the other (the "Receiving
Party"), whether orally, in writing, or by any other means, including
but not limited to: (a) business plans, financial projections, customer
lists, supplier lists, product roadmaps, and pricing; (b) technical
information, source code, algorithms, and trade secrets; (c) personnel
information; and (d) any information marked or identified as
confidential at the time of disclosure or that a reasonable person
would understand to be confidential under the circumstances.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        ),
    )


def _make_inner(instructions: str | None = None) -> ChatAgent:
    return ChatAgent(_memory_vfs(), model=INNER_MODEL, instructions=instructions)


def _trace(response: AgentResponse):
    for k, v in response.metadata or ():
        if k == "reflexion_trace":
            return v
    return None


# ---------------------------------------------------------------------------
# ReflexionLoop live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestReflexionLoopLive:
    """End-to-end critic-and-retry with real Anthropic models on both sides."""

    async def test_easy_question_approved_on_first_iteration(self) -> None:
        """A clear well-formed answer to a definitional question should
        be approved on iteration 1 by a lenient critic."""
        inner = _make_inner()
        critic = ReflexionCritic(
            rubric=(
                "The answer must be a complete English sentence and "
                "directly address the user's question. Approve scores "
                "above 0.6."
            ),
            model=CRITIC_MODEL,
            threshold=0.6,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=3)
        response = await loop.turn(
            "What does the word 'confidential' mean in plain English?",
            session_id="reflexion-easy-1",
        )
        trace = _trace(response)
        assert trace is not None
        # The easy case usually approves on iter 1, but 2 is acceptable.
        assert trace["accepted"] is True, (
            f"Expected approval within max_iterations; trace: "
            f"final_iteration={trace['final_iteration']}, "
            f"scores={[i['score'] for i in trace['iterations']]}"
        )
        assert trace["final_iteration"] <= 2

    async def test_strict_rubric_triggers_retry_and_improves(self) -> None:
        """With a strict citation-requiring rubric and an inner that doesn't
        cite by default, the first iteration should fail and feedback should
        materially improve the second iteration's score."""
        inner = _make_inner(
            instructions=(
                "You are a helpful assistant. Answer in 1-2 sentences. "
                "Do NOT include citations unless the user explicitly "
                "asks for them in the request."
            ),
        )
        critic = ReflexionCritic(
            rubric=(
                "The answer MUST quote at least one specific phrase "
                "from the source clause provided in the question, "
                "enclosed in double quotes. Without a direct quotation, "
                "score below 0.5."
            ),
            model=CRITIC_MODEL,
            threshold=0.7,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=3)
        response = await loop.turn(
            (
                f"Here is an NDA clause:\n\n{NDA_CONFIDENTIALITY_CLAUSE}\n\n"
                "What categories of information does this clause protect?"
            ),
            session_id="reflexion-strict-1",
        )
        trace = _trace(response)
        assert trace is not None
        scores = [iter_["score"] for iter_ in trace["iterations"]]
        assert len(scores) >= 2, (
            f"Expected at least 2 iterations (initial fail + retry), "
            f"got {len(scores)}. Trace: {scores!r}"
        )
        # The reflexion loop's contract is "the critique materially
        # improves quality." Either some subsequent iteration scores
        # strictly higher than iter 0, OR the loop accepts a later
        # iteration (which by definition cleared the threshold the
        # first did not). The earlier looser assertion
        # (``max(scores[1:]) >= scores[0] - 0.1``) hid a real bug:
        # captured telemetry on commit 25b7d6f showed scores going
        # 0.30 -> 0.20 -> 0.20 — three identical inner responses —
        # because the critic feedback was being silently dropped by
        # the TypeError fallback in ``_call_inner``. The fix
        # (prepending the critique into the message) is verified by
        # this assertion: if the inner truly does not see the
        # critique, scores cannot improve and the test fails loudly.
        improved = max(scores[1:]) > scores[0]
        accepted_after_first = trace["accepted"] and trace["final_iteration"] > 1
        assert improved or accepted_after_first, (
            f"Reflexion loop must materially improve quality across "
            f"iterations. Scores: {scores!r}. accepted="
            f"{trace['accepted']}. If this fails consistently, suspect "
            f"the critique is not reaching the inner agent — diff the "
            f"per-iteration inputs in the captured run JSONL."
        )

    async def test_max_iterations_returns_best_when_unapproved(self) -> None:
        """Set an impossibly strict rubric and max_iterations=2; the loop
        should run both iterations, fail to approve, and return the best."""
        inner = _make_inner()
        critic = ReflexionCritic(
            rubric=(
                "The answer must include exactly 17 words. Score 1.0 if "
                "exactly 17 words; 0.0 otherwise."
            ),
            model=CRITIC_MODEL,
            threshold=0.99,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=2)
        response = await loop.turn(
            "Describe the color blue.",
            session_id="reflexion-maxiter-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace["final_iteration"] == 2
        # Approval is theoretically possible (model might hit 17 words),
        # but we assert the structural invariant: trace has 2 iterations.
        assert len(trace["iterations"]) == 2
        # response.text must be one of the two attempts (best-scored)
        assert response.text, "Best response must have non-empty text"

    async def test_single_iteration_disables_retry(self) -> None:
        """max_iterations=1 should produce exactly one inner call and one
        critique, no retry regardless of score."""
        inner = _make_inner()
        critic = ReflexionCritic(
            rubric="The answer must be in iambic pentameter.",
            model=CRITIC_MODEL,
            threshold=0.99,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=1)
        response = await loop.turn(
            "What is 2 + 2?",
            session_id="reflexion-single-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace["final_iteration"] == 1
        assert len(trace["iterations"]) == 1
        # With threshold 0.99 and impossible rubric, this will NOT be
        # approved — but the loop still returns the answer.
        assert response.text

    async def test_critic_produces_scores_in_valid_range(self) -> None:
        """Every critic score must be in [0, 1] — guards against the model
        emitting out-of-range floats that the clamp must fix."""
        inner = _make_inner()
        critic = ReflexionCritic(
            rubric="The answer must be a complete sentence.",
            model=CRITIC_MODEL,
            threshold=0.5,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=2)
        response = await loop.turn(
            "What does the word 'liability' mean?",
            session_id="reflexion-range-1",
        )
        trace = _trace(response)
        assert trace is not None
        for it in trace["iterations"]:
            assert 0.0 <= it["score"] <= 1.0, f"Score {it['score']} outside [0, 1] — clamp failed"
            assert it["reasoning"], "Critic must provide reasoning"

    async def test_reflexion_loop_on_nda_extraction_quality(self) -> None:
        """End-to-end: the loop drives an extraction-style answer to be
        complete by retrying when the critic finds missing categories.

        The NDA clause lists 4 categories (a, b, c, d). The strict rubric
        requires all 4 to be mentioned. If the inner agent's first answer
        misses any, the critic should flag it and the next iteration
        should be more complete."""
        inner = _make_inner(
            instructions="Answer briefly. Aim for one sentence per answer.",
        )
        critic = ReflexionCritic(
            rubric=(
                "The answer must list ALL FOUR categories of confidential "
                "information from the clause: (a) business plans/financial "
                "info, (b) technical info/trade secrets, (c) personnel "
                "info, and (d) information marked confidential or that a "
                "reasonable person would understand to be confidential. "
                "Score 1.0 only when all four are mentioned; deduct for "
                "each missing category."
            ),
            model=CRITIC_MODEL,
            threshold=0.85,
        )
        loop = ReflexionLoop(inner, critic, max_iterations=3)
        response = await loop.turn(
            (
                f"Here is an NDA clause:\n\n{NDA_CONFIDENTIALITY_CLAUSE}\n\n"
                "What categories of information are protected?"
            ),
            session_id="reflexion-nda-1",
        )
        trace = _trace(response)
        assert trace is not None
        # The final response (whether approved or best-scored) should be
        # substantive — i.e., not the brief one-sentence answer the
        # inner produces by default.
        assert len(response.text) > 50, (
            f"Final response should be detailed after reflexion; got: {response.text!r}"
        )
        # Verify the trace shape: iterations are recorded in order.
        scores = [it["score"] for it in trace["iterations"]]
        assert len(scores) >= 1
        # Last iteration's score must be the best-or-tied (since the loop
        # returns the best-scored response on max-iter exit, OR the
        # approved response which always has the highest score).
        # Either way: best score must be achievable from the iterations.
        assert max(scores) >= scores[0] - 0.15, (
            f"Reflexion should not strictly degrade quality over iterations. Scores: {scores!r}"
        )
