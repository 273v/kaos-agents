"""Live evals for ClassifyIntentSignature on the TOML-grounded few-shot pool.

Iteration 5 of kaos-modules/docs/audits/2026-05-24-kaos-agents-prompt-smells.md:
classify_intent now reads its calibration examples from
``kaos_agents/_assets/examples/classify_intent.toml`` rather than the
prior empty ``examples=`` argument. This file pins the 5 routing labels
on the live model floor — the floor list itself is centralized in
``tests/integration/_models.py``.

Anti-regression: session 01KS0R64Q744DTVZ53KCS9VC7M routed
"current US senator" → ``respond`` and answered from training memory
for three iterations. The ``tool_use_current_officeholder_with_web_category``
case below pins the fix.

Run with::

    uv run pytest tests/integration/test_classify_intent_live.py -v -m live
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from kaos_agents.context.classify import _classify_with_llm
from kaos_agents.memory.session import SessionMemory
from tests.integration._models import critic_model


def _have_key_for(model: str) -> bool:
    provider = model.split(":", 1)[0]
    env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    return bool(env and os.environ.get(env))


@dataclass(frozen=True, slots=True)
class ClassifyCase:
    case_id: str
    message: str
    expected_intent: str
    context_text: str = "(no prior messages)"
    available_tool_categories: str = ""


_CASES: tuple[ClassifyCase, ...] = (
    ClassifyCase(
        case_id="respond-thanks-acknowledgement",
        message="Thanks, that's helpful!",
        expected_intent="respond",
        context_text=(
            "User: Can you help me find the FRCP rule on response deadlines?\n"
            "Assistant: FRCP 12(a)(1)(A)(i) gives 21 days from service of the complaint."
        ),
        available_tool_categories=(
            "web: web search + page fetch\nsource: federal regulations, eCFR, SEC EDGAR retrieval"
        ),
    ),
    ClassifyCase(
        case_id="tool-use-current-senator-with-web-category",
        message="Who is the current junior US Senator from Michigan?",
        expected_intent="tool_use",
        available_tool_categories=(
            "web: web search + page fetch\nsource: federal regulations, eCFR, SEC EDGAR retrieval"
        ),
    ),
    ClassifyCase(
        case_id="respond-stable-concept",
        message="What is judicial review?",
        expected_intent="respond",
        available_tool_categories=("web: web search + page fetch\nsource: federal regulations"),
    ),
    ClassifyCase(
        case_id="plan-multistep-workflow",
        message=(
            "First parse the 10-K I uploaded, then extract every related-party "
            "transaction, then summarize the top three in a table."
        ),
        expected_intent="plan",
        context_text="(one 10-K filing attached)",
        available_tool_categories=(
            "content: search and read attached documents\n"
            "office: parse DOCX/PPTX/XLSX\npdf: parse PDF"
        ),
    ),
    ClassifyCase(
        case_id="clarify-ambiguous-reference",
        message="Can you look into that?",
        expected_intent="clarify",
        available_tool_categories="web: web search + page fetch",
    ),
)


# Intent classification is a CRITIC role — the model decides which
# intent the user expressed, and ``check_goal`` / ``IntentExtractor``
# downstream consumes the verdict the same way as the Reflexion
# critic. The default resolves to Sonnet; override with
# ``KAOS_TEST_CRITIC_MODEL=openai:gpt-5.4-mini`` to exercise the
# OpenAI provider variant of the same contract.
@pytest.mark.live
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
@pytest.mark.asyncio
async def test_classify_intent_label_emission(
    case: ClassifyCase,
) -> None:
    model = critic_model()
    if not _have_key_for(model):
        pytest.skip(f"missing API key for {model}")

    memory = SessionMemory(session_id="test")
    result = await _classify_with_llm(
        case.message,
        memory,
        model=model,
        context_text=case.context_text,
        available_tool_categories=case.available_tool_categories,
    )

    assert result.intent.value == case.expected_intent, (
        f"{case.case_id} x {model}: expected intent={case.expected_intent!r}, "
        f"got intent={result.intent.value!r}, confidence={result.confidence:.2f}, "
        f"reasoning={result.reasoning!r}"
    )
