"""S8 — research/corpus across API + MCP.

2 surfaces (API + MCP) x 2 providers = 4 tests. CLI variant covered
by ladder T08 (which uses Runner.corpus= injection — a Python-object
path neither API nor MCP exposes today).

The API/MCP surfaces don't accept ``corpus=ContentDocumentCorpus``
as a per-request override. The honest test is to pass the same 3-doc
mini-corpus INLINE in the prompt and assert that the model
synthesizes facts from all three docs. The agent has to *read* and
*combine* — not just echo — to pass.

This is weaker than ladder T08 (no RAG retrieval, no grounding
spans) but is the most-rigorous variant the surfaces support today.
The architectural gap — surfaces lack ContentDocumentCorpus
injection — is intentionally documented in this docstring rather
than papered over with a mocked test.
"""

from __future__ import annotations

import pytest

from tests.integration.surface_parity.conftest import (
    PROVIDERS,
    api_call,
    assert_no_error,
    mcp_call,
    model_for,
)

pytestmark = pytest.mark.live


# Identical corpus to ladder T08: a fictional animal so the model
# cannot fabricate from training data — every fact must come from
# the provided context.
_DOCS = (
    (
        "doc-0-mauve-lemur",
        "The mauve-tailed lemur (Lemur purpurus, fictional) was first "
        "described in 1923 by Dr. Estelle Carriere working in southern "
        "Madagascar. She published the description in the Bulletin of "
        "Madagascar Zoology, volume 4.",
    ),
    (
        "doc-1-mauve-lemur",
        "Adult mauve-tailed lemurs weigh between 800 and 1100 grams and "
        "measure 35-42 cm head-to-tail. They live primarily in deciduous "
        "forests of the south-east region and forage at dusk.",
    ),
    (
        "doc-2-mauve-lemur",
        "The IUCN Red List classifies the mauve-tailed lemur as "
        "Endangered (EN) as of the 2024 assessment, citing habitat loss "
        "from agricultural expansion as the primary threat.",
    ),
)

_QUESTION = (
    "Who first described the mauve-tailed lemur, what does it weigh, "
    "and what is its IUCN Red List status?"
)


def _build_prompt() -> str:
    context = "\n\n".join(f"=== DOC: {uri} ===\n{text}" for uri, text in _DOCS)
    return (
        "You are a research assistant. Answer the user's question using "
        "ONLY the documents below. If the answer requires combining "
        "facts from multiple documents, do so. Cite each doc you used.\n\n"
        f"QUESTION: {_QUESTION}\n\n"
        f"{context}"
    )


def _assert_synthesizes_three_docs(text: str, label: str) -> None:
    """The 3 facts buried in 3 separate docs all surface in the answer."""
    lower = text.lower()
    # Doc 0 — describer name
    assert "carriere" in lower or "estelle" in lower, (
        f"{label}: missing describer (doc 0: Estelle Carriere). Answer: {text[:400]!r}"
    )
    # Doc 1 — weight (any of the canonical numbers in the range)
    has_weight = any(n in lower for n in ("800", "1100", "1,100", "900", "1000"))
    assert has_weight, f"{label}: missing weight range (doc 1: 800-1100 g). Answer: {text[:400]!r}"
    # Doc 2 — IUCN status
    assert "endangered" in lower, (
        f"{label}: missing IUCN status (doc 2: Endangered). Answer: {text[:400]!r}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s8_research_via_api(provider: str) -> None:
    result = await api_call(
        _build_prompt(),
        provider=provider,
        session_id=f"s8-api-{provider}",
    )
    assert_no_error(result)
    _assert_synthesizes_three_docs(result.text, f"api/{provider}")


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_s8_research_via_mcp(provider: str) -> None:
    result = await mcp_call(
        "kaos-agent-chat",
        arguments={
            "message": _build_prompt(),
            "session_id": f"s8-mcp-{provider}",
            "model": model_for(provider),
        },
    )
    assert_no_error(result)
    _assert_synthesizes_three_docs(result.text, f"mcp/{provider}")
