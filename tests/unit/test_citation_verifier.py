"""Tests for B0.8 — citation verification against CourtListener.

Pre-B0.8 (broad-reliability roadmap §B0.8), no surface caught the
``Brown v. Board of Education, 347 U.S. 495 (1954)`` failure mode
(the real page is 483). Harvey publishes a 0.2% citation-error rate;
we didn't measure ours. This is the malpractice failure for attorneys.

Post-B0.8, ``kaos_agents.citations.verify_case_citation`` resolves a
:class:`kaos_citations.CaseCitation` against CourtListener's v4
citation-lookup endpoint and returns a structured
:class:`CitationVerificationResult`. The AgenticLoop post-worker
gate (next change) folds these results into a forced-replan when any
cite is ``mismatch`` or ``not_found``.

These tests pin the contract:

1. Live HTTP is gated by ``KAOS_AGENT_CITATION_VERIFY_ENABLED`` —
   without it every call returns ``unreachable`` (sandbox safety).
2. Successful CourtListener match → ``verified`` with cluster URL +
   observed name / year.
3. Name or year mismatch → ``mismatch`` with diagnostic.
4. Empty CourtListener response → ``not_found``.
5. Transport / 5xx errors → ``unreachable`` (never raises).
6. Missing structural fields (no volume / reporter / page) →
   ``skipped`` without ever hitting the network.
7. Multi-cite text → extract + verify each in parallel.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from kaos_citations import CaseCitation

from kaos_agents.citations import (
    CitationVerificationResult,
    verify_case_citation,
    verify_citations_in_text,
)


def _make_case(
    *,
    raw: str = "347 U.S. 483 (1954)",
    volume: int | None = 347,
    reporter: str | None = "U.S.",
    page: int | None = 483,
    year: int | None = 1954,
    case_name: str | None = "Brown v. Board of Education",
) -> CaseCitation:
    """Build a CaseCitation with the structural fields tests need."""
    return CaseCitation(
        raw=raw,
        normalized=raw,
        span=(0, len(raw)),
        cite_id="brown-v-board",
        volume=volume,
        reporter=reporter,
        page=page,
        year=year,
        case_name=case_name,
    )


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient shim for tests.

    Lets each test inject the exact response body / status / exception
    it wants without mocking the full transport stack.
    """

    def __init__(
        self,
        *,
        response_json: Any = None,
        status_code: int = 200,
        raise_on_post: Exception | None = None,
    ) -> None:
        self._response_json = response_json
        self._status_code = status_code
        self._raise = raise_on_post
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        self.posts.append((url, json))
        if self._raise is not None:
            raise self._raise
        # Build a minimal httpx.Response shaped enough for the caller.
        request = httpx.Request("POST", url, json=json, headers=headers)
        return httpx.Response(
            status_code=self._status_code,
            json=self._response_json,
            request=request,
        )


# ── Gating ──────────────────────────────────────────────────────────


class TestVerificationGating:
    """Live HTTP must be opt-in via env var. Default = sandbox-safe."""

    @pytest.mark.asyncio
    async def test_default_disabled_returns_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", raising=False)
        result = await verify_case_citation(_make_case())
        assert result.status == "unreachable"
        assert result.diagnostic is not None
        assert "disabled" in result.diagnostic.lower()
        # is_problem must be False — the cite might be correct;
        # the AgenticLoop gate should NOT force a replan on
        # unreachable.
        assert result.is_problem is False

    @pytest.mark.asyncio
    async def test_enabled_via_env_attempts_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "absolute_url": "/opinion/103218/brown-v-board-of-education/",
                        }
                    ]
                }
            ]
        )
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "verified"
        assert client.posts, "expected at least one HTTP call"


# ── Status mapping ──────────────────────────────────────────────────


class TestStatusMapping:
    """Each documented status comes from a specific CL response shape."""

    @pytest.mark.asyncio
    async def test_verified_when_name_and_year_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "absolute_url": "/opinion/103218/brown-v-board-of-education/",
                        }
                    ]
                }
            ]
        )
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "verified"
        assert result.observed_case_name == "Brown v. Board of Education"
        assert result.observed_year == 1954
        assert result.courtlistener_url == (
            "https://www.courtlistener.com/opinion/103218/brown-v-board-of-education/"
        )
        assert result.is_problem is False

    @pytest.mark.asyncio
    async def test_case_name_loose_match_accepts_extended_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extracted cites often carry extra context ("of Topeka").
        CL's canonical form is shorter. Either-direction containment
        passes — no false mismatch on common abbreviation noise."""
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "absolute_url": "/x/",
                        }
                    ]
                }
            ]
        )
        cite = _make_case(case_name="Brown v. Board of Education of Topeka")
        result = await verify_case_citation(cite, client=client)
        assert result.status == "verified"

    @pytest.mark.asyncio
    async def test_mismatch_on_wrong_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Plessy v. Ferguson",  # wrong case!
                            "date_filed": "1954-05-17",
                            "absolute_url": "/x/",
                        }
                    ]
                }
            ]
        )
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "mismatch"
        assert result.diagnostic is not None
        assert "case_name" in result.diagnostic
        assert result.is_problem is True

    @pytest.mark.asyncio
    async def test_mismatch_on_wrong_year(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1896-05-18",  # decade-off
                            "absolute_url": "/x/",
                        }
                    ]
                }
            ]
        )
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "mismatch"
        assert "year" in (result.diagnostic or "")

    @pytest.mark.asyncio
    async def test_year_off_by_one_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CL stores filing date; printed reporter year sometimes
        rolls forward when an opinion crosses calendar-year. ±1 is
        deliberate tolerance — false mismatches here drown out the
        real ones."""
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(
            response_json=[
                {
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1953-12-09",
                            "absolute_url": "/x/",
                        }
                    ]
                }
            ]
        )
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "verified"

    @pytest.mark.asyncio
    async def test_not_found_when_clusters_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(response_json=[{"clusters": []}])
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "not_found"
        assert result.is_problem is True

    @pytest.mark.asyncio
    async def test_unreachable_on_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(response_json={}, status_code=503)
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "unreachable"
        assert result.is_problem is False

    @pytest.mark.asyncio
    async def test_unreachable_on_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(raise_on_post=httpx.ConnectError("dns failure"))
        result = await verify_case_citation(_make_case(), client=client)
        assert result.status == "unreachable"
        assert "ConnectError" in (result.diagnostic or "")


# ── Skipped (no structural fields) ─────────────────────────────────


class TestSkippedWithoutFields:
    """Citations missing volume / reporter / page can't be resolved.
    Status ``skipped`` keeps the gate honest — neither pass nor fail."""

    @pytest.mark.asyncio
    async def test_missing_volume_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(response_json=[])
        result = await verify_case_citation(_make_case(volume=None), client=client)
        assert result.status == "skipped"
        assert client.posts == [], "skipped cite must NOT hit the network"

    @pytest.mark.asyncio
    async def test_missing_reporter_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(response_json=[])
        result = await verify_case_citation(_make_case(reporter=None), client=client)
        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_missing_page_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", "1")
        client = _FakeAsyncClient(response_json=[])
        result = await verify_case_citation(_make_case(page=None), client=client)
        assert result.status == "skipped"


# ── Frozen-value contract ──────────────────────────────────────────


class TestResultIsFrozen:
    """``CitationVerificationResult`` is a frozen value type — safe
    to attach to events / persist to memory."""

    def test_frozen(self) -> None:
        r = CitationVerificationResult(status="verified", raw_cite="x")
        # ty correctly flags ``r.status = ...`` at static analysis;
        # going through setattr keeps the test ty-clean while still
        # exercising the frozen-dataclass guarantee at runtime.
        with pytest.raises((AttributeError, TypeError)):
            setattr(r, "status", "mismatch")  # noqa: B010

    def test_is_problem_only_true_for_mismatch_or_not_found(self) -> None:
        for status in ("verified", "unreachable", "skipped"):
            assert CitationVerificationResult(status=status, raw_cite="x").is_problem is False
        for status in ("mismatch", "not_found"):
            assert CitationVerificationResult(status=status, raw_cite="x").is_problem is True


# ── Multi-cite path ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_citations_in_text_empty_for_no_case_cites() -> None:
    """Non-legal text yields zero case citations → empty tuple.
    Common case for non-legal queries; must not waste CL calls."""
    results = await verify_citations_in_text("What is the weather in Tokyo today?")
    assert results == ()


@pytest.mark.asyncio
async def test_verify_citations_in_text_extracts_and_calls_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-cite path extracts via kaos_citations.extract_citations
    then calls verify_case_citation for each in parallel."""
    monkeypatch.delenv("KAOS_AGENT_CITATION_VERIFY_ENABLED", raising=False)
    text = (
        "See Brown v. Board of Education, 347 U.S. 483 (1954). "
        "Compare Plessy v. Ferguson, 163 U.S. 537 (1896)."
    )
    results = await verify_citations_in_text(text)
    # With CL disabled, every cite resolves to ``unreachable``
    # without network traffic. The contract is that the function
    # returns one result per case cite extracted.
    assert len(results) >= 1
    for r in results:
        assert r.status == "unreachable"


# ── Defensive imports stay clean ────────────────────────────────────


def test_module_imports_without_kaos_citations_runtime_call() -> None:
    """Importing the module must not eagerly call into
    kaos_citations.extract_citations or any HTTP client.

    Regression: a previous version of this module called
    ``extract_citations`` at import time during a constant
    initialization — broke offline CI."""
    import importlib

    import kaos_agents.citations as mod

    importlib.reload(mod)
    # No assertion needed; if reload raises, the test fails.
    assert hasattr(mod, "verify_case_citation")
