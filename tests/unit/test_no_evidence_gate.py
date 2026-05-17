"""Tests for :mod:`kaos_agents.grounding.no_evidence_gate`.

Covers the gate that refuses to ship a fabricated answer when every
tool call attempted in a turn failed AND the user explicitly
referenced files. Production trigger: session
``01KRVYAEA3B1HG95DBAG6H0DJ3`` — five NDA .docx uploads, every
``kaos-office-parse-docx`` call returned "File not found", agent
hallucinated jurisdiction/term analysis citing files it never read.

Test matrix covers the decision boundaries:
- happy path: tools succeed → no refusal (LLM ships its draft)
- partial success: some tools fail → no refusal
- all-fail + file mention in message → refuse
- all-fail + attached docs (no filename in message) → refuse
- all-fail + no file context → no refusal (don't muzzle generic chat)
- empty observation list → no refusal
- filename extraction across the documented extensions
- refusal text composition includes file list, tool list, error excerpts
- error-excerpt extraction from kaos error envelopes (closed + truncated)
"""

from __future__ import annotations

from kaos_agents.grounding.no_evidence_gate import (
    NoEvidenceVerdict,
    ToolObservationSummary,
    _excerpt_error,
    evaluate_no_evidence_gate,
    extract_referenced_files,
    render_refusal_text,
)

# ---------------------------------------------------------------------------
# evaluate_no_evidence_gate
# ---------------------------------------------------------------------------


def test_no_tool_calls_attempted_does_not_refuse() -> None:
    verdict = evaluate_no_evidence_gate(
        observations=[],
        user_message="What's in my Contract.pdf?",
        attached_documents=["Contract.pdf"],
    )
    assert verdict == NoEvidenceVerdict(refuse=False)


def test_at_least_one_success_does_not_refuse() -> None:
    obs = [
        ToolObservationSummary(tool_name="kaos-office-parse-docx", is_error=True),
        ToolObservationSummary(tool_name="kaos-pdf-extract-parse", is_error=False),
    ]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="summarize the attached files",
        attached_documents=["NDA.docx", "Brief.pdf"],
    )
    assert verdict.refuse is False


def test_all_failed_with_filename_in_message_refuses() -> None:
    obs = [
        ToolObservationSummary(
            tool_name="kaos-office-parse-docx",
            is_error=True,
            result_preview='{"error": true, "message": "File not found: EMNA Mutual NDA.docx"}',
        ),
        ToolObservationSummary(
            tool_name="kaos-office-parse-docx",
            is_error=True,
            result_preview='{"error": true, "message": "File not found: MNDA - Acme.docx"}',
        ),
    ]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="summarize the key terms in EMNA Mutual NDA.docx and MNDA - Acme.docx",
    )
    assert verdict.refuse is True
    assert verdict.failed_tool_count == 2
    # The extractor captures the all-caps + hyphen filename tokens but
    # may clip lowercase-only middle words like "Mutual"; that's a
    # known trade-off (see _looks_filename_part docstring). What
    # matters is that the gate detected file references and refused.
    assert any("NDA.docx" in name for name in verdict.referenced_files)
    assert any("Acme.docx" in name for name in verdict.referenced_files)
    # Excerpts pull the human message out of the kaos error envelope.
    assert any("File not found" in e for e in verdict.error_excerpts)


def test_all_failed_with_attached_documents_refuses_even_without_filename_in_message() -> None:
    # Reproduces the production trigger: user said "summarize the key
    # terms in a table" with no filename in the text, but attached 5
    # NDA documents.
    obs = [
        ToolObservationSummary(tool_name="kaos-office-parse-docx", is_error=True) for _ in range(5)
    ]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="summarize the key terms in a table",
        attached_documents=[
            "EMNA Mutual NDA.docx",
            "MNDA - Acme.docx",
            "MNDA - BI.docx",
            "MNDA - CC Final 2.docx",
            "MNDA - DynaMo.docx",
        ],
    )
    assert verdict.refuse is True
    assert verdict.failed_tool_count == 5
    assert len(verdict.referenced_files) == 5


def test_all_failed_filename_extracted_from_tool_call_args() -> None:
    # Reproduces the production trigger exactly: user said
    # "summarize the key terms in a table" with no filename in the
    # message AND no auto-hydrated documents — but the agent's
    # OWN tool calls reached for filenames. The gate must catch
    # this path or the production hallucination is still possible.
    obs = [
        ToolObservationSummary(
            tool_name="kaos-office-parse-docx",
            is_error=True,
            arguments_preview=(
                '{"path": "sessions/01KRVYAEA3B1HG95DBAG6H0DJ3/files/EMNA Mutual NDA.docx"}'
            ),
            result_preview='{"error": true, "message": "File not found"}',
        ),
        ToolObservationSummary(
            tool_name="kaos-office-parse-docx",
            is_error=True,
            arguments_preview=(
                '{"path": "sessions/01KRVYAEA3B1HG95DBAG6H0DJ3/files/MNDA - Acme.docx"}'
            ),
            result_preview='{"error": true, "message": "File not found"}',
        ),
    ]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="summarize the key terms in a table",
        attached_documents=(),
    )
    assert verdict.refuse is True
    # The all-caps + hyphen filename tokens get captured; lowercase
    # middle words like "Mutual" may be clipped (see
    # _looks_filename_part). What matters is the gate detected the
    # file references.
    assert any("NDA.docx" in name for name in verdict.referenced_files)
    assert any("Acme.docx" in name for name in verdict.referenced_files)


def test_all_failed_with_no_file_context_does_not_refuse() -> None:
    # Generic chat: every tool happened to fail (e.g. SERP outage) but
    # the question wasn't about named files. The agent should still be
    # allowed to respond from priors — this gate is specifically about
    # refusing fabricated facts ABOUT NAMED FILES.
    obs = [
        ToolObservationSummary(tool_name="kaos-web-fetch-page", is_error=True),
        ToolObservationSummary(tool_name="kaos-source-fetch-url", is_error=True),
    ]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="what is the capital of France?",
        attached_documents=[],
    )
    assert verdict.refuse is False


def test_dedupe_referenced_files_across_attached_and_message() -> None:
    obs = [ToolObservationSummary(tool_name="x", is_error=True)]
    verdict = evaluate_no_evidence_gate(
        observations=obs,
        user_message="look at contract.pdf again",
        attached_documents=["Contract.pdf"],
    )
    # Same file referenced via attachment + message — appears once.
    assert verdict.refuse is True
    assert verdict.referenced_files in (
        ("Contract.pdf",),
        ("contract.pdf",),
    )


# ---------------------------------------------------------------------------
# extract_referenced_files
# ---------------------------------------------------------------------------


def test_extract_filename_simple() -> None:
    assert extract_referenced_files("read Contract.pdf please") == ("Contract.pdf",)


def test_extract_filename_with_spaces() -> None:
    # Production case — kaos-office tools accept names with spaces. The
    # extractor absorbs all-caps prefix words (EMNA, NDA) but clips
    # lowercase middle words ("Mutual"). The captured token is enough
    # for the gate to refuse and tell the user which file failed.
    files = extract_referenced_files("summarize EMNA Mutual NDA.docx")
    assert any("NDA.docx" in f for f in files)


def test_extract_multiple_extensions() -> None:
    msg = "check the Contract.pdf and the data.csv and the report.docx"
    files = extract_referenced_files(msg)
    assert "Contract.pdf" in files
    assert "data.csv" in files
    assert "report.docx" in files


def test_extract_returns_empty_when_no_filenames() -> None:
    assert extract_referenced_files("how are you today?") == ()


def test_extract_does_not_match_bare_words() -> None:
    # "my file" (no extension) must not trigger.
    assert extract_referenced_files("check my file") == ()


def test_extract_case_insensitive_dedupe() -> None:
    files = extract_referenced_files("Read Contract.PDF and contract.pdf")
    # Either casing wins but only one entry survives.
    assert len(files) == 1


# ---------------------------------------------------------------------------
# render_refusal_text
# ---------------------------------------------------------------------------


def test_render_returns_empty_when_not_refusing() -> None:
    assert render_refusal_text(NoEvidenceVerdict(refuse=False)) == ""


def test_render_includes_file_list_and_tools() -> None:
    verdict = NoEvidenceVerdict(
        refuse=True,
        reason="all-failed",
        referenced_files=("EMNA.docx", "Acme.docx"),
        failed_tool_count=2,
        failed_tools=("kaos-office-parse-docx", "kaos-office-parse-docx"),
        error_excerpts=("File not found: EMNA.docx", "File not found: Acme.docx"),
    )
    text = render_refusal_text(verdict)
    assert "EMNA.docx" in text
    assert "Acme.docx" in text
    # Dedupe identical tool names so the chip doesn't repeat.
    assert text.count("kaos-office-parse-docx") == 1
    # Calls out the absolute count so the user knows the scale of the failure.
    assert "2 call" in text
    # Has the load-bearing "I will NOT fabricate" line.
    assert "NOT fabricate" in text


def test_render_truncates_long_file_lists() -> None:
    files = tuple(f"file{i}.docx" for i in range(20))
    verdict = NoEvidenceVerdict(
        refuse=True,
        reason="all-failed",
        referenced_files=files,
        failed_tool_count=20,
        failed_tools=("kaos-office-parse-docx",) * 20,
        error_excerpts=(),
    )
    text = render_refusal_text(verdict, max_files=5)
    assert "and 15 more" in text


def test_render_includes_excerpt_lines_for_distinct_errors() -> None:
    verdict = NoEvidenceVerdict(
        refuse=True,
        reason="",
        referenced_files=("a.docx",),
        failed_tool_count=3,
        failed_tools=("t",) * 3,
        error_excerpts=("not found", "not found", "permission denied"),
    )
    text = render_refusal_text(verdict)
    # Two unique excerpts → both rendered, duplicates suppressed.
    assert text.count("- not found") == 1
    assert text.count("- permission denied") == 1


# ---------------------------------------------------------------------------
# _excerpt_error
# ---------------------------------------------------------------------------


def test_excerpt_pulls_message_from_closed_envelope() -> None:
    payload = '{"error": true, "message": "File not found: x.docx", "locator": "x"}'
    assert _excerpt_error(payload) == "File not found: x.docx"


def test_excerpt_pulls_message_from_truncated_envelope() -> None:
    # Wire format chops at ~200 chars — the closing quote may be missing
    # and json.loads would fail. Our regex still extracts.
    payload = (
        '{"error": true, "message": "Fetch failed for '
        "'https://finance.yahoo.com/quote/AAPL/history': HTTP source returned"
    )
    out = _excerpt_error(payload)
    # Excerpt is bounded — no full URL noise.
    assert "Fetch failed" in out
    assert len(out) <= 161


def test_excerpt_falls_back_when_no_message_field() -> None:
    payload = "plain error string with no JSON envelope"
    assert _excerpt_error(payload) == payload


def test_excerpt_empty_in_empty_out() -> None:
    assert _excerpt_error("") == ""
