"""Pin the IRI-safety contract for ``CitationFound.source_uri``.

The first version of the NDA-matrix P3 citation JOIN fix built the
composite URI as ``f"{source_uri}#{block_ref}"``. That looked sensible
in isolation but produced ``"filename##/body/3"`` (double-hash) at
runtime because ``block_ref`` is *already* a JSON-pointer fragment
beginning with ``#`` (e.g. ``"#/body/3"``).

The kaos-graph Turtle exporter rejects double-hash IRIs with
``ValueError: Invalid IRI code point '#'`` per RFC 3987 §2.2 /
RFC 3986 §3 (an IRI may carry at most one fragment-identifier ``#``).
The corpus-stress suite reproduced the failure as 9 ValueErrors when
the agent persisted CitationFound to its session graph
(``kaos_agents/memory/store.py:save -> to_turtle``).

This test pins the contract: ``_build_citation_uri`` MUST emit a
string that contains at most one ``#``. Anything else silently breaks
graph persistence for any agent that runs findings-dispatch and tries
to save its memory — every kaos-ui SPA session, every corpus-stress
test, every audit-trail recording.
"""

from __future__ import annotations

import pytest

from kaos_agents.runtime.agent import _build_citation_uri


class TestSingleHashContract:
    """Every code path must emit at most one ``#`` in the citation URI.

    Failure of these assertions silently breaks kaos-graph RDF Turtle
    persistence — see module docstring for the full RFC reference.
    """

    def test_filename_plus_block_ref_has_single_hash(self) -> None:
        uri = _build_citation_uri(
            source_uri="EMNA Mutual NDA.docx",
            block_ref="#/body/3",
            finding_id="3e01701cccc7",
        )
        assert uri == "EMNA Mutual NDA.docx#/body/3"
        assert uri.count("#") == 1, f"double-hash regression: {uri!r}"

    def test_filename_only_has_no_hash(self) -> None:
        uri = _build_citation_uri(
            source_uri="EMNA Mutual NDA.docx",
            block_ref="",
            finding_id="3e01701cccc7",
        )
        assert uri == "EMNA Mutual NDA.docx"
        assert "#" not in uri

    def test_block_ref_only_preserves_single_hash(self) -> None:
        """Legacy path: no source_uri, block_ref present — emit it raw."""
        uri = _build_citation_uri(
            source_uri=None,
            block_ref="#/body/3",
            finding_id="3e01701cccc7",
        )
        assert uri == "#/body/3"
        assert uri.count("#") == 1

    def test_finding_id_fallback_when_nothing_else_resolved(self) -> None:
        """Last-resort opaque hex anchor when no source_uri AND no block_ref."""
        uri = _build_citation_uri(
            source_uri=None,
            block_ref="",
            finding_id="3e01701cccc7",
        )
        assert uri == "3e01701cccc7"
        assert "#" not in uri

    def test_deeply_nested_block_ref_single_hash(self) -> None:
        """Nested JSON pointer (``#/body/12/children/3``) still single-hash."""
        uri = _build_citation_uri(
            source_uri="MNDA - Acme.docx",
            block_ref="#/body/12/children/3",
            finding_id="abcd1234",
        )
        assert uri == "MNDA - Acme.docx#/body/12/children/3"
        assert uri.count("#") == 1


class TestUnicodeAndSpecialCharsInFilenames:
    """The composite URI must remain single-hash even when the filename
    contains spaces, dashes, or unicode — same RFC contract."""

    @pytest.mark.parametrize(
        "filename",
        [
            "EMNA Mutual NDA.docx",  # spaces
            "MNDA - Acme.docx",  # spaces + dash
            "Vertrag — München & Köln 中文.docx",  # unicode + ampersand
            "doc_with_underscores_and-dashes.pdf",
            "no_extension_doc",
        ],
    )
    def test_diverse_filenames_emit_single_hash(self, filename: str) -> None:
        uri = _build_citation_uri(
            source_uri=filename,
            block_ref="#/body/0",
            finding_id="aaa",
        )
        assert uri.count("#") == 1, f"double-hash for filename {filename!r}: {uri!r}"


class TestSubResourceGraphContract:
    """The IRI shape a hypothetical kaos-graph subject would build from
    these citation URIs must not violate RFC 3987 §2.2 (single fragment
    identifier).

    We don't import kaos-graph here — the contract is the IRI form
    itself. A real subject IRI built from this citation looks like:
    ``f"https://kaos.273ventures.com/ns/doc/{citation_uri}"``."""

    def test_subject_iri_from_composite_has_single_hash(self) -> None:
        cite = _build_citation_uri(
            source_uri="EMNA Mutual NDA.docx",
            block_ref="#/body/3",
            finding_id="abc",
        )
        subject_iri = f"https://kaos.273ventures.com/ns/doc/{cite}"
        assert subject_iri.count("#") == 1, (
            "subject IRI has >1 fragment-identifier '#' — kaos-graph "
            "Turtle exporter will raise: "
            f"{subject_iri!r}"
        )
