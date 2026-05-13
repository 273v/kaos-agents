"""Regression test for the bundled JSONL telemetry viewer assets (KC18).

The viewer is a static HTML page plus a thin :mod:`http.server` launcher.
Rendering correctness needs a browser and is exercised manually; here we
verify that:

* ``index.html`` is parseable (no broken tags),
* the CDN scripts referenced in the README and the spec are present,
* enough Alpine-wired attributes exist that the SPA is actually live,
* the CLI launcher exposes a ``main()`` entry point.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from kaos_agents.examples.viewer import __main__ as viewer_cli

INDEX_PATH = Path(str(files("kaos_agents.examples.viewer").joinpath("index.html")))

# CDN URLs the viewer must load — kept in sync with the spec. If you bump
# a version here, also update the README example so audit instructions
# match what the page actually fetches.
EXPECTED_CDN_FRAGMENTS = (
    "https://cdn.tailwindcss.com",
    "cdn.jsdelivr.net/npm/alpinejs@3",
    "cdn.jsdelivr.net/npm/marked",
    "cdn.jsdelivr.net/npm/dompurify@3",
)


def test_index_html_exists_and_size() -> None:
    assert INDEX_PATH.exists(), f"missing viewer asset: {INDEX_PATH}"
    size = INDEX_PATH.stat().st_size
    # > 5 KB ensures we shipped something substantial; < 200 KB guards
    # against an accidental commit of inlined CDN bundles.
    assert 5 * 1024 < size < 200 * 1024, f"unexpected size: {size} bytes"


def test_index_html_parses_with_lxml() -> None:
    lxml_html = pytest.importorskip("lxml.html")
    text = INDEX_PATH.read_text(encoding="utf-8")
    doc = lxml_html.fromstring(text)
    # Parsing returns *something* even on broken HTML; assert that we
    # got the expected top-level structure.
    assert doc.tag == "html"
    assert doc.xpath("//head/title")[0].text == "kaos-agents Run Inspector"
    assert doc.xpath("//body[@x-data]"), "missing root Alpine x-data binding"


def test_index_html_references_required_cdn_scripts() -> None:
    text = INDEX_PATH.read_text(encoding="utf-8")
    for fragment in EXPECTED_CDN_FRAGMENTS:
        assert fragment in text, f"missing CDN reference: {fragment}"


def test_index_html_is_alpine_wired() -> None:
    """Smoke that the SPA has enough Alpine wiring to actually react.

    Short of running the page, this is the cheapest signal that the
    Alpine layer is connected to UI events.
    """
    text = INDEX_PATH.read_text(encoding="utf-8")
    # One root x-data is sufficient — the rest of the reactivity is in
    # x-model / @click / x-show / x-for, all of which Alpine reads off
    # the root component.
    assert "x-data" in text, "missing Alpine x-data binding"
    click_handlers = text.count("@click")
    model_bindings = text.count("x-model")
    x_for_loops = text.count("x-for")
    assert click_handlers >= 5, f"too few @click handlers ({click_handlers})"
    assert model_bindings >= 3, f"too few x-model bindings ({model_bindings})"
    assert x_for_loops >= 3, f"too few x-for loops ({x_for_loops})"


def test_viewer_cli_has_main_entry_point() -> None:
    assert callable(viewer_cli.main), "viewer __main__ has no main() callable"
    # --help should exit cleanly and not raise.
    with pytest.raises(SystemExit) as exc_info:
        viewer_cli.main(["--help"])
    assert exc_info.value.code == 0


def test_viewer_cli_missing_file_returns_exit_code_2() -> None:
    rc = viewer_cli.main(["/does/not/exist/never.jsonl"])
    assert rc == 2
