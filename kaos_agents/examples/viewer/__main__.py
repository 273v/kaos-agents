"""Launch the kaos-agents JSONL telemetry viewer in the default browser.

    python -m kaos_agents.examples.viewer

By default opens the bundled ``index.html`` with no file pre-loaded; the
user drags-and-drops their JSONL. Optional positional arg pre-loads a
file by spinning up a local :mod:`http.server` in a temp directory that
contains a copy of ``index.html`` plus the JSONL, and passing the file
to the page via ``?file=<basename>``.

    python -m kaos_agents.examples.viewer /tmp/kaos-runs/subprocess-*.jsonl

Press Ctrl-C to stop the server.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socket
import socketserver
import sys
import tempfile
import webbrowser
from importlib.resources import files as _resource_files
from pathlib import Path
from urllib.parse import quote

INDEX_HTML = _resource_files(__package__).joinpath("index.html")


def _find_free_port(start: int, attempts: int = 10) -> int:
    """Return the first port in ``[start, start+attempts)`` that bind() accepts."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No free port in [{start}, {start + attempts}). "
        f"Pick another --port or stop the conflicting process."
    )


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that suppresses the per-request access log."""

    def log_message(self, format: str, *args: object) -> None:
        del format, args  # quiet: parent signature requires the name `format`
        return None


def _serve(serve_dir: Path, port: int, open_browser: bool, url_path: str) -> None:
    """Run a blocking HTTP server in ``serve_dir`` until Ctrl-C."""
    handler_factory = functools.partial(_QuietHandler, directory=str(serve_dir))

    with socketserver.TCPServer(("127.0.0.1", port), handler_factory) as httpd:
        url = f"http://127.0.0.1:{port}/{url_path}"
        print(f"kaos-agents Run Inspector → {url}")
        print("Press Ctrl-C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kaos_agents.examples.viewer",
        description="Launch the kaos-agents JSONL telemetry viewer.",
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        help="Optional JSONL recording to pre-load. If omitted, the viewer "
        "opens empty and waits for a drag-and-drop / paste.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local HTTP port to bind. Tries up to +10 if busy. Default: 8765.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL but do not open a browser window.",
    )
    args = parser.parse_args(argv)

    index_text = INDEX_HTML.read_text(encoding="utf-8")

    if args.jsonl is None:
        # Fast path — no fixture, just open the file:// URL.
        with tempfile.TemporaryDirectory(prefix="kaos-viewer-") as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "index.html").write_text(index_text, encoding="utf-8")
            port = _find_free_port(args.port)
            _serve(tmp_path, port, not args.no_open, "index.html")
        return 0

    jsonl_path = args.jsonl.expanduser().resolve()
    if not jsonl_path.exists():
        print(
            f"error: file not found: {jsonl_path}\n"
            f"hint: run 'KAOS_LLM_CORE_RECORDER_DIR=/tmp/kaos-runs python -m "
            f"kaos_agents.examples.nda_review.hello' to generate one.",
            file=sys.stderr,
        )
        return 2
    if not jsonl_path.is_file():
        print(f"error: not a regular file: {jsonl_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="kaos-viewer-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "index.html").write_text(index_text, encoding="utf-8")
        fixture_name = jsonl_path.name
        shutil.copy2(jsonl_path, tmp_path / fixture_name)
        port = _find_free_port(args.port)
        url_path = f"index.html?file={quote(fixture_name)}"
        _serve(tmp_path, port, not args.no_open, url_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
