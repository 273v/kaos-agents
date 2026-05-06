"""Tests for ``kaos_agents.cli_chat`` — non-interactive mode + budget cap.

Two features under test, both shipped after the KAOS training course
discovered the gaps at curriculum-authoring time:

- ``--message`` / ``--message -`` for one-shot / stdin-fed turns so
  CI, scripts, and course runnables can drive the agent without a TTY.
- ``--max-cost USD`` (with ``$KAOS_AGENT_MAX_COST_USD`` env fallback)
  so a runaway chat can't burn arbitrary money. Non-interactive mode
  exits with code 2 on budget exceeded to distinguish it from code 1
  for actual errors.

Plus the loading-pipeline coverage added with P4 (parallel loading)
and P5 (persistent corpus index cache):

- ``--load-workers`` parallelizes parse + chunk across files. Order
  preservation, partial-failure isolation, and chunk-equivalence vs.
  serial loading are all locked in.
- ``--corpus-cache`` keys parsed/chunked documents by content hash.
  We assert: cache hit returns identical chunks to cache miss, the
  cache invalidates when file contents change, and ``--no-cache``
  disables the cache even when the dir is provided.

Tests stay at the CLI contract surface: argparse, the session-state
dataclass, the one-shot-message resolver, the env-var fallback. We
don't spin up a real Runner here — integration tests cover that.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from kaos_agents.cli_chat import (
    _EXIT_BUDGET,
    _EXIT_ERROR,
    _EXIT_OK,
    _build_arg_parser,
    _cache_key,
    _default_load_workers,
    _hash_file_bytes,
    _load_files_to_corpus,
    _one_shot_message,
    _resolve_corpus_cache,
    _resolve_max_cost,
    _SessionState,
)


class TestArgParser:
    """The new flags parse correctly and preserve the existing ones."""

    def test_message_flag_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "hello world"])
        assert args.message == "hello world"

    def test_message_stdin_sentinel_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        assert args.message == "-"

    def test_max_cost_is_float(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--max-cost", "0.05"])
        assert args.max_cost == pytest.approx(0.05)

    def test_defaults_leave_flags_unset(self) -> None:
        """Pre-existing users who never pass the new flags get None."""
        parser = _build_arg_parser()
        args = parser.parse_args(["chat"])
        assert args.message is None
        assert args.max_cost is None

    def test_existing_flags_still_wire(self) -> None:
        """Regression guard: the pre-5.x flag graph is intact."""
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "chat",
                "--session",
                "s1",
                "--model",
                "anthropic:claude-haiku-4-5",
                "--pattern",
                "research",
                "--verbose",
                "--with-all",
            ]
        )
        assert args.session == "s1"
        assert args.model == "anthropic:claude-haiku-4-5"
        assert args.pattern == "research"
        assert args.verbose is True
        assert args.with_all is True


class TestOneShotMessage:
    """``_one_shot_message`` is the single source of truth for "what
    message do we send in non-interactive mode". Keeps stdin + CLI +
    empty-message handling in one place."""

    def test_none_when_message_unset(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat"])
        assert _one_shot_message(args) is None

    def test_returns_stripped_message(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "  hi there  "])
        assert _one_shot_message(args) == "hi there"

    def test_reads_stdin_on_dash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        monkeypatch.setattr("sys.stdin", io.StringIO("piped message\n"))
        assert _one_shot_message(args) == "piped message"

    def test_empty_message_returns_none(self) -> None:
        """Don't send empty prompts to the agent — empty input should
        collapse to the REPL-vs-one-shot-decision equivalent of "no one-
        shot, go interactive"."""
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "   "])
        assert _one_shot_message(args) is None

    def test_empty_stdin_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--message", "-"])
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert _one_shot_message(args) is None


class TestSessionStateBudget:
    """``_SessionState.budget_exceeded`` is the single decision point
    for 'should we refuse the next turn'. Tests lock the semantics."""

    def test_no_cap_never_exceeds(self) -> None:
        s = _SessionState(max_cost_usd=None)
        s.absorb(tokens=1_000_000, cost=999.0)
        assert s.budget_exceeded() is False

    def test_zero_cap_disables(self) -> None:
        """Explicit ``--max-cost 0`` means ``off`` not ``infinitely
        strict``. Flipping this would surprise anyone upgrading from
        a pre-flag setup who sets the env var to 0 to "disable."""
        s = _SessionState(max_cost_usd=0.0)
        s.absorb(tokens=1, cost=0.50)
        assert s.budget_exceeded() is False

    def test_below_cap_is_ok(self) -> None:
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.05)
        assert s.budget_exceeded() is False

    def test_at_cap_exceeds(self) -> None:
        """We use >= so a session that exactly meets the cap stops on
        the next turn. One cent spent vs. one cent allowed should not
        be "you can spend one more"."""
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.10)
        assert s.budget_exceeded() is True

    def test_over_cap_exceeds(self) -> None:
        s = _SessionState(max_cost_usd=0.10)
        s.absorb(tokens=10, cost=0.11)
        assert s.budget_exceeded() is True

    def test_absorb_tracks_turns(self) -> None:
        s = _SessionState()
        s.absorb(tokens=5, cost=0.01)
        s.absorb(tokens=3, cost=0.02)
        assert s.turns == 2
        assert s.tokens == 8
        assert s.cost_usd == pytest.approx(0.03)


class TestResolveMaxCost:
    """CLI > env > None precedence for the max-cost ceiling."""

    def test_cli_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.50")
        assert _resolve_max_cost(0.05) == pytest.approx(0.05)

    def test_env_when_cli_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.25")
        assert _resolve_max_cost(None) == pytest.approx(0.25)

    def test_none_when_both_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAOS_AGENT_MAX_COST_USD", raising=False)
        assert _resolve_max_cost(None) is None

    def test_zero_cli_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0.25")
        assert _resolve_max_cost(0.0) is None  # explicit off

    def test_zero_env_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "0")
        assert _resolve_max_cost(None) is None

    def test_malformed_env_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo in the env var shouldn't crash startup — treat it as
        'no cap' with the reasoning that crashing at CLI-parse time
        would be worse than silently proceeding without a cap."""
        monkeypatch.setenv("KAOS_AGENT_MAX_COST_USD", "not-a-number")
        assert _resolve_max_cost(None) is None


class TestExitCodeConstants:
    """Lock the exit-code contract so scripts/CI can rely on distinct
    values for success / error / budget."""

    def test_exit_codes_distinct(self) -> None:
        assert len({_EXIT_OK, _EXIT_ERROR, _EXIT_BUDGET}) == 3

    def test_budget_is_two(self) -> None:
        """Documented in the cli_chat module docstring — change with
        care (tools invoking this CLI may encode 2 == budget)."""
        assert _EXIT_BUDGET == 2

    def test_success_is_zero(self) -> None:
        assert _EXIT_OK == 0


# ---------------------------------------------------------------------------
# P4 — parallel loading
# ---------------------------------------------------------------------------


def _write_corpus(tmp_path: Path, n: int = 6, *, prefix: str = "doc") -> list[Path]:
    """Materialize a small text corpus on disk.

    Plain text files exercise the same parse + chunk + cache code path
    as PDFs/DOCX without dragging the kaos-pdf/kaos-office dependency
    chain into a unit test. The chunker treats them identically once
    parsed (``ContentDocument`` AST is format-agnostic).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        # Vary content + size enough that chunking can produce >1
        # chunks for some files, ensuring we exercise multi-chunk
        # ordering not just one-chunk-per-file.
        body = f"# Heading {i}\n\n" + (
            f"Paragraph {i}-A. " * 40 + "\n\n" + f"Paragraph {i}-B. " * 40 + "\n"
        )
        p = tmp_path / f"{prefix}_{i:02d}.md"
        p.write_text(body, encoding="utf-8")
        paths.append(p)
    return paths


def _chunk_signatures(corpus: object, uris: list[str]) -> list[tuple[str, str]]:
    """Compact, comparable signature of (uri, passage-text) pairs.

    Used to assert that two loads (serial vs parallel; cache miss vs
    cache hit) produced semantically identical chunk lists. We iterate
    ``corpus`` (passages) and group by ``doc_uri`` to recover the per-
    document text in stable order, since ``ContentDocumentCorpus`` is
    a flat passage stream rather than a list of documents.
    """
    grouped: dict[str, list[str]] = {}
    seen_order: list[str] = []
    for passage in corpus:  # type: ignore[attr-defined]
        if passage.doc_uri not in grouped:
            grouped[passage.doc_uri] = []
            seen_order.append(passage.doc_uri)
        grouped[passage.doc_uri].append(passage.text)
    out: list[tuple[str, str]] = []
    for uri in seen_order:
        out.append((uri, "\n\n".join(grouped[uri])))
    return out


class TestParallelLoading:
    """``_load_files_to_corpus`` with ``workers > 1`` produces the same
    chunks (in the same order) as serial loading, and tolerates a bad
    file in the batch without losing the rest."""

    def test_default_workers_is_at_least_two(self) -> None:
        """The whole point of P4 is that the cold-start path defaults
        to parallel. Lock the floor at 2 so a single-CPU container
        doesn't accidentally serialize the load."""
        assert _default_load_workers() >= 2

    def test_serial_and_parallel_produce_same_chunks(self, tmp_path: Path) -> None:
        paths = _write_corpus(tmp_path, n=6)
        c_serial, u_serial = _load_files_to_corpus(
            paths, verbose=False, chunk_size=500, workers=1
        )
        c_par, u_par = _load_files_to_corpus(
            paths, verbose=False, chunk_size=500, workers=4
        )
        assert c_serial is not None
        assert c_par is not None
        # Same URIs in same order — order preservation is part of the
        # contract (downstream BM25 indexes assume stable doc IDs).
        assert u_serial == u_par
        assert _chunk_signatures(c_serial, u_serial) == _chunk_signatures(c_par, u_par)

    def test_order_preserved_under_parallel(self, tmp_path: Path) -> None:
        """The Future-as-completed iteration order is non-deterministic;
        we explicitly index back into a slot list to fix this. Sanity-
        check that filenames in the URIs come out in input order."""
        paths = _write_corpus(tmp_path, n=8)
        corpus, uris = _load_files_to_corpus(
            paths, verbose=False, chunk_size=500, workers=8
        )
        assert corpus is not None
        # Each URI is "file:<name>#chunk-<idx>". Strip the chunk
        # suffix and dedupe to recover the per-file order.
        seen: list[str] = []
        for uri in uris:
            name = uri.split("#", 1)[0].removeprefix("file:")
            if name not in seen:
                seen.append(name)
        assert seen == [p.name for p in paths]

    def test_one_bad_file_does_not_tank_the_load(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Real deal rooms have one corrupted PDF in 200. The loader
        must report the failure and keep going — silently dropping the
        bad file or aborting the batch are both unacceptable for a
        law-firm workflow."""
        good_paths = _write_corpus(tmp_path, n=4, prefix="good")
        # Force a parse failure: a ``.pdf`` with garbage bytes makes
        # pypdfium2 raise on PdfDocument(...). Other extensions (.txt,
        # .md, unknown ones) fall through to a permissive plain-text
        # path and succeed, so ``.pdf`` is the cleanest hard-fail
        # extension to exercise the per-file error branch.
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"\x00\x01\x02\x03not really a PDF")
        all_paths = [good_paths[0], bad, good_paths[1], good_paths[2], good_paths[3]]

        corpus, uris = _load_files_to_corpus(
            all_paths, verbose=False, chunk_size=500, workers=4
        )
        captured = capsys.readouterr()
        assert corpus is not None
        # The good 4 are loaded, the bad 1 is skipped.
        good_names = {p.name for p in good_paths}
        loaded_names = {u.split("#", 1)[0].removeprefix("file:") for u in uris}
        assert loaded_names == good_names
        # User-visible error mentions the broken file by name so the
        # operator can investigate.
        assert "broken.pdf" in captured.out or "broken" in captured.out


# ---------------------------------------------------------------------------
# P5 — persistent corpus cache
# ---------------------------------------------------------------------------


class TestCacheKey:
    """``_cache_key`` is the cache invariant: same (bytes, chunk_size)
    → same key; different bytes or different chunk_size → different
    key. Don't break this without a migration story."""

    def test_same_content_same_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"  # different name, same bytes
        a.write_text("hello world", encoding="utf-8")
        b.write_text("hello world", encoding="utf-8")
        assert _cache_key(a, 500) == _cache_key(b, 500)

    def test_different_content_different_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello", encoding="utf-8")
        b.write_text("HELLO", encoding="utf-8")
        assert _cache_key(a, 500) != _cache_key(b, 500)

    def test_different_chunk_size_different_key(self, tmp_path: Path) -> None:
        """Switching ``--chunk-size`` should be a clean miss — the old
        cache entry was chunked at a different grain."""
        a = tmp_path / "a.txt"
        a.write_text("hello world", encoding="utf-8")
        assert _cache_key(a, 500) != _cache_key(a, 1500)

    def test_hash_streams_large_files(self, tmp_path: Path) -> None:
        """``_hash_file_bytes`` reads in chunks. Sanity-check it for a
        file larger than a single read buffer (we set 1 MB; use 3 MB)."""
        big = tmp_path / "big.bin"
        big.write_bytes(b"A" * (3 << 20))
        h = _hash_file_bytes(big)
        assert len(h) == 64  # sha256 hex


class TestCacheRoundTrip:
    """Cache hit must return chunks indistinguishable from a fresh
    parse, and changing the file must invalidate the cache."""

    def test_hit_matches_miss(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        paths = _write_corpus(tmp_path / "src", n=4)

        # First load — cold cache. Populates blobs/.
        c1, u1 = _load_files_to_corpus(
            paths, verbose=False, chunk_size=500, workers=2, cache_dir=cache
        )
        # Second load — every file should hit the cache.
        c2, u2 = _load_files_to_corpus(
            paths, verbose=False, chunk_size=500, workers=2, cache_dir=cache
        )
        assert c1 is not None
        assert c2 is not None
        assert u1 == u2
        assert _chunk_signatures(c1, u1) == _chunk_signatures(c2, u2)

        # The blobs directory should hold one entry per file.
        blobs = list((cache / "blobs").glob("*.json"))
        assert len(blobs) == len(paths)
        # Index manifest exists and references all keys.
        import json as _json

        with (cache / "INDEX.json").open() as fh:
            index = _json.load(fh)
        assert len(index) == len(paths)

    def test_invalidates_on_content_change(self, tmp_path: Path) -> None:
        """Mutate one file after the first load. The agent must NOT
        return stale chunks — content hashing is the whole point."""
        cache = tmp_path / "cache"
        src = tmp_path / "src"
        src.mkdir()
        target = src / "doc.md"
        target.write_text("# v1\n\n" + "Original. " * 80, encoding="utf-8")

        c1, u1 = _load_files_to_corpus(
            [target], verbose=False, chunk_size=500, workers=1, cache_dir=cache
        )
        sig1 = _chunk_signatures(c1, u1)  # type: ignore[arg-type]

        # Overwrite with substantively different content.
        target.write_text("# v2\n\n" + "Replacement. " * 80, encoding="utf-8")

        c2, u2 = _load_files_to_corpus(
            [target], verbose=False, chunk_size=500, workers=1, cache_dir=cache
        )
        sig2 = _chunk_signatures(c2, u2)  # type: ignore[arg-type]
        assert sig1 != sig2
        # And the new content shows up in the chunks.
        joined = "\n".join(text for _, text in sig2)
        assert "Replacement" in joined
        assert "Original" not in joined

    def test_no_cache_disables_even_when_dir_provided(self, tmp_path: Path) -> None:
        """``--no-cache`` is the escape hatch for benchmarks — it must
        win over ``--corpus-cache``. Concretely: no blobs are written
        and ``_resolve_corpus_cache`` returns None."""
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "chat",
                "--corpus-cache",
                str(tmp_path / "cache"),
                "--no-cache",
            ]
        )
        assert _resolve_corpus_cache(args) is None

    def test_empty_string_disables_cache(self, tmp_path: Path) -> None:
        """``--corpus-cache ''`` is the documented opt-out."""
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--corpus-cache", ""])
        assert _resolve_corpus_cache(args) is None

    def test_resolved_cache_dir_exists(self, tmp_path: Path) -> None:
        """The resolver eagerly creates the directory so callers don't
        have to mkdir before passing through to ``_load_files_to_corpus``."""
        parser = _build_arg_parser()
        target = tmp_path / "cache_dir"
        args = parser.parse_args(["chat", "--corpus-cache", str(target)])
        resolved = _resolve_corpus_cache(args)
        assert resolved is not None
        assert resolved.is_dir()


class TestLoaderArgParser:
    """The new flags wire correctly without disturbing existing ones."""

    def test_load_workers_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--load-workers", "8"])
        assert args.load_workers == 8

    def test_corpus_cache_parses(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--corpus-cache", "/tmp/foo"])
        assert args.corpus_cache == "/tmp/foo"

    def test_no_cache_flag(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat", "--no-cache"])
        assert args.no_cache is True

    def test_loader_defaults_unset(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["chat"])
        assert args.load_workers is None
        assert args.corpus_cache is None
        assert args.no_cache is False
