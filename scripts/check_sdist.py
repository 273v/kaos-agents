"""Release-time gate: refuse to publish an sdist that smuggles telemetry.

KC17-P0-4 (release blocker, 2026-05-13). `pyproject.toml`'s
`[tool.hatch.build.targets.sdist].exclude` is the primary wall;
this script is the second wall — it asserts the assumption holds
on the actual built artifact. CI calls it after `uv build`; a
maintainer can call it locally before tagging.

Forbidden patterns:

- ``tests/integration/runs/.*\\.jsonl`` — pre-KC16-4 live-test
  captures with unredacted LLM input/output and, in some files,
  attorney-client privileged-marker boilerplate.
- ``docs/benchmarks/_private/`` — Harvey-Lab benchmark outputs
  that contain LLM-generated deliverable text with the same
  PRIVILEGED-AND-CONFIDENTIAL / ATTORNEY-CLIENT boilerplate. The
  public numbers live in ``harvey-coc-pipeline-comparison-*.md``.
- ``tests/scratch/`` — exploratory probe scripts; not a security
  problem, but they're ``.gitignore``'d for a reason and should
  not ship.

Exit codes:

  0 — sdist is clean.
  1 — forbidden file(s) found in sdist.
  2 — usage / IO error (e.g. sdist not found).

Usage:

  python scripts/check_sdist.py                 # auto-discovers dist/*.tar.gz
  python scripts/check_sdist.py path/to/sdist.tar.gz
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path

# Each pattern is matched against the *member name inside the tarball*
# (i.e. ``kaos_agents-0.1.0a1/tests/integration/runs/2026-05-11/...``).
# Anchoring to the project-name prefix is intentional — it ensures a
# rogue ``tests/integration/runs/...`` outside the project root would
# also trip the check.
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unredacted telemetry JSONL (KC16-4 / KC17-P0-4)",
        re.compile(r"^[^/]+/tests/integration/runs/.+\.jsonl$"),
    ),
    (
        "private benchmark deliverables with privileged markers",
        re.compile(r"^[^/]+/docs/benchmarks/_private/"),
    ),
    (
        "throwaway scratch probes (tests/scratch/)",
        re.compile(r"^[^/]+/tests/scratch/"),
    ),
    (
        "stray JSONL fixtures under tests/ (defense in depth)",
        # This intentionally overlaps with the telemetry pattern
        # above so the reported reason is the more specific one
        # when the file matches both.
        re.compile(r"^[^/]+/tests/.+\.jsonl$"),
    ),
)


def _discover_sdist() -> Path:
    """Find a single sdist in ``dist/``. Error if 0 or 2+."""
    here = Path(__file__).resolve().parent.parent
    dist = here / "dist"
    if not dist.is_dir():
        print(
            f"ERROR: no dist/ directory at {dist}. Run `uv build` first.",
            file=sys.stderr,
        )
        sys.exit(2)
    candidates = sorted(dist.glob("kaos_agents-*.tar.gz"))
    if not candidates:
        print(
            f"ERROR: no kaos_agents-*.tar.gz under {dist}. Run `uv build`.",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(candidates) > 1:
        print(
            "ERROR: multiple sdists found under dist/; pass an explicit path.\n"
            "Candidates:\n  " + "\n  ".join(str(p) for p in candidates),
            file=sys.stderr,
        )
        sys.exit(2)
    return candidates[0]


def check(sdist: Path) -> int:
    """Return 0 if clean, 1 if any forbidden member is present."""
    if not sdist.is_file():
        print(f"ERROR: sdist not found: {sdist}", file=sys.stderr)
        return 2
    findings: list[tuple[str, str]] = []
    with tarfile.open(sdist, mode="r:gz") as tf:
        for name in tf.getnames():
            for reason, pattern in FORBIDDEN_PATTERNS:
                if pattern.match(name):
                    findings.append((reason, name))
                    break  # report first matching reason only
    if findings:
        print(
            f"FAIL: sdist {sdist.name} contains {len(findings)} forbidden file(s).",
            file=sys.stderr,
        )
        # Group by reason for readable output.
        by_reason: dict[str, list[str]] = {}
        for reason, name in findings:
            by_reason.setdefault(reason, []).append(name)
        for reason, names in by_reason.items():
            print(f"\n  [{reason}]  ({len(names)} file(s))", file=sys.stderr)
            for name in names[:5]:
                print(f"    {name}", file=sys.stderr)
            if len(names) > 5:
                print(f"    ... and {len(names) - 5} more", file=sys.stderr)
        print(
            "\nThis is a release-blocker. Update "
            "pyproject.toml [tool.hatch.build.targets.sdist].exclude "
            "to drop these patterns, then rebuild and re-run this check.\n"
            "See docs/audit-02/kaos-agents.md (P0-4) for context.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: sdist {sdist.name} contains no forbidden files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a built kaos-agents sdist contains no telemetry "
        "or privileged-marker fixtures (KC17-P0-4)."
    )
    parser.add_argument(
        "sdist",
        nargs="?",
        default=None,
        help="Path to the sdist .tar.gz. If omitted, auto-discovers a single "
        "kaos_agents-*.tar.gz under ./dist/.",
    )
    args = parser.parse_args(argv)
    sdist = Path(args.sdist).resolve() if args.sdist else _discover_sdist()
    return check(sdist)


if __name__ == "__main__":
    sys.exit(main())
