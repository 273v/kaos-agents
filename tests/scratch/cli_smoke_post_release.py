"""Post-release acceptance smoke: kaos-agent chat on a real NDA.

Run after kaos-office 0.1.3 + kaos-pdf 0.1.4 + kaos-agents 0.1.19/0.1.20
are on PyPI and the venv is `uv sync`'d.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

if "ANTHROPIC_API_KEY" in os.environ and "SIMULATOR_ANTHROPIC_API_KEY" not in os.environ:
    os.environ["SIMULATOR_ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

NDA_PATH = Path.home() / "Documents" / "NDA" / "EMNA Mutual NDA.docx"
EXPECTED_FLOORS = {
    "kaos_agents": "0.1.19",
    "kaos_office": "0.1.3",
    "kaos_pdf": "0.1.4",
}


def _vt(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:3])


def main() -> int:
    if not NDA_PATH.exists():
        print(f"FAIL: NDA fixture not found at {NDA_PATH}")
        return 1
    for pkg, floor in EXPECTED_FLOORS.items():
        try:
            mod = __import__(pkg)
        except ImportError as exc:
            print(f"FAIL: {pkg} not installed: {exc}")
            return 1
        v = getattr(mod, "__version__", "0.0.0")
        if _vt(v) < _vt(floor):
            print(f"FAIL: {pkg} {v} below floor {floor}")
            return 1
        print(f"[versions] {pkg}={v} (floor={floor})")

    prompt = (
        "Read the attached NDA and tell me the survival period for the "
        "confidentiality obligation. Quote the operative clause verbatim."
    )
    cmd = [
        "kaos-agent",
        "chat",
        "--pattern",
        "chat",
        "--message",
        prompt,
        "--files",
        str(NDA_PATH),
        "--max-cost",
        "0.50",
        "--session",
        "smoke-post-release",
        "--verbose",
    ]
    print(f"\n[+] running: {shlex.join(cmd)}\n")
    return subprocess.run(
        cmd,
        env={
            **os.environ,
            "KAOS_AGENT_DEFAULT_LLM_MODEL": "anthropic:claude-sonnet-4-6",
            "KAOS_AGENT_PLANNING_LLM_MODEL": "anthropic:claude-sonnet-4-6",
        },
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
