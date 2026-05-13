"""Base-install smoke tests for ``kaos_agents`` (KC17-P0-1 regression).

The ``kaos-agents`` distribution ships a small base install plus
opt-in extras (``[llm]`` → ``kaos_llm_core``; ``[api]`` → ``fastapi``;
``[mcp]`` → ``kaos_mcp``).  The README promises the base install is
importable without any extra.  Pre-KC17-P0-1 this was untrue —
``kaos_agents/__init__.py`` eagerly imported names whose home modules
required ``kaos_llm_core`` and ``fastapi``, so a clean
``pip install kaos-agents`` left ``import kaos_agents`` broken.

These tests freeze the contract in place: ``import kaos_agents``
succeeds even when the optional-extra top-level packages are
unimportable, and a curated set of always-on names is reachable
without any lazy materialisation.  Each test runs in a subprocess
with an import blocker installed via ``sys.meta_path`` so we don't
have to manage a separate virtualenv.

Regression source: ``docs/audit-02/kaos-agents.md`` (P0-1).
"""

from __future__ import annotations

import subprocess
import sys

_BLOCKER_PRELUDE = '''
import sys

class _Blocker:
    """Block imports of optional-extra top-level packages."""

    _BLOCKED = {"kaos_llm_core", "fastapi"}

    def find_spec(self, name, path, target=None):
        # Block top-level packages and any of their submodules.
        top = name.split(".", 1)[0]
        if top in self._BLOCKED:
            raise ImportError(f"simulated missing: {name}")
        return None


sys.meta_path.insert(0, _Blocker())
'''


def _run_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a clean subprocess with the import blocker installed."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER_PRELUDE + script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_kaos_agents_imports_without_optional_extras() -> None:
    """``import kaos_agents`` succeeds with [llm] and [api] unimportable.

    This is the headline KC17-P0-1 acceptance test.  If the top-level
    ``__init__.py`` regresses to eager imports of any module that pulls
    ``kaos_llm_core`` or ``fastapi`` at module load, this fails with the
    full traceback in ``stderr``.
    """
    result = _run_subprocess(
        "import kaos_agents\nprint('OK:', kaos_agents.__version__)\n",
    )
    assert result.returncode == 0, (
        "base import failed without optional extras\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert "OK:" in result.stdout, f"unexpected stdout: {result.stdout!r}"


def test_always_on_names_reachable_without_extras() -> None:
    """Always-on names (Agent, KaosEvent, …) work with no extra installed.

    These are the names whose home modules do NOT transitively touch
    ``kaos_llm_core`` or ``fastapi``.  They must be reachable as plain
    attribute access on the package — no lazy materialisation needed —
    so that consumers building on the base install (settings, event
    serialisation, trigger types, permission policy, …) have a stable
    surface to depend on.
    """
    always_on = [
        "Agent",
        "AgentPattern",
        "KaosAgent",
        "KaosAgentError",
        "KaosAgentSettings",
        "KaosEvent",
        "KaosHook",
        "KaosPattern",
        "PermissionPolicy",
        "Reversibility",
        "Span",
        "SpanPhase",
        "SpanSubject",
        "TurnSummary",
        "deserialize_event",
        "serialize_event",
    ]
    script = (
        "import kaos_agents\n"
        f"names = {always_on!r}\n"
        "for n in names:\n"
        "    obj = getattr(kaos_agents, n)\n"
        "    assert obj is not None, f'{n} resolved to None'\n"
        "print('OK')\n"
    )
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        "always-on names not reachable without optional extras\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert result.stdout.strip().endswith("OK"), f"unexpected stdout: {result.stdout!r}"


def test_optional_extra_names_raise_on_access_not_on_import() -> None:
    """``Actor`` / ``Runner`` / ``FindingsAgent`` raise ImportError on USE,
    not on package import — the whole point of the lazy table.

    Verifies that the lazy-resolution machinery correctly defers the
    ``kaos_llm_core`` dependency to the moment of attribute access.
    """
    script = (
        "import kaos_agents\n"
        "# Package import must succeed.\n"
        "print('package_ok')\n"
        "\n"
        "# But accessing each lazy name must raise ImportError —\n"
        "# never a silent None, never a surprise success.\n"
        "for name in ('Actor', 'Runner', 'FindingsAgent', 'BaseAgent'):\n"
        "    try:\n"
        "        getattr(kaos_agents, name)\n"
        "    except ImportError as exc:\n"
        "        print(f'{name}: ImportError ok')\n"
        "    else:\n"
        "        print(f'{name}: UNEXPECTED resolved')\n"
    )
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess errored:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    out = result.stdout
    assert "package_ok" in out, out
    for name in ("Actor", "Runner", "FindingsAgent", "BaseAgent"):
        assert f"{name}: ImportError ok" in out, (
            f"{name} did not raise ImportError as expected:\n{out}"
        )


def test_actor_lazy_resolution_emits_install_hint() -> None:
    """``kaos_agents.Actor`` access surfaces a ``pip install`` hint.

    The lazy ``__getattr__`` in :mod:`kaos_agents.action` wraps the
    underlying ``ImportError`` with an actionable install hint that
    points at the ``[llm]`` extra.  Without this hint, the failure mode
    is a raw module-not-found that doesn't tell the consumer how to
    fix it.
    """
    script = (
        "import kaos_agents\n"
        "try:\n"
        "    kaos_agents.Actor\n"
        "except ImportError as exc:\n"
        "    print('ERR:', str(exc))\n"
        "else:\n"
        "    print('UNEXPECTED resolved')\n"
    )
    result = _run_subprocess(script)
    assert result.returncode == 0
    assert "kaos-agents[llm]" in result.stdout, (
        f"missing install hint in error message:\n{result.stdout}"
    )


def test_api_subpackage_imports_without_fastapi() -> None:
    """``import kaos_agents.api`` and ``api.wire`` work without ``[api]``.

    ``api/__init__.py`` previously eagerly imported ``create_app`` from
    ``api/server.py``, which imports ``fastapi``.  After the fix
    ``create_app`` is resolved lazily and the wire serialisers
    (``events_to_sse`` / ``events_to_jsonl`` / ``events_to_ws``) are
    immediately available because they have no FastAPI dependency.
    """
    script = (
        "import kaos_agents.api\n"
        "from kaos_agents.api import events_to_sse, events_to_jsonl, events_to_ws\n"
        "assert callable(events_to_sse)\n"
        "assert callable(events_to_jsonl)\n"
        "assert callable(events_to_ws)\n"
        "\n"
        "# create_app should defer the fastapi import until accessed.\n"
        "try:\n"
        "    kaos_agents.api.create_app\n"
        "except ImportError as exc:\n"
        "    msg = str(exc)\n"
        "    assert 'kaos-agents[api]' in msg, msg\n"
        "    print('create_app deferred correctly:', msg[:80])\n"
        "else:\n"
        "    raise AssertionError('create_app resolved without fastapi')\n"
    )
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"api subpackage import failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert "create_app deferred correctly:" in result.stdout, result.stdout
