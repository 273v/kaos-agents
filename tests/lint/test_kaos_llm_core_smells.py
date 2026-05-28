"""Guardrail — mechanical enforcement of the kaos-llm-core design audit.

Iteration 14 of ``kaos-modules/docs/audits/2026-05-24-kaos-agents-prompt-smells.md``.

This module is a CI-level lint. It walks the kaos-agents source tree at
test time and asserts the patterns established by Iterations 4-7:

- **S8** — every production ``Call(SigClass, model=...)`` instantiation passes
  ``examples=load_examples("<name>")``. The single documented hold-out
  is ``intent/extractor.py:IntentSignature`` (deferred — typed Goal /
  Constraint / Ambiguity outputs need a richer round-trip design).
- **S14** — no inline ``Example(inputs=..., outputs=...)`` construction
  inside ``kaos_agents/`` production code; example data must live under
  ``kaos_agents/_assets/examples/<sig>.toml``.
- **S7** — no ``self._instructions = ...`` assignments outside
  ``__init__`` (the canonical override path is
  ``BaseAgent.override_instructions`` ContextVar, see Iteration 7).
- **TOML loader contract** — every ``_assets/examples/*.toml`` file
  round-trips through ``load_examples()`` and contains at least one
  ``[[example]]`` row with non-empty ``inputs`` + ``outputs``.

When this lint fires on a new file, the fix is mechanical: lift the
inline data to a TOML pool, wire ``examples=load_examples("...")``, or
switch the mutation to the ContextVar override.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "kaos_agents"
_ASSETS_ROOT = _SRC_ROOT / "_assets" / "examples"

# Files that are allowed to define ``Call(...)`` without ``examples=`` for
# documented design reasons. Keep this list short — each entry must cite
# the audit doc rationale.
_S8_DEFERRED_FILES: frozenset[str] = frozenset(
    {
        # Typed structured outputs (Goal / Constraint / Ambiguity) don't
        # round-trip cleanly through hand-written TOML — deferred per
        # the audit doc, Iteration 5 row 25.
        "kaos_agents/intent/extractor.py",
        # PR-1b (2026-05-28 dynamic-deliverable-schema architecture):
        # the ``Extract_<schema_id>_v<n>`` Signature is RUNTIME-BUILT
        # from whatever schema the LLM designer proposes for the
        # current user question + corpus. There's no fixed Signature
        # class to author static TOML examples against. A future
        # iteration may add examples keyed on (column_type, label)
        # pairs at runtime, but the static pre-load model doesn't
        # apply. Plan §6.1 + §7.
        "kaos_agents/tools/design_extraction.py",
    }
)

# The _examples loader module itself contains a docstring example of
# Call(...) usage. Skip docstring matches.
_S8_DOCSTRING_FILES: frozenset[str] = frozenset(
    {
        "kaos_agents/_examples.py",
    }
)


def _python_files_under(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.parts
    ]


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO_ROOT))


# ─── S8: every Call(SigClass, ...) passes examples= ─────────────────────


class _CallSiteVisitor(ast.NodeVisitor):
    """Find every ``Call(SigClass, ...)`` invocation and record whether
    it passes ``examples=``.
    """

    def __init__(self) -> None:
        self.sites: list[tuple[int, bool]] = []  # (lineno, has_examples)

    def visit_Call(self, node: ast.Call) -> None:
        is_call_ctor = (isinstance(node.func, ast.Name) and node.func.id == "Call") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "Call"
        )
        if is_call_ctor and node.args:
            # First positional arg = Signature class. Look for examples= kw.
            has_examples = any(
                isinstance(kw, ast.keyword) and kw.arg == "examples" for kw in node.keywords
            )
            self.sites.append((node.lineno, has_examples))
        self.generic_visit(node)


def test_s8_every_call_site_passes_examples() -> None:
    """Every production ``Call(Sig, model=)`` must pass ``examples=``."""
    violations: list[str] = []
    for path in _python_files_under(_SRC_ROOT):
        rel = _rel(path)
        if rel in _S8_DEFERRED_FILES or rel in _S8_DOCSTRING_FILES:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:  # pragma: no cover - defensive
            pytest.fail(f"{rel}: failed to parse ({exc})")
        v = _CallSiteVisitor()
        v.visit(tree)
        for lineno, has_examples in v.sites:
            if not has_examples:
                violations.append(f"{rel}:{lineno} — Call(...) without examples=")

    assert not violations, (
        "S8 violations — every Call(Signature, model=) must pass "
        "examples=load_examples('...').\n"
        "Lift example data to kaos_agents/_assets/examples/<name>.toml and "
        "wire it via the loader.\n" + "\n".join(f"  - {v}" for v in violations)
    )


# ─── S14: no inline Example(...) construction in production code ────────


class _InlineExampleVisitor(ast.NodeVisitor):
    """Find any ``Example(inputs=..., outputs=...)`` construction."""

    def __init__(self) -> None:
        self.sites: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        is_example_ctor = (isinstance(node.func, ast.Name) and node.func.id == "Example") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "Example"
        )
        if is_example_ctor and any(
            kw.arg in ("inputs", "outputs") for kw in node.keywords if isinstance(kw, ast.keyword)
        ):
            self.sites.append(node.lineno)
        self.generic_visit(node)


def test_s14_no_inline_example_construction() -> None:
    """Production code must NOT construct ``Example(inputs=..., outputs=...)`` inline.

    Example data lives in ``kaos_agents/_assets/examples/<sig>.toml`` and
    is loaded via :func:`kaos_agents._examples.load_examples`. The TOML
    is the canonical source of truth that optimizers (MIPRO,
    BootstrapFewShot) can read AND write back without touching Python.

    Documented exceptions (each must cite a rationale):
    - ``kaos_agents/_examples.py`` — the loader itself constructs
      ``Example(inputs, outputs)`` once at the TOML→Pydantic boundary.
    - ``kaos_agents/patterns/router.py`` — wraps *user-supplied*
      classifier examples (``specialist.examples: tuple[str, ...]``)
      into ``Example`` records at runtime. The data is not hardcoded;
      it comes from the caller's :class:`RouterAgent` config. Lifting
      this to TOML would defeat the user-supplied-data contract.
    """
    inline_exceptions: frozenset[str] = frozenset(
        {
            "kaos_agents/_examples.py",
            "kaos_agents/patterns/router.py",
        }
    )
    violations: list[str] = []
    for path in _python_files_under(_SRC_ROOT):
        if _rel(path) in inline_exceptions:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        v = _InlineExampleVisitor()
        v.visit(tree)
        for lineno in v.sites:
            violations.append(f"{_rel(path)}:{lineno} — inline Example(inputs=, outputs=)")

    assert not violations, (
        "S14 violations — Example data must live in TOML, not Python.\n"
        "Move the rows to kaos_agents/_assets/examples/<name>.toml and load via load_examples().\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ─── S7: no ``self._instructions = ...`` reassignment outside __init__ ──


class _InstructionsMutationVisitor(ast.NodeVisitor):
    """Find ``self._instructions = ...`` assignments outside __init__.

    The canonical per-call override path is the ContextVar-backed
    :meth:`BaseAgent.override_instructions` (Iteration 7). Direct
    instance reassignment is hostile to concurrent invocations and
    flagged as a regression.
    """

    def __init__(self) -> None:
        self.sites: list[int] = []
        self._in_init: list[bool] = [False]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._in_init.append(node.name == "__init__")
        try:
            self.generic_visit(node)
        finally:
            self._in_init.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_init.append(node.name == "__init__")
        try:
            self.generic_visit(node)
        finally:
            self._in_init.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._in_init[-1]:
            return  # Allowed in __init__.
        for tgt in node.targets:
            if (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
                and tgt.attr == "_instructions"
            ):
                self.sites.append(node.lineno)
        self.generic_visit(node)


def test_s7_no_instructions_mutation_outside_init() -> None:
    """``self._instructions =`` outside ``__init__`` is forbidden.

    Use ``BaseAgent.override_instructions(augmented)`` as a contextmanager
    instead — it pushes a task-local ContextVar override, propagates
    correctly across awaits, and auto-restores on exit. See Iteration 7
    of the kaos-llm-core design audit.
    """
    violations: list[str] = []
    for path in _python_files_under(_SRC_ROOT):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        v = _InstructionsMutationVisitor()
        v.visit(tree)
        for lineno in v.sites:
            violations.append(f"{_rel(path)}:{lineno} — self._instructions = ... outside __init__")

    assert not violations, (
        "S7 violations — replace with `with BaseAgent.override_instructions(augmented):`.\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ─── TOML loader contract ────────────────────────────────────────────────


def test_every_toml_pool_round_trips() -> None:
    """Every TOML in ``_assets/examples/`` must have at least one row + valid shape."""
    pool_files = sorted(_ASSETS_ROOT.glob("*.toml"))
    assert pool_files, f"No example pools under {_ASSETS_ROOT}; loader has nothing to ground."

    failures: list[str] = []
    for path in pool_files:
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:
            failures.append(f"{path.name}: TOML parse error — {exc}")
            continue
        rows = data.get("example") or []
        if not rows:
            failures.append(f"{path.name}: no [[example]] rows")
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row.get("inputs"), dict) or not row["inputs"]:
                failures.append(f"{path.name}[{idx}]: missing/empty inputs")
            if not isinstance(row.get("outputs"), dict) or not row["outputs"]:
                failures.append(f"{path.name}[{idx}]: missing/empty outputs")

    assert not failures, "TOML pool violations:\n" + "\n".join(f"  - {f}" for f in failures)


def test_every_toml_pool_has_a_loader_call_site() -> None:
    """Each TOML pool name should appear in at least one ``load_examples("<name>")`` call.

    Orphan TOMLs are dead data that drift away from the runtime contract.
    """
    pool_names = {p.stem for p in _ASSETS_ROOT.glob("*.toml")}
    referenced: set[str] = set()
    for path in _python_files_under(_SRC_ROOT):
        text = path.read_text()
        for name in pool_names:
            if f'load_examples("{name}")' in text or f"load_examples('{name}')" in text:
                referenced.add(name)
    orphans = sorted(pool_names - referenced)
    assert not orphans, "Orphan TOML pools — no Python call site references these:\n" + "\n".join(
        f"  - {n}.toml" for n in orphans
    )
