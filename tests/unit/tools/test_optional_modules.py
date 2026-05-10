"""Regression tests for the shared optional-module registration helper.

These tests exist because both ``kaos_agents/cli/chat.py`` and
``kaos_agents/api/serve.py`` historically hand-rolled their own
flag→module dispatch and silently drifted (chat had office/citations,
serve did not; chat had a wrong import path for source). The shared
helper at ``kaos_agents.tools.optional_modules`` is now the only
dispatcher both surfaces use.
"""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from kaos_agents.tools.optional_modules import (
    OPTIONAL_MODULES,
    OptionalModule,
    add_optional_module_flags,
    register_optional_modules,
)


def _make_args(**overrides: bool) -> SimpleNamespace:
    """Build an argparse-shaped object with only the requested flags set."""
    base: dict[str, Any] = {spec.namespace_attr: False for spec in OPTIONAL_MODULES}
    base["with_all"] = False
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeRuntime:
    """Stand-in runtime — registration helpers ignore it for these tests."""


def test_optional_modules_table_is_authoritative() -> None:
    """Every OPTIONAL_MODULES entry must declare every required field."""
    for spec in OPTIONAL_MODULES:
        assert isinstance(spec, OptionalModule)
        assert spec.cli_flag.startswith("--with-")
        assert spec.namespace_attr == spec.cli_flag.lstrip("-").replace("-", "_")
        assert spec.package.startswith("kaos_")
        assert spec.register_fn.startswith("register_")
        assert spec.register_fn.endswith("_tools")
        assert spec.label


def test_optional_modules_table_has_no_duplicates() -> None:
    """No two entries may share a flag, attr, package, or label."""
    flags = [s.cli_flag for s in OPTIONAL_MODULES]
    attrs = [s.namespace_attr for s in OPTIONAL_MODULES]
    pkgs = [s.package for s in OPTIONAL_MODULES]
    assert len(flags) == len(set(flags))
    assert len(attrs) == len(set(attrs))
    assert len(pkgs) == len(set(pkgs))


def test_register_skips_when_no_flags_set() -> None:
    """No flags → no calls to any package's register fn."""
    loaded: list[OptionalModule] = []
    skipped: list[OptionalModule] = []
    n = register_optional_modules(
        _FakeRuntime(),
        _make_args(),
        on_loaded=lambda s, _n: loaded.append(s),
        on_skipped=lambda s, _e: skipped.append(s),
    )
    assert n == 0
    assert loaded == []
    assert skipped == []


def test_register_with_all_attempts_every_module() -> None:
    """``--with-all`` must hit every spec in OPTIONAL_MODULES."""
    attempted: list[OptionalModule] = []

    def _record(spec: OptionalModule, _n_or_exc: Any) -> None:
        attempted.append(spec)

    register_optional_modules(
        _FakeRuntime(),
        _make_args(with_all=True),
        on_loaded=_record,
        on_skipped=_record,
    )
    attempted_packages = {s.package for s in attempted}
    expected_packages = {s.package for s in OPTIONAL_MODULES}
    assert attempted_packages == expected_packages


def test_register_handles_missing_package_gracefully() -> None:
    """Importing a missing optional dep must call on_skipped, not raise."""
    skipped: list[tuple[OptionalModule, BaseException]] = []
    fake_spec = OptionalModule(
        cli_flag="--with-doesnotexist",
        namespace_attr="with_doesnotexist",
        package="kaos_doesnotexist_xyz",
        register_fn="register_doesnotexist_tools",
        label="ghost",
    )
    fake_args = SimpleNamespace(with_doesnotexist=True, with_all=False)
    # Patch OPTIONAL_MODULES via the helper's module namespace
    import kaos_agents.tools.optional_modules as opt

    original = opt.OPTIONAL_MODULES
    opt.OPTIONAL_MODULES = (fake_spec,)
    try:
        n = register_optional_modules(
            _FakeRuntime(),
            fake_args,
            on_skipped=lambda s, e: skipped.append((s, e)),
        )
    finally:
        opt.OPTIONAL_MODULES = original
    assert n == 0
    assert len(skipped) == 1
    assert skipped[0][0] is fake_spec
    assert isinstance(skipped[0][1], ImportError)


def test_register_handles_missing_register_fn() -> None:
    """A package that imports but lacks register_*_tools is skipped, not raised."""
    skipped: list[tuple[OptionalModule, BaseException]] = []
    # Use the real `os` module — it imports but has no register_os_tools.
    fake_spec = OptionalModule(
        cli_flag="--with-os",
        namespace_attr="with_os",
        package="os",
        register_fn="register_os_tools",
        label="os",
    )
    fake_args = SimpleNamespace(with_os=True, with_all=False)
    import kaos_agents.tools.optional_modules as opt

    original = opt.OPTIONAL_MODULES
    opt.OPTIONAL_MODULES = (fake_spec,)
    try:
        n = register_optional_modules(
            _FakeRuntime(),
            fake_args,
            on_skipped=lambda s, e: skipped.append((s, e)),
        )
    finally:
        opt.OPTIONAL_MODULES = original
    assert n == 0
    assert len(skipped) == 1
    assert isinstance(skipped[0][1], AttributeError)


def test_register_invokes_register_fn_and_returns_count() -> None:
    """When the register fn is present, it's called with the runtime and total accumulates."""
    import sys
    import types

    import kaos_agents.tools.optional_modules as opt

    # Build a fake module with a register_fake_tools that records calls.
    calls: list[Any] = []

    def fake_register(runtime: Any) -> int:
        calls.append(runtime)
        return 7

    fake_mod = types.ModuleType("kaos_fake_test_xyz")
    # Module attribute set via __dict__ — ruff and ty both accept this and
    # neither setattr() nor `mod.attr = ...` survive both checkers in CI.
    fake_mod.__dict__["register_fake_tools"] = fake_register
    sys.modules["kaos_fake_test_xyz"] = fake_mod

    fake_spec = OptionalModule(
        cli_flag="--with-fake",
        namespace_attr="with_fake",
        package="kaos_fake_test_xyz",
        register_fn="register_fake_tools",
        label="fake",
    )
    runtime = _FakeRuntime()
    args = SimpleNamespace(with_fake=True, with_all=False)
    original = opt.OPTIONAL_MODULES
    opt.OPTIONAL_MODULES = (fake_spec,)
    try:
        n = register_optional_modules(runtime, args)
    finally:
        opt.OPTIONAL_MODULES = original
        sys.modules.pop("kaos_fake_test_xyz", None)
    assert n == 7
    assert calls == [runtime]


@pytest.mark.parametrize("spec", OPTIONAL_MODULES)
def test_each_declared_package_resolves_or_is_optional(spec: OptionalModule) -> None:
    """Every declared spec either imports cleanly + has the register fn,
    or raises ImportError (acceptable for optional deps not in the install)."""
    try:
        mod = importlib.import_module(spec.package)
    except ImportError:
        # Optional dep not installed in this environment — register helper
        # is required to handle this gracefully (covered by the
        # missing-package test above).
        return
    fn = getattr(mod, spec.register_fn, None)
    assert fn is not None, (
        f"{spec.package} imported but exposes no `{spec.register_fn}` — "
        "either the public API changed or OPTIONAL_MODULES is stale."
    )
    assert callable(fn)


def test_add_optional_module_flags_wires_every_spec() -> None:
    """The argparse helper must register every spec's flag + ``--with-all``."""
    parser = argparse.ArgumentParser()
    add_optional_module_flags(parser)
    args = parser.parse_args([])
    for spec in OPTIONAL_MODULES:
        assert hasattr(args, spec.namespace_attr)
        assert getattr(args, spec.namespace_attr) is False
    assert hasattr(args, "with_all")
    assert args.with_all is False


@pytest.mark.parametrize("spec", OPTIONAL_MODULES)
def test_each_optional_module_has_pyproject_extra(spec: OptionalModule) -> None:
    """Every entry in OPTIONAL_MODULES must have a matching pyproject extra.

    The extra name follows the spec's ``label`` attribute (lowercased).
    Without this contract, ``--with-X`` flags can silently fail for
    end users who never installed ``kaos-agents[X]``.

    The ``full`` extra must also pull every spec's package transitively.
    """
    import tomllib  # stdlib in Python 3.11+, kaos-agents requires 3.13+
    from pathlib import Path

    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extras = data["project"]["optional-dependencies"]

    extra_name = spec.label.lower()
    assert extra_name in extras, (
        f"OPTIONAL_MODULES has '{spec.package}' (label={spec.label!r}) but "
        f"pyproject.toml has no `[project.optional-dependencies]` entry "
        f"named '{extra_name}'. Users cannot `pip install "
        f"'kaos-agents[{extra_name}]'` to enable this tool module."
    )

    extra_packages = extras[extra_name]
    pkg_dist_name = spec.package.replace("_", "-")
    assert any(pkg_dist_name in dep for dep in extra_packages), (
        f"Extra `{extra_name}` does not pull `{pkg_dist_name}`; "
        f"declared deps are: {extra_packages!r}"
    )

    # `full` must transitively pull every per-tool extra so
    # `pip install 'kaos-agents[full]'` is a one-shot install.
    assert "full" in extras, "Missing `full` meta-extra in pyproject"
    full_deps = extras["full"]
    assert any(extra_name in dep for dep in full_deps), (
        f"`full` extra must include `{extra_name}` (current full = {full_deps!r})"
    )


def test_add_optional_module_flags_parses_each_individually() -> None:
    """Each --with-X flag must flip exactly its own attr."""
    parser = argparse.ArgumentParser()
    add_optional_module_flags(parser)
    for spec in OPTIONAL_MODULES:
        args = parser.parse_args([spec.cli_flag])
        for other in OPTIONAL_MODULES:
            expected = other.namespace_attr == spec.namespace_attr
            assert getattr(args, other.namespace_attr) is expected
