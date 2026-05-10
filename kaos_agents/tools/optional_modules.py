"""Shared registration for optional kaos-* tool modules.

The chat CLI and the FastAPI/MCP serve entry point both surface
``--with-source / --with-pdf / --with-office / --with-web /
--with-citations / --with-all`` flags. Historically each surface
hand-rolled its own ``importlib.import_module`` + ``getattr``
dispatch loop, and the two drifted: chat.py used a mix of submodule
paths (``kaos_pdf.tools``, ``kaos_source.runtime.tools``) while
serve.py used top-level imports (``kaos_source``, ``kaos_pdf``,
``kaos_web``) and was missing ``--with-office`` /
``--with-citations`` / ``--with-all`` entirely.

This module is the single source of truth: every kaos-* tool module
that opts into ``kaos-agent[s]`` registration appears in
``OPTIONAL_MODULES`` once, and both surfaces call
:func:`register_optional_modules`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OptionalModule:
    """Declarative spec for one optionally-registered tool module."""

    cli_flag: str
    """The argparse flag name with leading dashes (e.g. ``--with-pdf``)."""

    namespace_attr: str
    """The argparse Namespace attribute name (e.g. ``with_pdf``)."""

    package: str
    """Top-level package import path (e.g. ``kaos_pdf``).

    The top-level package is preferred over a submodule because every
    KAOS tool-bearing package re-exports its ``register_*_tools``
    helper at the top level — that is the public contract.
    """

    register_fn: str
    """Function name to invoke (e.g. ``register_pdf_tools``)."""

    label: str
    """Short human-readable label (e.g. ``PDF`` / ``source``)."""


OPTIONAL_MODULES: tuple[OptionalModule, ...] = (
    OptionalModule(
        cli_flag="--with-source",
        namespace_attr="with_source",
        package="kaos_source",
        register_fn="register_source_tools",
        label="source",
    ),
    OptionalModule(
        cli_flag="--with-pdf",
        namespace_attr="with_pdf",
        package="kaos_pdf",
        register_fn="register_pdf_tools",
        label="PDF",
    ),
    OptionalModule(
        cli_flag="--with-office",
        namespace_attr="with_office",
        package="kaos_office",
        register_fn="register_office_tools",
        label="office",
    ),
    OptionalModule(
        cli_flag="--with-web",
        namespace_attr="with_web",
        package="kaos_web",
        register_fn="register_web_tools",
        label="web",
    ),
    OptionalModule(
        cli_flag="--with-citations",
        namespace_attr="with_citations",
        package="kaos_citations",
        register_fn="register_citations_tools",
        label="citations",
    ),
)


LoadedCallback = Callable[[OptionalModule, int], None]
SkippedCallback = Callable[[OptionalModule, BaseException], None]


def _default_loaded(spec: OptionalModule, n: int) -> None:
    print(f"  Loaded {spec.package}: {n} {spec.label} tools", file=sys.stderr)


def _default_skipped(spec: OptionalModule, exc: BaseException) -> None:
    print(f"  (skip) {spec.package}: {exc}", file=sys.stderr)


def register_optional_modules(
    runtime: Any,
    args: Any,
    *,
    on_loaded: LoadedCallback | None = None,
    on_skipped: SkippedCallback | None = None,
) -> int:
    """Register every optional kaos-* tool module the user opted into.

    For each :data:`OPTIONAL_MODULES` entry, if ``args.with_X`` (or the
    blanket ``args.with_all``) is truthy, import the top-level package
    and call its ``register_*_tools(runtime)`` function. Missing
    optional packages and missing register functions are reported via
    ``on_skipped`` (default: stderr) and registration continues — the
    helper never raises for a missing optional dep.

    Args:
        runtime: A ``KaosRuntime`` instance.
        args: argparse Namespace (or any object with the expected
            ``with_*`` attributes).
        on_loaded: Callback for successful registration. Receives the
            spec and the number of tools registered. Defaults to a
            stderr print.
        on_skipped: Callback when a package can't be loaded. Receives
            the spec and the exception. Defaults to a stderr print.

    Returns:
        Total number of tools registered across all enabled modules.
    """
    loaded_cb = on_loaded or _default_loaded
    skipped_cb = on_skipped or _default_skipped

    with_all = bool(getattr(args, "with_all", False))
    total = 0

    for spec in OPTIONAL_MODULES:
        enabled = with_all or bool(getattr(args, spec.namespace_attr, False))
        if not enabled:
            continue

        try:
            mod = importlib.import_module(spec.package)
        except ImportError as exc:
            skipped_cb(spec, exc)
            continue

        fn = getattr(mod, spec.register_fn, None)
        if fn is None:
            skipped_cb(
                spec,
                AttributeError(
                    f"{spec.package} has no `{spec.register_fn}` "
                    "(did the package's public API change?)"
                ),
            )
            continue

        try:
            n = fn(runtime)
        except Exception as exc:
            skipped_cb(spec, exc)
            continue

        total += n
        loaded_cb(spec, n)

    return total


def add_optional_module_flags(parser: Any, *, include_with_all: bool = True) -> None:
    """Wire every optional-module flag onto an argparse parser.

    Both the chat CLI and the serve CLI use this so the two surfaces
    can never drift on flag coverage.
    """
    for spec in OPTIONAL_MODULES:
        parser.add_argument(spec.cli_flag, action="store_true")
    if include_with_all:
        parser.add_argument(
            "--with-all",
            action="store_true",
            help="Register all optional tool modules",
        )


__all__ = [
    "OPTIONAL_MODULES",
    "OptionalModule",
    "add_optional_module_flags",
    "register_optional_modules",
]
