"""``@hook`` decorator — Tier-1 on-ramp for :class:`KaosHook`.

Wraps a plain async function as a :class:`FunctionHook` (subclass of
:class:`KaosHook`) and auto-registers it into
:data:`kaos_agents.registry.hook_registry.default_hook_registry`.

The decorator infers the hook's ``listens_to`` set from the type
annotation of the function's first parameter — pass an
:class:`IntentClassified` and the hook listens for ``intent_classified``;
pass a :class:`KaosEvent` (or unannotated) and it listens for all events.

Mirrors :func:`kaos_core.decorators.tool_decorator.kaos_tool` shape.

Tier-1 (decorator) vs Tier-2 (subclass) coexistence:

    # Tier-1 — decorator
    @hook
    async def log_intent(event: IntentClassified) -> None:
        print(f"Intent: {event.intent} (confidence={event.confidence})")

    # Tier-2 — subclass
    class IntentLogger(KaosHook):
        @classmethod
        def metadata(cls) -> HookMetadata:
            return HookMetadata(name="intent-logger", description="...",
                                listens_to=("intent_classified",))
        async def on_intent_classified(self, event: IntentClassified) -> None:
            print(f"Intent: {event.intent}")

Both produce :class:`KaosHook` instances usable identically in Runner.
"""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, get_type_hints, overload

from kaos_agents.hooks.base import HookAction, KaosHook
from kaos_agents.registry.hook_registry import default_hook_registry
from kaos_agents.types.metadata import HookMetadata

if TYPE_CHECKING:
    from kaos_agents.events import KaosEvent
    from kaos_agents.registry.hook_registry import HookRegistry


_CAMEL_TO_SNAKE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    """``IntentClassified`` -> ``intent_classified``."""
    return _CAMEL_TO_SNAKE.sub("_", name).lower()


def _infer_listens_to(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Infer ``listens_to`` from the function's first-param annotation.

    Empty tuple is the "listens to all events" sentinel — used for
    unannotated functions and functions annotated with the
    :class:`KaosEvent` base.

    Robust to ``from __future__ import annotations``: if
    ``get_type_hints`` raises because the annotation references a class
    only imported in the function's local scope, we silently treat the
    function as a catch-all listener. Callers can override with an
    explicit ``listens_to=`` kwarg in that case.
    """
    # Lazy import to dodge the events <-> decorators cycle on package load.
    from kaos_agents.base.event import KaosEvent

    try:
        hints = get_type_hints(fn)
    except Exception:
        return ()

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return ()

    first_name = params[0].name
    annotation = hints.get(first_name)
    if annotation is None:
        return ()

    # Walk to the underlying class — Annotated[X, ...], Optional[X], etc.
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        if args:
            annotation = args[0]

    if not (isinstance(annotation, type) and issubclass(annotation, KaosEvent)):
        return ()

    # KaosEvent itself = listen to everything.
    if annotation is KaosEvent:
        return ()

    return (_camel_to_snake(annotation.__name__),)


class FunctionHook(KaosHook):
    """A :class:`KaosHook` whose entire behavior is one async function.

    Constructed by the :func:`hook` decorator. The wrapped function is
    invoked from :meth:`on_event` for every dispatched event whose
    type matches the hook's ``listens_to`` set (or always, if
    ``listens_to`` is empty). Functions returning a :class:`HookAction`
    additionally drive tool-call gating via :meth:`on_tool_call_start`.
    """

    __slots__ = ("_fn", "_metadata", "_returns_action")

    def __init__(
        self,
        fn: Callable[[Any], Awaitable[Any]],
        *,
        metadata: HookMetadata,
        returns_action: bool,
    ) -> None:
        self._fn = fn
        self._metadata = metadata
        self._returns_action = returns_action

    @property
    def fn(self) -> Callable[[Any], Awaitable[Any]]:
        """The wrapped function — exposed for introspection / testing."""
        return self._fn

    @classmethod
    def metadata(cls) -> HookMetadata:
        # Default metadata if instantiated as a class (rare); instances
        # override via the per-instance metadata stored at construction.
        return HookMetadata(name="kaos-agents-function-hook", description="A function hook.")

    def metadata_instance(self) -> HookMetadata:
        """Per-instance metadata — what the registry actually sees.

        Set at decorator time from ``name`` / ``description`` /
        ``listens_to``. Stored separately because :meth:`metadata` is a
        classmethod by KaosHook convention; per-instance identity needs
        its own accessor.
        """
        return self._metadata

    async def on_event(self, event: KaosEvent) -> None:
        """Dispatch every event the hook's ``listens_to`` matches."""
        listens = self._metadata.listens_to
        if listens:
            type_name = type(event).event_type()
            if type_name not in listens:
                return
        # Fire and forget — return value used by on_tool_call_start path.
        await self._fn(event)

    async def on_tool_call_start(self, event: KaosEvent) -> HookAction:
        """Forward to the wrapped function when it returns a HookAction.

        If the wrapped fn doesn't return HookAction, default to CONTINUE.
        """
        if not self._returns_action:
            await self._fn(event)
            return HookAction.CONTINUE
        result = await self._fn(event)
        if isinstance(result, HookAction):
            return result
        return HookAction.CONTINUE


@overload
def hook(fn: Callable[[Any], Awaitable[Any]], /) -> FunctionHook: ...
@overload
def hook(
    *,
    name: str | None = None,
    description: str | None = None,
    listens_to: tuple[str, ...] | None = None,
    register: bool = True,
    registry: HookRegistry | None = None,
) -> Callable[[Callable[[Any], Awaitable[Any]]], FunctionHook]: ...


def hook(  # type: ignore[misc]
    fn: Callable[[Any], Awaitable[Any]] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    listens_to: tuple[str, ...] | None = None,
    register: bool = True,
    registry: HookRegistry | None = None,
) -> FunctionHook | Callable[[Callable[[Any], Awaitable[Any]]], FunctionHook]:
    """Decorator that wraps an async function as a :class:`FunctionHook`.

    Two call shapes:

    - Bare: ``@hook`` — uses defaults from the function signature.
    - With kwargs: ``@hook(listens_to=("turn_summary",))`` — explicit.

    Args:
        name: Hook discriminator (defaults to a kebab-cased function name
            prefixed with ``kaos-agents-fn-``).
        description: Human description (defaults to the first line of
            the function's docstring, or a fallback).
        listens_to: Event type names to filter on. ``None`` infers from
            the first-parameter annotation; ``()`` listens to all events.
        register: When ``True`` (default), auto-registers into the
            target registry. ``False`` returns the hook without
            registering — useful for tests.
        registry: Target :class:`HookRegistry`. Defaults to the
            module-level :data:`default_hook_registry`.

    Returns:
        The :class:`FunctionHook` instance.
    """

    def _wrap(_fn: Callable[[Any], Awaitable[Any]]) -> FunctionHook:
        if not inspect.iscoroutinefunction(_fn):
            raise TypeError(
                f"@hook requires an async function; "
                f"{getattr(_fn, '__name__', _fn)!r} is sync. "
                "Define with `async def`. "
                "Alternative: subclass KaosHook directly for sync logic in __init__/__slots__."
            )

        # Resolve identity.
        fn_name = getattr(_fn, "__name__", "function-hook")
        resolved_name = name or f"kaos-agents-fn-{fn_name.replace('_', '-').lower()}"
        resolved_description = description or (inspect.getdoc(_fn) or f"Hook wrapping {fn_name}")
        # Description's first line only — HookMetadata is a single-line description.
        resolved_description = resolved_description.strip().splitlines()[0]
        resolved_listens_to = listens_to if listens_to is not None else _infer_listens_to(_fn)

        # Detect whether the function returns HookAction so we know
        # whether to plumb the return value through on_tool_call_start.
        # Robust to ``from __future__ import annotations``: if
        # ``get_type_hints`` raises (e.g. because the parameter
        # annotation references a class only imported in the function's
        # local scope), fall back to a string check on the raw
        # ``__annotations__["return"]`` value.
        returns_action = False
        return_annotation: Any = _fn.__annotations__.get("return")
        try:
            hints = get_type_hints(_fn)
            return_annotation = hints.get("return", return_annotation)
        except Exception:
            pass
        if return_annotation is HookAction or return_annotation == "HookAction":
            returns_action = True

        metadata = HookMetadata(
            name=resolved_name,
            description=resolved_description,
            listens_to=tuple(resolved_listens_to),
        )

        function_hook = FunctionHook(_fn, metadata=metadata, returns_action=returns_action)
        # Patch the instance's metadata() classmethod-style accessor so
        # registry / introspection see the per-instance identity, not
        # the FunctionHook class default. The HookRegistry calls
        # ``hook.metadata()`` — bind the instance's metadata directly.
        function_hook.metadata = metadata_instance_factory(metadata)  # ty: ignore[invalid-assignment]

        if register:
            # Don't use ``registry or default_hook_registry`` — empty
            # HookRegistry is falsy via __len__, which would silently
            # route to the default. Be explicit.
            target = default_hook_registry if registry is None else registry
            target.register(function_hook, force=True)

        # Keep the wrapped function callable directly too — useful for
        # tests that want to invoke the underlying logic without going
        # through dispatch. ``update_wrapper`` typing expects a callable
        # wrapper; FunctionHook isn't formally callable but the runtime
        # contract (copying __name__/__doc__/etc.) is what matters here.
        functools.update_wrapper(function_hook, _fn, updated=())  # ty: ignore[invalid-argument-type]
        return function_hook

    # Bare-decorator path: @hook
    if fn is not None and callable(fn):
        return _wrap(fn)

    # Kwargs path: @hook(name="...", ...)
    return _wrap


def metadata_instance_factory(metadata: HookMetadata) -> Callable[[], HookMetadata]:
    """Build a closure returning the given metadata.

    Used to bind per-instance metadata onto :class:`FunctionHook`
    instances so the registry sees the decorator-supplied identity
    rather than the class default.
    """

    def _metadata() -> HookMetadata:
        return metadata

    return _metadata


__all__ = ["FunctionHook", "hook"]
