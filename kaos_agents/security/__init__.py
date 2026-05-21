"""Security primitives for kaos-agents.

This package contains adversarial-robustness defenses that protect
agent-tool dispatch and prompt-context assembly from common attacks.

Modules:

- :mod:`kaos_agents.security.injection` — prompt-injection heuristic
  + XML isolation envelope. The canonical surface for wrapping
  untrusted document content before it reaches an LLM. Previously
  lived inside :mod:`kaos_agents.patterns.findings` and only protected
  the ``FindingsAgent`` path; hoisted in B0.9 so the default
  ChatAgent ingestion can use it too.
- :mod:`kaos_agents.runtime.pii_scrubber` — pre-execution PII masker
  for tool-call kwargs (B0.7). Kept under ``runtime/`` for
  historical-locality reasons (called from ``tool_bridge.executor``).

Both modules are intentionally pure-function / dataclass-only — they
do not touch the runtime or settings and can be imported by any layer
without import-cycle hazards.
"""

from __future__ import annotations

from kaos_agents.security.injection import (
    INJECTION_PATTERNS,
    is_injection_suspected,
    wrap_untrusted_content,
)

__all__ = [
    "INJECTION_PATTERNS",
    "is_injection_suspected",
    "wrap_untrusted_content",
]
