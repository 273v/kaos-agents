"""Benchmark utilities for kaos-agents.

This subpackage holds support code for the benchmarks under
``tests/benchmarks/``. The benchmarks themselves remain executable
scripts; this module exists so the scoring helpers can be imported
and unit-tested without going through ``sys.path`` gymnastics.
"""

from __future__ import annotations

from kaos_agents.benchmarks.llm_judge import (
    JudgeUnavailable,
    JudgeVerdict,
    llm_judge,
)

__all__ = [
    "JudgeUnavailable",
    "JudgeVerdict",
    "llm_judge",
]
