"""NDA-batch review quickstart — the public-facing kaos-agents demo.

Loads 5 mutual NDAs from a deal-room and runs a
:class:`~kaos_agents.patterns.findings.FindingsAgent` over the corpus to
surface governing-law deviations, term-length variance, confidentiality-
period drift, signature-block template artifacts, and other deviations
a careful senior associate would flag.

Run with::

    python -m kaos_agents.examples.nda_review.quickstart

The fixture NDAs ship as package data under ``ndas/`` and are reachable
via :func:`importlib.resources.files` from any install (editable or
wheel). See ``ndas/README.md`` for the per-file provenance manifest.
"""
