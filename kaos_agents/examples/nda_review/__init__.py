"""Two runnable NDA-review demos shipped as package data.

- :mod:`kaos_agents.examples.nda_review.hello` — **Hello-World**
  version: defaults only, asks the agent for a markdown table of key
  terms across all 5 NDAs in a single ``ResearchAgent.turn`` call. Best
  first-impression demo; ~$0.05-0.10 on ``claude-haiku-4-5``::

      python -m kaos_agents.examples.nda_review.hello

- :mod:`kaos_agents.examples.nda_review.quickstart` — **production**
  senior-counsel deviation review with per-doc
  :class:`~kaos_agents.patterns.findings.FindingsAgent`, typed findings
  with ``block_ref`` provenance, strict ``max_cost_usd`` cap, refusal
  contract, and full audit trail. The version a partner would sign off
  on; ~$0.05-0.10 on Haiku 4.5 filter + Sonnet 4.6 synthesis::

      python -m kaos_agents.examples.nda_review.quickstart

Both ship the same 5 real mutual NDAs under ``ndas/`` (reachable via
:func:`importlib.resources.files` from any install — editable, wheel,
or sdist). See ``ndas/README.md`` for fixture provenance and
``tests/integration/ladder/fixtures/nda_ground_truth.py`` for the
lawyer-authored ground truth the live regressions assert against.
"""
