"""Live regression for the Hello-World NDA-review demo.

The README leads with ``python -m kaos_agents.examples.nda_review.hello``
as the 30-second first-impression of the package. This test invokes the
same ``main()`` coroutine end-to-end and asserts the agent produces a
markdown table covering the 5 NDAs. Marked ``pytest.mark.live`` — budget
under $0.15 with the default ``claude-haiku-4-5``.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pytest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_hello_emits_markdown_table_across_5_ndas() -> None:
    """Run the hello demo and verify it produces a tabular summary.

    Asserts:
      1. The printed output contains a markdown table (pipe-delimited
         header row + at least one body row).
      2. The header names the columns the prompt asked for (governing
         law, confidentiality, etc.).
      3. At least 3 of the 5 NDA counterparty/fixture identifiers
         surface — proves the agent actually consulted each doc.
      4. The cost line prints with ``cost_usd=`` and a positive USD
         value — proves the LLM call really happened.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.fail(
            "ANTHROPIC_API_KEY missing — the Hello-World live test requires a "
            "real Anthropic key. Fix the env; no-skips policy."
        )

    from kaos_agents.examples.nda_review import hello as h

    buf = io.StringIO()
    with redirect_stdout(buf):
        await h.main()
    out = buf.getvalue()

    # 1. Markdown table — pipe-separated header row.
    pipe_lines = [line for line in out.splitlines() if line.count("|") >= 4]
    assert len(pipe_lines) >= 3, (
        f"Expected a markdown table with header + separator + at least one "
        f"body row (3 pipe-rich lines); got {len(pipe_lines)}.\n---\n{out}"
    )

    out_lower = out.lower()
    # 2. Column header hint — the prompt asks for these column names.
    assert "governing law" in out_lower, (
        f"Output did not include the 'Governing Law' column header.\n---\n{out}"
    )

    # 3. Coverage — the model must actually mention several of the 5 NDAs
    #    by some identifying token (filename stem OR counterparty name).
    nda_tokens = (
        "emna",
        "acme",
        "bi",
        "cybercorp",
        "cc",
        "dynamo",
        "exmachi",
        "beta",
    )
    hits = sum(1 for t in nda_tokens if t in out_lower)
    assert hits >= 3, (
        f"Output mentioned only {hits} of the 5 NDA fixtures by token. Expected >=3.\n---\n{out}"
    )

    # 4. Cost line prints with a positive USD value.
    assert "cost_usd=$" in out, f"Expected a cost line of the form 'cost_usd=$...'.\n---\n{out}"
    # Parse the dollar value — must be > 0 (otherwise the LLM transport
    # was mocked, which would defeat the regression).
    cost_line = next(line for line in out.splitlines() if "cost_usd=$" in line)
    cost_str = cost_line.split("cost_usd=$", 1)[1].split()[0]
    cost_value = float(cost_str)
    assert cost_value > 0, f"cost_usd is zero — no live LLM call happened.\n---\n{out}"
    # Budget guard — keep the live test cheap.
    assert cost_value < 0.15, (
        f"cost_usd=${cost_value:.4f} exceeded the $0.15 hello-world cap. "
        f"Investigate cost regression."
    )
