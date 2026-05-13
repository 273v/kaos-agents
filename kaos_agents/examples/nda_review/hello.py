"""Hello-World NDA review — defaults-only summary table across 5 NDAs.

``pip install 'kaos-agents[llm,office]'`` + ``ANTHROPIC_API_KEY``, then
``python -m kaos_agents.examples.nda_review.hello``. ~$0.05-0.10 with
``claude-haiku-4-5``. For production review see ``quickstart.py`` here.
"""

from __future__ import annotations

import asyncio
from importlib.resources import files as _resource_files

from kaos_core.registry.container import KaosRuntime

from kaos_agents import ResearchAgent, SessionMemory, SessionStore

NDAS_DIR = _resource_files("kaos_agents.examples.nda_review").joinpath("ndas")


async def main() -> None:
    from kaos_content.serializers.markdown import serialize_markdown
    from kaos_office import parse_docx

    runtime = KaosRuntime.test_mode()  # in-memory VFS, throwaway session
    memory = SessionMemory("nda-hello")
    agent = ResearchAgent(runtime.vfs)  # default: anthropic:claude-haiku-4-5
    for path in sorted(p for p in NDAS_DIR.iterdir() if p.name.endswith(".docx")):
        uri = path.name.replace(" ", "_")  # IRI-safe
        agent.load_document(memory, uri, serialize_markdown(parse_docx(str(path))))
    await SessionStore(runtime.vfs).save(memory)
    response = await agent.turn(
        "Make a markdown table of key terms across these 5 NDAs. Columns: "
        "Document, Counterparty, Governing Law, Term Length, Confidentiality "
        "Period, Mutual?, Non-Solicit?. One row per NDA. Keep cells short.",
        session_id="nda-hello",
    )
    print(response.text)
    print(f"\ncost_usd=${response.cost_usd:.4f}  total_tokens={response.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
