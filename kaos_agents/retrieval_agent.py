"""RetrievalAgent — delegated sub-agent specialized in document retrieval.

The RetrievalAgent uses a ReAct loop with retrieval-specific tools to
dynamically search a document corpus. Unlike the hardcoded adaptive
retrieval pipeline, the agent decides which strategies to use based on
what it finds.

The parent agent (e.g., ResearchAgent) delegates retrieval to this
sub-agent via ``agent_as_tool``. The sub-agent returns the best
documents it found, and the parent agent uses them for RAG.

Usage::

    from kaos_agents.retrieval_agent import create_retrieval_agent

    retrieval_tool = create_retrieval_agent(runtime)
    parent = Agent.create(
        instructions="...",
        delegated_agents=[retrieval_tool],
    )
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

logger = get_logger(__name__)

_RETRIEVAL_INSTRUCTIONS = """\
You are a document retrieval specialist. Your job is to find the most
relevant documents from a corpus for a given search query.

STRATEGY:
1. Check the corpus first with kaos-retrieval-corpus-info to understand
   document count, vocabulary, and what you're searching.
2. Run a plain BM25 keyword search (kaos-retrieval-bm25).
3. Look at the results and the expansion_assessment signal. If results
   look good and expansion is not suggested, STOP.
4. If results are missing important aspects:
   a. Use kaos-retrieval-synonyms to find alternative terminology.
   b. Run additional BM25 searches with the synonyms.
5. If there's a large vocabulary gap (consumer language vs. technical language):
   a. Use kaos-retrieval-hyde to generate a pseudo-document in the corpus's
      vocabulary, then search against it.
6. If you have 10+ candidates and need better precision, use kaos-retrieval-rerank
   to reorder results by cross-encoder relevance scoring.
7. After 2-3 search rounds, evaluate coverage with kaos-retrieval-evaluate.
8. If coverage is sufficient, STOP and return your findings.

IMPORTANT:
- Do NOT run every tool on every query. Simple queries need only BM25.
- Do NOT over-search. 2-3 rounds is usually enough.
- Reranking is most useful when BM25 returned many candidates but you're not
  sure which are truly relevant. It does NOT find new documents — it reorders
  existing results.
- Report what you found AND what you could not find.
- Include the document previews in your response so the parent agent can use them.

Your response should be a summary of what you found, including:
- How many relevant documents you found
- What search strategies you used
- Key document previews (first 200 chars of each relevant doc)
- Any gaps in coverage
"""


def create_retrieval_agent(
    runtime: Any,
    *,
    model: str | None = None,
) -> Any:
    """Create a RetrievalAgent wrapped as a DelegatedAgent tool.

    Registers retrieval tools on the runtime and returns a DelegatedAgent
    that the parent agent can invoke.

    Args:
        runtime: KaosRuntime to register tools on.
        model: LLM model for the sub-agent. Defaults to settings default.

    Returns:
        A DelegatedAgent instance ready to pass to Agent.create(delegated_agents=[...]).
    """
    from kaos_agents.config import Agent, AgentPattern
    from kaos_agents.delegation import agent_as_tool
    from kaos_agents.retrieval_tools import register_retrieval_tools
    from kaos_agents.settings import DEFAULT_MODEL

    n_tools = register_retrieval_tools(runtime)
    logger.debug("retrieval_agent: registered %d retrieval tools", n_tools)

    agent = Agent.create(
        name="retrieval-agent",
        instructions=_RETRIEVAL_INSTRUCTIONS,
        model=model or DEFAULT_MODEL,
        pattern=AgentPattern.CHAT,
        tools=("kaos-retrieval-*",),
        max_tools=8,
        max_react_iterations=12,
    )

    return agent_as_tool(
        agent,
        name="search_documents",
        description=(
            "Search the document corpus for relevant content. Provide a natural-language "
            "query describing what you're looking for. The retrieval specialist will use "
            "BM25, synonym expansion, and other strategies to find the best documents."
        ),
    )
