"""Typed Signatures for LLM-driven deliverable composition.

Two Signatures, both ship at module level so the codec can cache the
schema:

* :class:`SectionWriterSignature` — produce a single section's body
  text from a :class:`SectionSpec` + recalled context. Driven by
  :func:`compose_narrative` per-section.
* :class:`DeliverableStructureSignature` — derive a structure
  (ordered list of :class:`SectionSpec`s) from a prose task
  description + chosen deliverable kind. Used when the caller doesn't
  supply a structure explicitly.

Both follow the kaos-llm-core convention: ``InputField``/``OutputField``
descriptors, decision rules in the docstring (= system prompt), no
hand-rolled prompt templates.
"""

from __future__ import annotations

from typing import Literal

from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents.output.types import DeliverableKind


class SectionWriterSignature(Signature):
    """Write the body of one section of a narrative deliverable.

    You are a senior associate drafting one section of a structured
    legal-work deliverable (M&A review, compliance memo, regulatory
    research, etc.). Treat the recalled context as the corpus of
    everything the agent has produced or retrieved during its plan
    execution; treat the citations as the authoritative source list.

    Decision rules:

    1. Stay scoped to ``section_title``. Don't drift into other
       sections' subject matter.
    2. Cite specifically. When you state a fact, attach the
       ``source_uri`` from the citations list. Use markdown inline
       ``[source_uri]`` markers, or ``[^N]`` footnote markers if you
       prefer that style — the composer accepts either and routes
       the rendering.
    3. Match the depth. ``section_depth=1`` is a top-level
       executive section (concise, structured headlines).
       ``section_depth=2`` or deeper is detail (full prose with
       inline cites).
    4. If the recalled context is empty for this section, say so
       explicitly ("No findings recorded for this section.") rather
       than fabricating.
    5. ``section_instruction`` overrides any defaults above. When it
       contradicts a default, follow the instruction.
    """

    section_id: str = InputField(description="Stable identifier for this section.")
    section_title: str = InputField(description="The section's heading text.")
    section_depth: int = InputField(
        description="Markdown heading depth (1 = top-level, 2 = subsection, ...).",
        ge=1,
    )
    section_instruction: str = InputField(
        description=("Per-section guidance. Empty falls back to the docstring's decision rules.")
    )
    recalled_context: str = InputField(
        description=(
            "Assembled context for this section: relevant FINDINGS, "
            "ACTIONS, prior MESSAGES. Source-attributed where possible."
        )
    )
    citations: str = InputField(
        description=(
            "Authoritative citation list as text — `source_uri` per line. "
            "Use these source_uri values in inline citation markers."
        )
    )

    body: str = OutputField(
        description=(
            "The section body in markdown. Headings are added by the "
            "composer (do NOT prepend the section title yourself). "
            "Inline citations as ``[source_uri]`` or ``[^N]`` markers."
        )
    )


class DeliverableStructureSignature(Signature):
    """Derive an ordered section structure from a prose task description.

    When the caller doesn't supply an explicit structure for
    :func:`compose_narrative`, the agent runs this Call to derive
    one. Output is the spine of the deliverable — section ids,
    titles, and depths — that subsequent SectionWriter calls fill in.

    Decision rules:

    1. Match the deliverable kind. ``narrative`` → all sections are
       narrative. ``hybrid`` → at least one section MAY be a
       data-shaped section (typically named ``"findings-table"`` or
       similar) that downstream composition resolves to a table.
       ``tabular`` → return a single-section structure; the table is
       the deliverable.
    2. Sections must be ordered for human readability — Executive
       Summary first, References / Appendices last when they appear.
    3. Section ids are snake_case stable identifiers.
       ``["executive-summary", "per-contract-review", ...]`` — used
       by composers + critics as cross-references.
    4. Default structure for an M&A-shaped deliverable:
       executive-summary, per-contract-review, cross-contract-risks,
       recommendations, references.
    5. If the task description names specific structure elements
       ("a table of obligations", "with citations footer"), include
       them — those are explicit contracts.
    """

    task_description: str = InputField(
        description="Prose description of what the deliverable should cover."
    )
    deliverable_kind: Literal["tabular", "narrative", "hybrid"] = InputField(
        description=(
            "Shape of the target deliverable. Constrains the structure: "
            "tabular → one section; narrative/hybrid → multi-section."
        )
    )
    section_titles: list[str] = OutputField(
        description=(
            "Ordered list of section titles. Length matches the number "
            "of sections in the structure."
        )
    )
    section_ids: list[str] = OutputField(
        description=(
            "Stable snake-case identifiers, parallel to section_titles. "
            "Length must match section_titles."
        )
    )
    section_depths: list[int] = OutputField(
        description=(
            "Markdown heading depth per section, parallel to "
            "section_titles. Length must match. Typical: 1 for the "
            "top-level title, 2 for body sections."
        )
    )


# Convenience: typed Literal for the kind input — re-exported so callers
# importing this module can construct DeliverableStructureSignature
# without importing types.py too.
__all__ = [
    "DeliverableKind",
    "DeliverableStructureSignature",
    "SectionWriterSignature",
]
