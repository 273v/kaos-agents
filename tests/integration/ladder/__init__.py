"""Use-case ladder — 10 progressively complex live integration tests.

Every tier exercises a distinct capability layer. Each test asserts
both the answer (semantic) and the event-stream shape (regression).
The whole ladder runs on every test-suite execution per the project's
no-skips policy; total cost target ~$0.50, total wall-clock ~90s.

Tiers:
    1. Smoke — chat with no tools
    2. Single tool call — kaos-source-fr-search
    3. Multi-tool ReAct
    4. Plan-execute structured
    5. Memory continuity (2 turns)
    6. Citation extraction
    7. PDF + extraction
    8. Research delegation
    9. Permission gating (ToolCallApprovalRequired)
   10. Budget cap (BudgetExceeded)
"""
