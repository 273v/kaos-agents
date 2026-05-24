# kaos-agents — common dev / test targets.
#
# Judge-gated integration targets (P1.1 / corpus-stress-suite follow-up):
#   - corpus-stress-judge   — 15 corpus-stress scenarios w/ Opus judge
#   - web-tools-judge       — 20 web-tools scenarios w/ Opus judge
#   - integration-judge     — both of the above
#
# Per the corpus-stress follow-up plan §2.1, the JUDGE model is Opus
# (per feedback_test_model_floor.md). The AGENT runs on the cheaper
# Sonnet-4-6 — the bar we ship against. Override either via the env
# vars at invocation time.

.PHONY: integration-judge corpus-stress-judge web-tools-judge

corpus-stress-judge:
	KAOS_TEST_MODEL=anthropic:claude-sonnet-4-6 \
	KAOS_TEST_JUDGE_MODEL=anthropic:claude-opus-4-7 \
	uv run pytest tests/integration/test_corpus_stress_suite.py \
	  -v -m live --no-cov --timeout=900

web-tools-judge:
	KAOS_TEST_MODEL=anthropic:claude-sonnet-4-6 \
	KAOS_TEST_JUDGE_MODEL=anthropic:claude-opus-4-7 \
	uv run pytest tests/integration/test_web_tools_legal_finance_suite.py \
	  -v -m live --no-cov --timeout=900

integration-judge: corpus-stress-judge web-tools-judge
