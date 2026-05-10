"""Ground-truth field values for the 5 NDA fixtures.

Built by reading each contract manually. The values below are what the
agent SHOULD return when extracting fields from the corresponding
DOCX. T11 (tabular extraction) and T12 (LLM-as-judge memo) both
reference this module.

10 fields per contract, picked to cover the deal-room essentials a
senior associate would actually need to review:

  1. counterparty           — opposing party name, normalized
  2. counterparty_jurisdiction — state / country of formation
  3. effective_date         — ISO YYYY-MM-DD
  4. governing_law          — state whose laws govern
  5. venue                  — court venue / location
  6. term_years             — years the agreement runs (None if open-ended)
  7. confidentiality_period_years — years confidentiality survives
                              past termination (None if perpetual)
  8. mutual                 — bool, all 5 fixtures are mutual
  9. solicitation_period_months — non-solicit period (None if absent)
 10. counterparty_signatory_name — natural-person signer

Also notes ANOMALIES that a careful reviewer would flag — used by
T12's LLM-as-judge to grade memo quality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NDAGroundTruth:
    """Per-contract ground-truth for the NDA tiers."""

    filename: str
    counterparty: str
    counterparty_jurisdiction: str
    effective_date: str  # ISO YYYY-MM-DD
    governing_law: str
    venue: str
    term_years: int | None  # None = open-ended / survives until release
    confidentiality_period_years: int | None  # None = perpetual
    mutual: bool
    solicitation_period_months: int | None
    counterparty_signatory_name: str
    notable_anomalies: tuple[str, ...] = ()


GROUND_TRUTH: tuple[NDAGroundTruth, ...] = (
    NDAGroundTruth(
        filename="EMNA Mutual NDA.docx",
        counterparty="ExMachi Bank N.A.",
        counterparty_jurisdiction="National banking association",
        effective_date="2023-05-20",
        governing_law="Delaware",
        venue="Delaware",
        term_years=2,
        confidentiality_period_years=1,
        mutual=True,
        solicitation_period_months=6,
        counterparty_signatory_name="Jeremy Doe",
        notable_anomalies=(
            # The EMNA agreement names ExMachi Bank N.A. as the party but the
            # signature block reads "DynaMo" — a copy-paste artifact from the
            # template. Worth flagging in any review.
            "Signature block reads 'DynaMo' even though the named counterparty "
            "is ExMachi Bank N.A. (template copy-paste artifact).",
        ),
    ),
    NDAGroundTruth(
        filename="MNDA - Acme.docx",
        counterparty="Acme Co.",
        counterparty_jurisdiction="Nevada",
        effective_date="2023-07-24",
        governing_law="Michigan",
        venue="Lansing, Michigan",
        term_years=None,  # "effective from and after the Effective Date" — open-ended
        confidentiality_period_years=None,  # "survive expiration ... unless ... written notice"
        mutual=True,
        solicitation_period_months=None,  # No non-solicit clause
        counterparty_signatory_name="Jane Doe",
    ),
    NDAGroundTruth(
        filename="MNDA - BI.docx",
        counterparty="Beta Inc.",
        counterparty_jurisdiction="Delaware",
        effective_date="2023-06-03",
        governing_law="Michigan",
        venue="Lansing, Michigan",
        term_years=3,
        confidentiality_period_years=None,  # "survive expiration ... unless ... written notice"
        mutual=True,
        solicitation_period_months=None,
        counterparty_signatory_name="John Doe",
    ),
    NDAGroundTruth(
        filename="MNDA - CC Final 2.docx",
        counterparty="CyberCorp Co.",
        counterparty_jurisdiction="California",
        effective_date="2023-05-18",
        governing_law="Michigan",
        venue="Lansing, Michigan",
        term_years=5,
        confidentiality_period_years=1,
        mutual=True,
        solicitation_period_months=12,
        counterparty_signatory_name="Jorge Doe",
    ),
    NDAGroundTruth(
        filename="MNDA - DynaMo.docx",
        counterparty="DynaMo GmbH",
        counterparty_jurisdiction="Germany",
        effective_date="2023-05-20",
        governing_law="Delaware",
        venue="Delaware",
        term_years=2,
        confidentiality_period_years=1,
        mutual=True,
        solicitation_period_months=6,
        counterparty_signatory_name="Jeremy Doe",
    ),
)


# Keys that the extraction prompt asks for — must match what
# T11 prompts the model to emit.
EXTRACTION_FIELDS: tuple[str, ...] = (
    "filename",
    "counterparty",
    "counterparty_jurisdiction",
    "effective_date",
    "governing_law",
    "venue",
    "term_years",
    "confidentiality_period_years",
    "mutual",
    "solicitation_period_months",
    "counterparty_signatory_name",
)


def ground_truth_by_filename() -> dict[str, NDAGroundTruth]:
    """Index the ground-truth tuple by filename for easy lookup."""
    return {gt.filename: gt for gt in GROUND_TRUTH}


# Notable cross-contract observations a senior reviewer would surface
# in a deviation memo. T12's LLM-as-judge uses these as expected
# findings.
EXPECTED_FINDINGS: tuple[str, ...] = (
    # Governing-law deviations from the apparent house standard (Michigan)
    "EMNA and DynaMo are governed by Delaware law; the other three are "
    "governed by Michigan law (the apparent house standard).",
    # Term-length variation
    "Term lengths vary widely: Acme is open-ended (no fixed termination "
    "date), BI is 3 years, EMNA and DynaMo are 2 years, CyberCorp is 5 "
    "years.",
    # Confidentiality period variation
    "Confidentiality survival periods differ: EMNA, CyberCorp, and "
    "DynaMo limit confidentiality protection to 1 year post-disclosure; "
    "Acme and BI extend until written notice from the disclosing party.",
    # Non-solicit clause presence
    "Three contracts (EMNA, CyberCorp, DynaMo) include non-solicitation "
    "clauses (6 to 12 months) covering employees of the other party; "
    "Acme and BI omit non-solicit altogether.",
    # EMNA-specific template artifact worth flagging
    "EMNA Mutual NDA's signature block names 'DynaMo' but the parties "
    "section names ExMachi Bank N.A. — likely a template copy-paste "
    "artifact that a careful counterparty would want corrected before "
    "executing.",
)


__all__ = [
    "EXPECTED_FINDINGS",
    "EXTRACTION_FIELDS",
    "GROUND_TRUTH",
    "NDAGroundTruth",
    "ground_truth_by_filename",
]
