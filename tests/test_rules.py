"""Tests for ``ato_mcp.indexer.rules`` — the unified metadata rule engine.

Covers:
- Ruling citations from docid path (parity with v1 parser).
- Ruling citations from title/headings (handles addenda/EC variants the
  docid flattens away).
- Neutral court citations and case-name fallbacks.
- The ITAA-1997 legislative-year trap (bare "1997" must not win when it
  appears only inside Act references).
- First-match-wins merge across rules.
"""
from __future__ import annotations

import pytest

from ato_mcp.indexer.rules import (
    DerivedMetadata,
    RuleInputs,
    derive_metadata,
    human_code_for_doc_id,
)


# ---------------------------------------------------------------------------
# Ruling citations — docid path


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        ("TXR/TR20243/NAT/ATO/00001", "TR 2024/3"),
        ("CLR/CR200117/NAT/ATO/00001", "CR 2001/17"),
        ("CGR/GSTR20243/NAT/ATO/00001", "GSTR 2024/3"),
        ("COC/LCR20181/NAT/ATO/00001", "LCR 2018/1"),
        ("COG/PCG20241/NAT/ATO/00001", "PCG 2024/1"),
        ("DPC/PCG2025D6/NAT/ATO/00001", "PCG 2025/D6"),
        ("DTR/TR2023D2/NAT/ATO/00001", "TR 2023/D2"),
        ("PSLA/PSLA202414/NAT/ATO/00001", "PS LA 2024/14"),
        ("DPS/PSD20191/NAT/ATO/00001", "PS LA 2019/D1"),
        ("ATOID/ATOID200114/NAT/ATO/00001", "ATO ID 2001/14"),
        ("AID/AID20011/NAT/ATO/00001", "ATO ID 2001/1"),
        ("TXR/TR9725/NAT/ATO/00001", "TR 97/25"),
        ("CGD/TD931/NAT/ATO/00001", "TD 93/1"),
        ("TXR/TR20081/NAT/ATO/00001", "TR 2008/1"),  # not TR 20/081
    ],
)
def test_docid_ruling_parity(doc_id: str, expected: str) -> None:
    assert derive_metadata(RuleInputs(doc_id=doc_id)).human_code == expected
    assert human_code_for_doc_id(doc_id) == expected


@pytest.mark.parametrize(
    "doc_id",
    [
        # Legacy un-yeared IT / legacy MT.
        "ITR/IT1/NAT/ATO/00001",
        "ITR/IT117/NAT/ATO/00001",
        "MTR/MT2005/NAT/ATO/00001",
        # Consolidated / addendum suffix — docid shape not supported.
        "CTR/TR2008EC5/NAT/ATO/00001",
        "CLR/CR20011A6/NAT/ATO/00001",
        # Malformed.
        "",
        "singleton",
        "FOO/BAR20241/NAT/ATO/00001",
    ],
)
def test_docid_unrecognised(doc_id: str) -> None:
    assert human_code_for_doc_id(doc_id) is None


# ---------------------------------------------------------------------------
# Heading citations beat docid parsing


def test_heading_addendum_wins_over_docid() -> None:
    """A TR whose docid is the EC-consolidated variant should surface the
    base ruling citation from its H2 rather than the opaque EC form."""
    ins = RuleInputs(
        doc_id="CTR/TR2008EC5/NAT/ATO/00001",
        title="Taxation Ruling",
        headings=("TR 2008/5 Addendum No 1",),
    )
    d = derive_metadata(ins)
    assert d.human_code == "TR 2008/5"
    assert d.citation_year == 2008


def test_heading_draft_marker() -> None:
    ins = RuleInputs(
        doc_id="DPC/PCG2025D6/NAT/ATO/00001",
        title="Practical Compliance Guideline",
        headings=("PCG 2025/D6",),
    )
    d = derive_metadata(ins)
    assert d.human_code == "PCG 2025/D6"
    assert d.status == "draft"


# ---------------------------------------------------------------------------
# Neutral court citations


@pytest.mark.parametrize(
    ("title", "expected_code", "expected_year"),
    [
        ("PepsiCo, Inc. v Commissioner of Taxation [2024] HCA 41", "[2024] HCA 41", 2024),
        ("[2024] FCAFC 123 — Some Case", "[2024] FCAFC 123", 2024),
        ("Re Applicant [2018] AATA 1123", "[2018] AATA 1123", 2018),
    ],
)
def test_neutral_citation(title: str, expected_code: str, expected_year: int) -> None:
    ins = RuleInputs(doc_id="JUD/unparseable/NAT/ATO/00001", title=title)
    d = derive_metadata(ins)
    assert d.human_code == expected_code
    assert d.citation_year == expected_year


def test_neutral_citation_ignores_body_cites() -> None:
    """A 2024 judgement citing `[1999] HCA 12` in reasoning must NOT
    mis-attribute 1999 to itself."""
    ins = RuleInputs(
        doc_id="JUD/unparseable/NAT/ATO/00001",
        title="PepsiCo Inc v Commissioner of Taxation",
        headings=("PepsiCo Inc v Commissioner of Taxation",),
        body_head="See Smith v Commissioner [1999] HCA 12. Filed 2024.",
        category="Cases",
    )
    d = derive_metadata(ins)
    assert d.human_code == "PepsiCo Inc v Commissioner of Taxation"
    # 2024 surfaces via body-year (max non-Act year) since no precise date given.
    assert d.citation_year == 2024


# ---------------------------------------------------------------------------
# Case name fallback


@pytest.mark.parametrize(
    "heading",
    [
        "PepsiCo, Inc. v Commissioner of Taxation",
        "Peterswald v Bartley",
        "Smith v Jones",
        "Acme Pty Ltd v Commissioner of Taxation",
    ],
)
def test_case_name_fallback(heading: str) -> None:
    ins = RuleInputs(doc_id="JUD/x/NAT/ATO/00001", title=heading)
    d = derive_metadata(ins)
    assert d.human_code and " v " in d.human_code


def test_case_name_ignores_prose_false_positives() -> None:
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        title="Taxation Ruling",
        headings=("TR 2024/3",),
        body_head="the agent applied a v b in the audit",
    )
    d = derive_metadata(ins)
    assert d.human_code == "TR 2024/3"


# ---------------------------------------------------------------------------
# Year classification / ITAA trap


def test_itaa_1997_references_do_not_dominate_year() -> None:
    """A 2024 TR that cites ITAA 1997 many times must still resolve to 2024."""
    body = (
        "This Ruling concerns section 355-25 of the Income Tax Assessment Act 1997 "
        "as amended. The ITAA 1997 applies. ITAA 1997 section 355-25 defines R&D "
        "activities. See also ITAA 1997 subdivision 355. The Tax Administration "
        "Act 1953 applies for filing. "
    ) * 5  # repeat to inflate 1997 counts
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        title="R&D Tax Offset",
        headings=("TR 2024/3",),
        body_head=body,
    )
    d = derive_metadata(ins)
    assert d.citation_year == 2024, d


def test_body_year_fallback_picks_latest_non_act() -> None:
    """When nothing else fires, body-year should pick the max plausible
    year AFTER masking Act references."""
    body = (
        "Earlier guidance issued 2018. Superseded by further guidance in 2022. "
        "Refers to Income Tax Assessment Act 1997 throughout."
    )
    ins = RuleInputs(
        doc_id="UNKNOWN/nothing/NAT/ATO/00001",
        body_head=body,
    )
    d = derive_metadata(ins)
    assert d.citation_year == 2022, d


def test_pub_date_takes_precedence_over_body_year() -> None:
    ins = RuleInputs(
        doc_id="UNKNOWN/nothing/NAT/ATO/00001",
        body_head="Some text mentioning 2020.",
        pub_date="2024-03-15",
    )
    d = derive_metadata(ins)
    assert d.first_published_date == "2024-03-15"
    assert d.citation_year == 2024


def test_precise_header_date_wins() -> None:
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        title="TR 2024/3",
        headings=("TR 2024/3", "Ruling"),
        body_head="Date of issue: 3 July 2024\n\nBody text follows...",
    )
    d = derive_metadata(ins)
    assert d.first_published_date == "2024-07-03"


# ---------------------------------------------------------------------------
# Status banners


def test_status_withdrawn() -> None:
    ins = RuleInputs(
        doc_id="ATOID/ATOID200114/NAT/ATO/00001",
        title="ATO ID 2001/14 (Withdrawn)",
        body_head="This ATO ID was withdrawn on 3 July 2020.",
    )
    d = derive_metadata(ins)
    assert d.status == "withdrawn"


def test_status_draft_marker_via_citation() -> None:
    ins = RuleInputs(
        doc_id="DTR/TR2024D2/NAT/ATO/00001",
        headings=("TR 2024/D2",),
    )
    d = derive_metadata(ins)
    assert d.status == "draft"


# ---------------------------------------------------------------------------
# Explanatory memorandum titles


def test_em_title_extracts_bill_name_and_year() -> None:
    ins = RuleInputs(
        doc_id="NEM/EM_TLB2024_01/NAT/ATO/00001",
        title="Explanatory Memorandum",
        headings=("Treasury Laws Amendment (Better Targeted Superannuation Concessions) Bill 2024",),
        category="Legislation_and_supporting_material",
    )
    d = derive_metadata(ins)
    assert d.human_code and "Treasury Laws Amendment" in d.human_code
    assert d.citation_year == 2024


# ---------------------------------------------------------------------------
# First-match-wins merge semantics


def test_first_match_wins_across_rules() -> None:
    """If multiple rules could produce `human_code`, the earlier one wins."""
    # Heading rule fires before docid rule; heading produces the addendum
    # variant; docid would have produced the plain TR 2008/5 too.
    ins = RuleInputs(
        doc_id="TXR/TR20085/NAT/ATO/00001",
        headings=("TR 2008/5A1",),  # addendum suffix preserved via heading regex
    )
    d = derive_metadata(ins)
    # Heading regex captures suffix, docid would have lost it.
    assert d.human_code == "TR 2008/5A1"
    assert d.variant == "A1"
