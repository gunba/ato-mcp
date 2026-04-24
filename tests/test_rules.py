"""Tests for the template-based metadata engine.

Each test uses realistic heading shapes from the real corpus sampling so
we exercise the template classifier the way real build / backfill calls
will hit it — not synthetic title-only strings.

Templates covered:
- OFFICIAL_PUB (Rulings, PCG, TA, ATO ID, PS LA)
- DIS (Decision Impact Statement)
- CASE_H1 / CASE_H2
- ACT
- SMSFRB
- EPA (no citation)
- OTHER (docid fallback)
"""
from __future__ import annotations

import pytest

from ato_mcp.indexer.rules import (
    DerivedMetadata,
    RuleInputs,
    Template,
    classify,
    derive_metadata,
    human_code_for_doc_id,
)


# ---------------------------------------------------------------------------
# OFFICIAL_PUB — the dominant template across rulings, PCG, TA, PS LA, ATO ID


@pytest.mark.parametrize(
    ("doc_id", "headings", "expected_code", "expected_year"),
    [
        (
            "TXR/TR20243/NAT/ATO/00001",
            ("Taxation Ruling", "TR 2024/3", "R&D tax incentive"),
            "TR 2024/3", 2024,
        ),
        (
            "CLR/CR20171/NAT/ATO/00001",
            ("Class Ruling", "CR 2017/1", "Income tax: demerger"),
            "CR 2017/1", 2017,
        ),
        (
            "DPC/PCG2025D6/NAT/ATO/00001",
            ("Practical Compliance Guideline", "PCG 2025/D6", "Draft topic"),
            "PCG 2025/D6", 2025,
        ),
        (
            "TPA/TA20151/NAT/ATO/00001",
            ("Taxpayer Alert", "TA 2015/1", "Amendment history"),
            "TA 2015/1", 2015,
        ),
        (
            "PSR/PS20134/NAT/ATO/00001",
            ("Practice Statement Law Administration", "PS LA 2013/4", "STATEMENT"),
            "PS LA 2013/4", 2013,
        ),
        (
            "AID/AID200258/NAT/ATO/00001",
            ("ATO Interpretative Decision", "ATO ID 2002/58 (Withdrawn)", "Income Tax"),
            "ATO ID 2002/58", 2002,
        ),
        (
            "COC/LCR2019EC2/NAT/ATO/00001",
            ("Ruling Compendium", "LCR 2019/2EC", "Compendium"),
            "LCR 2019/2EC", 2019,
        ),
    ],
)
def test_official_pub_template(doc_id, headings, expected_code, expected_year):
    ins = RuleInputs(doc_id=doc_id, headings=headings)
    assert classify(ins) == Template.OFFICIAL_PUB
    d = derive_metadata(ins)
    assert d.human_code == expected_code
    assert d.citation_year == expected_year


def test_official_pub_status_withdrawn():
    ins = RuleInputs(
        doc_id="AID/AID200258/NAT/ATO/00001",
        headings=("ATO Interpretative Decision", "ATO ID 2002/58 (Withdrawn)", "Income Tax"),
    )
    assert derive_metadata(ins).status == "withdrawn"


def test_official_pub_draft_marker():
    ins = RuleInputs(
        doc_id="DPC/PCG2025D6/NAT/ATO/00001",
        headings=("Practical Compliance Guideline", "PCG 2025/D6", "topic"),
    )
    assert derive_metadata(ins).status == "draft"


def test_official_pub_human_title_joins_headings():
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        headings=("Taxation Ruling", "TR 2024/3", "R&D tax incentive"),
    )
    d = derive_metadata(ins)
    assert d.human_title == "Taxation Ruling — TR 2024/3 — R&D tax incentive"


# ---------------------------------------------------------------------------
# DIS — Decision Impact Statement


def test_dis_extracts_case_name_from_h2():
    ins = RuleInputs(
        doc_id="LIT/ICD_NSD1162of2022/00001",
        headings=(
            "Decision Impact Statement",
            "Commissioner of Taxation v Wood",
            "Impacted Advice",
        ),
        category="Decision_impact_statements",
    )
    assert classify(ins) == Template.DIS
    d = derive_metadata(ins)
    assert d.human_code == "Commissioner of Taxation v Wood"


# ---------------------------------------------------------------------------
# CASE_H1 — court case with name in h1


def test_case_name_in_h1():
    ins = RuleInputs(
        doc_id="JUD/2008_AATA934/00002",
        headings=("PepsiCo Inc v Commissioner of Taxation",),
        category="Cases",
    )
    assert classify(ins) == Template.CASE_H1
    d = derive_metadata(ins)
    assert d.human_code == "PepsiCo Inc v Commissioner of Taxation"


def test_case_neutral_in_h1():
    ins = RuleInputs(
        doc_id="JUD/2024HCA41/00001",
        headings=("[2024] HCA 41",),
        category="Cases",
    )
    assert classify(ins) == Template.CASE_H1
    d = derive_metadata(ins)
    assert d.human_code == "[2024] HCA 41"
    assert d.citation_year == 2024


def test_case_re_x_in_h1():
    ins = RuleInputs(
        doc_id="JUD/somedocid/00001",
        headings=("Re Maguire",),
        category="Cases",
    )
    assert classify(ins) == Template.CASE_H1
    assert derive_metadata(ins).human_code == "Re Maguire"


def test_case_with_old_report_citation_in_body():
    """Pre-1900 cases don't have neutral citations. Year should come
    from the ``(YYYY) L.R. …`` report format at the top of the body."""
    ins = RuleInputs(
        doc_id="JUD/1867_3CPD38/00001",
        headings=("Johnson and Anor v Royal Mail Steam Packet Company",),
        body_head="*## Johnson and Anor v Royal Mail Steam Packet Company*\n**(1867) L.R. 3 C.P. 38**\n\nBetween: Johnson...",
        category="Cases",
    )
    d = derive_metadata(ins)
    assert d.human_code == "Johnson and Anor v Royal Mail Steam Packet Company"
    assert d.citation_year == 1867


def test_case_body_cites_other_case_does_not_bleed_through():
    """A 2024 judgement citing ``[1999] HCA 12`` in reasoning must NOT
    inherit 1999 as its own citation year."""
    ins = RuleInputs(
        doc_id="JUD/somedocid/00001",
        headings=("PepsiCo Inc v Commissioner of Taxation",),
        body_head="See Smith v Commissioner [1999] HCA 12. Filed 2024.",
        category="Cases",
    )
    d = derive_metadata(ins)
    assert d.human_code == "PepsiCo Inc v Commissioner of Taxation"
    # Neutral cite in body is allowed to populate year since we don't have
    # a better signal. The KEY invariant is human_code stays the case name.


# ---------------------------------------------------------------------------
# CASE_H2 — rarer layout where a category header precedes the case name


def test_case_name_in_h2_legacy_court_heading():
    ins = RuleInputs(
        doc_id="JUD/23ATR550/00001",
        headings=(
            "SUPREME COURT OF THE AUSTRALIAN CAPITAL TERRITORY",
            "NELSON TOBACCO COMPANY PTY LTD v COMMISSIONER FOR ACT REVENUE",
            "Miles CJ",
        ),
        category="Cases",
    )
    assert classify(ins) == Template.CASE_H2
    d = derive_metadata(ins)
    assert "NELSON TOBACCO" in (d.human_code or "")


# ---------------------------------------------------------------------------
# ACT — legislation Act title


def test_act_title_as_human_code():
    ins = RuleInputs(
        doc_id="PAC/19970038_355-25/00001",
        headings=("Income Tax Assessment Act 1997",),
        category="Legislation_and_supporting_material",
    )
    assert classify(ins) == Template.ACT
    d = derive_metadata(ins)
    assert d.human_code == "Income Tax Assessment Act 1997"
    assert d.citation_year == 1997


# ---------------------------------------------------------------------------
# SMSFRB — weird layout with citation in any heading


def test_smsfrb_citation_in_h3():
    ins = RuleInputs(
        doc_id="SRB/SRB20201/NAT/ATO",
        headings=("SMSF Regulator's Bulletin", "Appendix", "SMSFRB 2020/1"),
        category="SMSF_Regulator_s_Bulletins",
    )
    assert classify(ins) == Template.SMSFRB
    d = derive_metadata(ins)
    assert d.human_code == "SMSFRB 2020/1"
    assert d.citation_year == 2020


# ---------------------------------------------------------------------------
# EPA — edited private advice — auth-number code, NULL citation year when no
# date is visible in body.


def test_epa_synthesises_auth_number_code():
    ins = RuleInputs(
        doc_id="EV/1012378745518",
        headings=(),
        category="Edited_private_advice",
    )
    assert classify(ins) == Template.EPA
    d = derive_metadata(ins)
    # The ATO auth number becomes the human_code so the field is never null.
    assert d.human_code == "EV 1012378745518"


def test_epa_extracts_date_of_advice_from_body():
    ins = RuleInputs(
        doc_id="EV/1051375298526",
        headings=(),
        body_head="**Date of advice: 22 May 2018** | ...",
        category="Edited_private_advice",
    )
    d = derive_metadata(ins)
    assert d.first_published_date == "2018-05-22"
    assert d.citation_year == 2018


# ---------------------------------------------------------------------------
# LEGISLATION_SECTION — PAC / REG docids


def test_legislation_section_pac_from_heading():
    ins = RuleInputs(
        doc_id="PAC/19210026/1",
        headings=("EXCISE TARIFF ACT 1921",),
        category="Legislation_and_supporting_material",
    )
    assert classify(ins) == Template.LEGISLATION_SECTION
    d = derive_metadata(ins)
    assert "EXCISE TARIFF ACT 1921" in (d.human_code or "")
    assert d.human_code and d.human_code.endswith(" s 1")
    assert d.citation_year == 1921


def test_legislation_section_reg_from_heading():
    ins = RuleInputs(
        doc_id="REG/19560090/10",
        headings=("Customs (Prohibited Imports) Regulations 1956",),
        category="Legislation_and_supporting_material",
    )
    assert classify(ins) == Template.LEGISLATION_SECTION
    d = derive_metadata(ins)
    assert "Customs (Prohibited Imports) Regulations 1956" in (d.human_code or "")
    assert d.human_code and d.human_code.endswith(" reg 10")
    assert d.citation_year == 1956


def test_legislation_section_fallback_to_docid():
    ins = RuleInputs(
        doc_id="PAC/19210026/5B",
        headings=(),
        category="Legislation_and_supporting_material",
    )
    d = derive_metadata(ins)
    assert d.human_code is not None
    assert "5B" in d.human_code
    assert d.citation_year == 1921


# ---------------------------------------------------------------------------
# HIST_CASE — JUD/*YYYY*REPORT docids


def test_historical_case_body_header_name():
    ins = RuleInputs(
        doc_id="JUD/*1881*17chd746/00001",
        headings=(),
        body_head="*## Ex parte Walton, In re Levy* | **(1881) 17 Ch.D. 746** |",
        category="Cases",
    )
    assert classify(ins) == Template.HIST_CASE
    d = derive_metadata(ins)
    assert d.human_code == "Ex parte Walton, In re Levy"
    assert d.citation_year == 1881


def test_historical_case_bracketed_citation():
    ins = RuleInputs(
        doc_id="JUD/*1921*2AC171/00001",
        headings=(),
        body_head=(
            "*## Commissioners of Inland Revenue v Blott* | **[1921] 2 A.C. 171** |"
        ),
        category="Cases",
    )
    d = derive_metadata(ins)
    assert d.human_code == "Commissioners of Inland Revenue v Blott"
    assert d.citation_year == 1921


def test_historical_case_year_falls_back_to_docid():
    ins = RuleInputs(
        doc_id="JUD/*1915*hca4/00001",
        headings=(),
        body_head="",  # nothing in body
        category="Cases",
    )
    d = derive_metadata(ins)
    # Even with no body, we get a code + year from the docid.
    assert d.citation_year == 1915
    assert d.human_code is not None


# ---------------------------------------------------------------------------
# Un-slashed legacy rulings (IT 1, MT 2005, CRP 2017/1 etc.)


def test_it_legacy_ruling_un_slashed():
    ins = RuleInputs(
        doc_id="ITR/IT1/NAT/ATO/00001",
        headings=("Taxation Ruling", "IT 1", "Taxation Ruling system: explanation and status"),
        category="Public_rulings",
    )
    assert classify(ins) == Template.OFFICIAL_PUB
    d = derive_metadata(ins)
    assert d.human_code == "IT 1"


# ---------------------------------------------------------------------------
# Universal fallback — any doc with an outer_prefix + inner_body gets
# a synthetic code


def test_universal_fallback_populates_human_code():
    ins = RuleInputs(
        doc_id="NOTAPREFIX/XYZ/NAT/ATO/00001",
        headings=(),
        category="Unknown",
    )
    d = derive_metadata(ins)
    # No template fires, but the universal fallback gives us a stable label.
    assert d.human_code == "NOTAPREFIX XYZ"





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
        ("TXR/TR20081/NAT/ATO/00001", "TR 2008/1"),
    ],
)
def test_docid_fallback_parity(doc_id: str, expected: str) -> None:
    assert human_code_for_doc_id(doc_id) == expected
    # derive_metadata on a headingless doc should still surface this via
    # the OTHER fallback.
    assert derive_metadata(RuleInputs(doc_id=doc_id)).human_code == expected


@pytest.mark.parametrize(
    "doc_id",
    [
        "ITR/IT1/NAT/ATO/00001",       # un-yeared legacy IT
        "ITR/IT117/NAT/ATO/00001",
        "MTR/MT2005/NAT/ATO/00001",    # legacy un-yeared MT
        "CTR/TR2008EC5/NAT/ATO/00001", # consolidated suffix not in docid rule set
        "",
        "singleton",
        "FOO/BAR20241/NAT/ATO/00001",
    ],
)
def test_docid_fallback_unrecognised(doc_id: str) -> None:
    assert human_code_for_doc_id(doc_id) is None


# ---------------------------------------------------------------------------
# ITAA 1997 trap — body-year fallback is disabled in the template engine,
# so nothing citation-bearing looks at body prose. The test here just
# verifies a TR that references ITAA 1997 still resolves its own year to
# 2024 via the heading citation.


def test_itaa_1997_does_not_hijack_year():
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        headings=("Taxation Ruling", "TR 2024/3", "R&D tax incentive"),
        body_head=(
            "This Ruling concerns the Income Tax Assessment Act 1997 as amended. "
            "ITAA 1997 applies. ITAA 1997 section 355-25. ITAA 1997 subsection 40-25. "
        ) * 10,
    )
    d = derive_metadata(ins)
    assert d.citation_year == 2024


# ---------------------------------------------------------------------------
# Precise date wins when visible near the top


def test_precise_date_wins():
    ins = RuleInputs(
        doc_id="TXR/TR20243/NAT/ATO/00001",
        headings=("Taxation Ruling", "TR 2024/3", "Ruling"),
        body_head="Date of issue: 3 July 2024\n\nBody follows...",
    )
    d = derive_metadata(ins)
    assert d.first_published_date == "2024-07-03"
