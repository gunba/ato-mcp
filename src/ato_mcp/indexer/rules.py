"""Rule-based derivation of `human_code`, year, and status from a document's
docid path, title, headings, and body snippet.

Everything that can be computed from metadata alone — without touching
embeddings — lives here. The engine is pure-data input/output so the same
rules apply at build time (metadata derived once per ingest) and at
backfill time (re-derived on the live DB without GPU work).

Design:
- Every rule is a callable that takes `RuleInputs` and returns a partial
  `DerivedMetadata` dict (or None). The engine applies rules in order;
  each field is filled by the first rule that produces a value. Later
  rules can add fields but never override earlier ones.
- Rules are intentionally small. Adding a new citation shape means
  appending one rule, not editing the dispatch.
- Year extraction explicitly masks legislative-Act year references
  (`ITAA 1997`, `TAA 1953`, `GST Act 1999`, ...) before running the
  body-year fallback. Otherwise a modern TR that cites the 1997 Act
  dozens of times would misreport its own year as 1997.

Coverage target: the v1 docid-only parser hit 7%; with title/heading
rules grounded in real corpus samples this reaches ~48% on the current
~160 k corpus — Public Rulings ~95%, ATO IDs ~98%, PCG/PS LA/TA ~100%,
neutral case citations ~25%, case names as a recognisable label ~100%.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable


# ---------------------------------------------------------------------------
# Public data shapes


@dataclass(frozen=True)
class RuleInputs:
    """Everything the rule engine reads from for one document."""

    doc_id: str
    title: str | None = None
    headings: tuple[str, ...] = ()  # in document order, usually H1 first
    body_head: str = ""             # first ~2000 chars of extracted markdown
    category: str | None = None
    # Precomputed convenience: pub_date scraped from the body head earlier
    # in the pipeline. Falls back to None.
    pub_date: str | None = None

    # Derived once:
    @property
    def outer_prefix(self) -> str:
        segs = [s for s in self.doc_id.split("/") if s]
        return segs[0].upper() if segs else ""

    @property
    def inner_body(self) -> str:
        segs = [s for s in self.doc_id.split("/") if s]
        return segs[1] if len(segs) >= 2 else ""

    @property
    def joined_headings(self) -> str:
        return "\n".join(h for h in self.headings if h)


@dataclass(frozen=True)
class DerivedMetadata:
    """What every rule pass feeds into the documents table."""

    human_code: str | None = None
    first_published_date: str | None = None  # ISO yyyy-mm-dd or yyyy-01-01
    citation_year: int | None = None
    variant: str | None = None   # "Addendum No 1", "Erratum", "EC", ...
    status: str | None = None    # "draft" | "withdrawn" | "consolidated" | None


_EMPTY = DerivedMetadata()


def _merge(accum: DerivedMetadata, patch: dict) -> DerivedMetadata:
    """First-match-wins merge — only fill fields still None."""
    updates = {
        k: v for k, v in patch.items()
        if v is not None and getattr(accum, k) is None
    }
    return replace(accum, **updates) if updates else accum


# ---------------------------------------------------------------------------
# Regex toolbox


# Standard Australian ruling series that use `<CODE> YYYY/NN` form.
# Listed longest-first so the alternation is greedy on prefix length
# (Python's `|` is left-to-right, not longest-match).
_RULING_SERIES = sorted(
    [
        "SMSFRB", "SMSFR", "SMSFD",
        "GSTR", "GSTD", "FBTR", "WETR", "WETD",
        "LCR", "SGR", "FTR", "PCG", "LCG", "PRR", "CLR", "COG", "TXD",
        "TPA", "FBT", "GII",
        "CR", "PR", "TR", "TD", "MT", "TA", "LI", "LG", "WT",
    ],
    key=len,
    reverse=True,
)
_RULING_ALT = "|".join(_RULING_SERIES)

# In-title / in-heading ruling citation. Matches "TR 2024/3", "GSTR 2003/3EC",
# "LCR 2021/2A3", "PCG 2025/D5" (draft), with an optional suffix letter/digit
# tail for addenda, errata, consolidated variants.
_CIT_RULING_RE = re.compile(
    rf"\b(?P<series>{_RULING_ALT})\s+(?P<year>\d{{4}})/(?P<draft>D?)(?P<num>\d+)(?P<suffix>[A-Z]{{1,2}}\d*|)\b"
)

# Docid inner-body forms: TR20243, PCG2025D6, TR9725 (legacy 2-digit year).
_DOCID_YEAR4_RE = re.compile(
    rf"^(?P<series>{_RULING_ALT})(?P<year>\d{{4}})(?P<draft>D?)(?P<num>\d+)$"
)
# Legacy: require year to start with 8 or 9 (1980s/1990s) so "MT2005" (a
# legacy un-yeared MT ruling) doesn't become "MT 20/05".
_DOCID_YEAR2_RE = re.compile(
    rf"^(?P<series>{_RULING_ALT})(?P<year>[89]\d)(?P<num>\d+)$"
)

# PS LA: `PS LA YYYY/NN` in text, `PSLAYYYYNN` in docid. PSD inner prefix is
# the draft form and renders with a `/D` marker.
_CIT_PSLA_RE = re.compile(r"\bPS\s+LA\s+(?P<year>\d{4})/(?P<draft>D?)(?P<num>\d+)\b")
_DOCID_PSLA_RE = re.compile(r"^PSLA(?P<year>\d{4})(?P<num>\d+)$")
_DOCID_PSLA_DRAFT_RE = re.compile(r"^PSD(?P<year>\d{4})D?(?P<num>\d+)$")

# ATO ID: `ATO ID YYYY/NN` in text, `ATOIDYYYYNN` / `AIDYYYYNN` in docid.
_CIT_ATOID_RE = re.compile(r"\bATO\s+ID\s+(?P<year>\d{4})/(?P<num>\d+)\b")
_DOCID_ATOID_RE = re.compile(r"^(?:ATOID|AID)(?P<year>\d{4})(?P<num>\d+)$")

# Neutral court citations: `[YYYY] COURT NUM`. Courts captured are the ones
# actually seen in the ATO corpus plus the standard Australian set.
_NEUTRAL_CITATION_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s+(?P<court>HCA|FCAFC|FCA|FCC|FCCA|FMCA|AATA|ART|"
    r"NSWSC|NSWCA|NSWCCA|NSWADT|NSWDC|VSC|VSCA|VCAT|QSC|QCA|QCAT|SASC|SASCFC|"
    r"WASC|WASCA|TASSC|ACTSC|NTSC|HCATrans)\s+(?P<num>\d+)\b"
)

# Case name: "PepsiCo, Inc. v Commissioner of Taxation", "Smith v Jones".
# Tolerant of `v` / `v.` / ` vs `, optional Pty Ltd / Inc / Limited / Plc
# party-type tails. Anchored at the start of a heading/title (^ with
# re.MULTILINE). Intentionally strict about capitalisation to avoid matching
# prose like "applied a v b in the judgement".
_CASE_NAME_RE = re.compile(
    r"^(?P<party_a>(?:[A-Z][\w'.&-]*(?:\s+[A-Z][\w'.&-]*)*)"
    r"(?:,?\s+(?:Pty\s+)?(?:Ltd|Limited|Inc\.?|LLC|Corp|Co\.?|Plc))?)"
    r"\s+(?:v\.?|vs\.?)\s+"
    r"(?P<party_b>[A-Z][\w'.&-]*(?:\s+[\w'.&()-]+){0,8})"
    r"(?:\s*\[|\s*\(|\s*$)",
    re.MULTILINE,
)

# Explanatory Memorandum / Bill titles: "Treasury Laws Amendment (Foo) Bill 2024",
# "Tax Laws Amendment (Bar) Bill 2023". Capture the year so we also fill
# citation_year. Doesn't produce a ruling-style short code but gives a
# recognisable label.
_EM_TITLE_RE = re.compile(
    r"\b((?:Treasury|Tax|Superannuation|Customs|Excise|Petroleum)\s+Laws?\s+"
    r"(?:Amendment|Compliance)(?:\s+\([^)]{1,80}\))?\s+Bill)\s+(\d{4})\b"
)
_ACT_TITLE_RE = re.compile(
    r"\b((?:Income\s+Tax|Fringe\s+Benefits\s+Tax|Goods\s+and\s+Services\s+Tax|"
    r"Superannuation|Tax\s+Administration|Luxury\s+Car\s+Tax|Wine\s+Equalisation\s+Tax|"
    r"Petroleum\s+Resource\s+Rent\s+Tax|A\s+New\s+Tax\s+System[^.]{0,80})"
    r"\s+(?:Assessment\s+)?(?:Act|Regulations?))\s+(\d{4})\b"
)

# Status banners.
_STATUS_WITHDRAWN_RE = re.compile(r"\b(withdrawn|replaced\s+by|superseded)\b", re.IGNORECASE)
_STATUS_DRAFT_RE = re.compile(r"\bdraft\b", re.IGNORECASE)
_STATUS_CONSOLIDATED_RE = re.compile(r"\bconsolidated\b", re.IGNORECASE)

# Precise date scraping: "3 July 2024".
_PRECISE_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH_INDEX = {
    m.lower(): i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
        start=1,
    )
}

# Any bare 4-digit year.
_BARE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Legislative Act references — mask before scanning for the doc's own year.
# Each pattern captures the year; we rewrite the captured span to `XXXX` so
# downstream year scans ignore it. Ordered so longer / more specific forms
# match first (ITAA > Act, ANTS (GST Act) > GST Act).
_ACT_PHRASE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bITAA\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bTAA\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bFBTAA\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bSISA\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bSGAA\s*(\d{4})\b", re.IGNORECASE),
    re.compile(
        r"\bA\s+New\s+Tax\s+System\s*\([^)]*\)\s*Act\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bANTS[A-Z]*\s*(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bGST\s+Act\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bTax\s+Administration\s+Act\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bIncome\s+Tax\s+Assessment\s+Act\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bFringe\s+Benefits\s+Tax\s+Assessment\s+Act\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bSuperannuation\s+(?:Guarantee\s+\(Administration\)|Industry\s+\(Supervision\))\s+Act\s+(\d{4})\b", re.IGNORECASE),
    # Generic fallback last — any "<words> Act YYYY" / "<words> Regulations YYYY".
    re.compile(r"\b(?:Act|Regulations?|Code)\s+(\d{4})\b", re.IGNORECASE),
)


def _mask_act_years(text: str) -> str:
    """Replace legislative `<Act> YYYY` year captures with `XXXX`.

    Called before any bare-year scan so that a modern ruling citing the
    `Income Tax Assessment Act 1997` twelve times doesn't get classified
    as a 1997 document.
    """
    def _blank(match: re.Match[str]) -> str:
        start, end = match.span(1)
        full_start, full_end = match.span(0)
        # Rebuild the full match with the year span replaced.
        prefix = match.string[full_start:start]
        suffix = match.string[end:full_end]
        return f"{prefix}XXXX{suffix}"

    out = text
    for pat in _ACT_PHRASE_RES:
        out = pat.sub(_blank, out)
    return out


# ---------------------------------------------------------------------------
# Individual rules
#
# Each returns a partial dict of DerivedMetadata fields, or None.


def _rule_docid_ruling(ins: RuleInputs) -> dict | None:
    """Parse a canonical ruling citation from the docid second segment."""
    m = _DOCID_YEAR4_RE.match(ins.inner_body)
    if m:
        series = m["series"]
        year = m["year"]
        draft = m["draft"] or ""
        num = m["num"]
        return {
            "human_code": f"{series} {year}/{draft}{num}",
            "citation_year": int(year),
            "status": "draft" if draft else None,
        }
    return None


def _rule_docid_ruling_legacy(ins: RuleInputs) -> dict | None:
    """Pre-2000 legacy 2-digit-year form in the docid second segment."""
    m = _DOCID_YEAR2_RE.match(ins.inner_body)
    if m:
        series = m["series"]
        year_short = m["year"]
        num = m["num"]
        full_year = 1900 + int(year_short)
        return {
            "human_code": f"{series} {year_short}/{num}",
            "citation_year": full_year,
        }
    return None


def _rule_docid_psla(ins: RuleInputs) -> dict | None:
    m = _DOCID_PSLA_RE.match(ins.inner_body)
    if m:
        return {
            "human_code": f"PS LA {m['year']}/{m['num']}",
            "citation_year": int(m["year"]),
        }
    m = _DOCID_PSLA_DRAFT_RE.match(ins.inner_body)
    if m:
        return {
            "human_code": f"PS LA {m['year']}/D{m['num']}",
            "citation_year": int(m["year"]),
            "status": "draft",
        }
    return None


def _rule_docid_atoid(ins: RuleInputs) -> dict | None:
    m = _DOCID_ATOID_RE.match(ins.inner_body)
    if m:
        return {
            "human_code": f"ATO ID {m['year']}/{m['num']}",
            "citation_year": int(m["year"]),
        }
    return None


def _rule_heading_ruling(ins: RuleInputs) -> dict | None:
    """Look for a ruling citation in title + headings (the doc's OWN citation).

    Deliberately excludes body text: a Taxation Ruling that cites another
    TR in its body shouldn't borrow that other TR's code.
    """
    for field in [ins.title or "", *ins.headings[:5]]:
        m = _CIT_RULING_RE.search(field)
        if m:
            series = m["series"]
            year = m["year"]
            draft = m["draft"] or ""
            num = m["num"]
            suffix = m["suffix"] or ""
            full = f"{series} {year}/{draft}{num}{suffix}"
            return {
                "human_code": full,
                "citation_year": int(year),
                "variant": suffix if suffix else None,
                "status": "draft" if draft else None,
            }
    return None


def _rule_heading_psla(ins: RuleInputs) -> dict | None:
    for field in [ins.title or "", *ins.headings[:5]]:
        m = _CIT_PSLA_RE.search(field)
        if m:
            draft = m["draft"] or ""
            return {
                "human_code": f"PS LA {m['year']}/{draft}{m['num']}",
                "citation_year": int(m["year"]),
                "status": "draft" if draft else None,
            }
    return None


def _rule_heading_atoid(ins: RuleInputs) -> dict | None:
    for field in [ins.title or "", *ins.headings[:5]]:
        m = _CIT_ATOID_RE.search(field)
        if m:
            return {
                "human_code": f"ATO ID {m['year']}/{m['num']}",
                "citation_year": int(m["year"]),
            }
    return None


def _rule_neutral_citation(ins: RuleInputs) -> dict | None:
    """Court neutral citation: `[2024] FCAFC 123` — ONLY in title/headings.

    Body-text scanning is deliberately excluded: a 2024 judgement that
    discusses `Smith v Commissioner [1999] HCA 12` in its reasoning would
    otherwise borrow 1999 as its own citation. A document's own neutral
    citation is always rendered in the title or first few headings when
    present at all.
    """
    for field in [ins.title or "", *ins.headings[:3]]:
        m = _NEUTRAL_CITATION_RE.search(field)
        if m:
            return {
                "human_code": f"[{m['year']}] {m['court']} {m['num']}",
                "citation_year": int(m["year"]),
            }
    return None


def _rule_case_name(ins: RuleInputs) -> dict | None:
    """Anglo-style case name: `PepsiCo Inc v Commissioner of Taxation`.

    Only populates human_code when the earlier neutral-citation rule hasn't
    fired, so cases with both forms get the neutral citation (more compact,
    legally canonical) and the case name is left to `human_title`. Here we
    surface the case name explicitly so the agent can filter / display it.

    Searches title + each heading in order; first qualifying start-of-line
    match wins. Party-name heuristic: capitalised word cluster (optionally
    with ``Pty Ltd`` / ``Inc`` tails) on either side of ``v`` / ``v.`` /
    ``vs``. Strict anchoring prevents prose false positives like
    ``applied a v b`` inside body text.
    """
    candidates: list[str] = []
    if ins.title:
        candidates.append(ins.title)
    candidates.extend(ins.headings[:4])
    for text in candidates:
        m = _CASE_NAME_RE.search(text)
        if m:
            a = m["party_a"].strip(" ,.")
            b = m["party_b"].strip(" ,.")
            # Collapse whitespace and trim any trailing bracket-fragment.
            b = re.sub(r"\s*\[[^\]]*$", "", b).strip()
            name = f"{a} v {b}"
            if len(name) > 180:  # sanity cap — garbage matches from long lists
                continue
            return {"human_code": name}
    return None


def _rule_em_title(ins: RuleInputs) -> dict | None:
    """Explanatory Memorandum / Bill title → `EM to <Bill Name> YYYY`."""
    search = (ins.title or "") + "\n" + ins.joined_headings
    m = _EM_TITLE_RE.search(search)
    if m:
        bill = m.group(1).strip()
        year = int(m.group(2))
        return {
            "human_code": f"EM to {bill} {year}",
            "citation_year": year,
        }
    return None


def _rule_act_title(ins: RuleInputs) -> dict | None:
    """Act / Regulations title → `<Act> YYYY`.

    Guard: only applied when the category is a legislative bucket so
    ordinary rulings (which cite Acts all the time) don't get mis-labelled
    as legislative material.
    """
    if ins.category != "Legislation_and_supporting_material":
        return None
    search = (ins.title or "") + "\n" + ins.joined_headings
    m = _ACT_TITLE_RE.search(search)
    if m:
        act = m.group(1).strip()
        year = int(m.group(2))
        return {
            "human_code": f"{act} {year}",
            "citation_year": year,
        }
    return None


def _rule_precise_date(ins: RuleInputs) -> dict | None:
    """Extract ISO date from the first 500 chars (title + first headings)."""
    snippet = (ins.title or "") + "\n" + ins.joined_headings + "\n" + ins.body_head[:500]
    m = _PRECISE_DATE_RE.search(snippet)
    if m:
        day, month_name, year = m.groups()
        month = _MONTH_INDEX[month_name.lower()]
        iso = f"{int(year):04d}-{month:02d}-{int(day):02d}"
        return {"first_published_date": iso, "citation_year": int(year)}
    return None


def _rule_pub_date_fallback(ins: RuleInputs) -> dict | None:
    """Use the body-scraped pub_date if nothing more precise surfaced."""
    if ins.pub_date:
        year = None
        if len(ins.pub_date) >= 4 and ins.pub_date[:4].isdigit():
            year = int(ins.pub_date[:4])
        return {"first_published_date": ins.pub_date, "citation_year": year}
    return None


def _rule_docid_year(ins: RuleInputs) -> dict | None:
    """Fill citation_year and year-only first_published_date from the docid body."""
    # 4-digit year in the inner body takes priority.
    for pat in (_DOCID_YEAR4_RE, _DOCID_PSLA_RE, _DOCID_PSLA_DRAFT_RE, _DOCID_ATOID_RE):
        m = pat.match(ins.inner_body)
        if m:
            year = int(m["year"])
            return {"citation_year": year, "first_published_date": f"{year}-01-01"}
    # Generic fallback: scan for (19|20)YY in the first two path segments.
    segs = [s for s in ins.doc_id.split("/") if s][:2]
    for seg in segs:
        match = _BARE_YEAR_RE.search(seg)
        if match:
            year = int(match.group(0))
            return {"citation_year": year, "first_published_date": f"{year}-01-01"}
    return None


def _rule_body_year(ins: RuleInputs) -> dict | None:
    """Latest non-legislative year in the body head.

    After `_mask_act_years` blanks out references to named Acts, we take
    the maximum bare 4-digit year left. A document can't cite a year that
    hasn't happened yet (future-dated references are rare), so the highest
    year in the text is usually within a year or two of the publication
    date. This sidesteps the "most-common-year wins" trap where legislative
    boilerplate would swamp the doc's own year.
    """
    if not ins.body_head:
        return None
    masked = _mask_act_years(ins.body_head)
    years = [int(m.group(0)) for m in _BARE_YEAR_RE.finditer(masked)]
    if not years:
        return None
    # Cap at current year + 2 so an errant forward-looking reference doesn't
    # dominate. Hard-coded upper bound updates with the current year.
    import datetime as _dt
    cap = _dt.datetime.now(tz=_dt.timezone.utc).year + 2
    plausible = [y for y in years if 1900 <= y <= cap]
    if not plausible:
        return None
    year = max(plausible)
    return {"citation_year": year, "first_published_date": f"{year}-01-01"}


def _rule_status_banner(ins: RuleInputs) -> dict | None:
    """Status overlay — fills `status` if a banner is visible in the body head."""
    snippet = ins.body_head[:1500]
    if _STATUS_WITHDRAWN_RE.search(snippet):
        return {"status": "withdrawn"}
    if _STATUS_DRAFT_RE.search(snippet) and not _STATUS_CONSOLIDATED_RE.search(snippet):
        # Don't flip a consolidated draft as purely draft.
        return {"status": "draft"}
    if _STATUS_CONSOLIDATED_RE.search(snippet):
        return {"status": "consolidated"}
    return None


RULES: tuple[Callable[[RuleInputs], dict | None], ...] = (
    # Precise date beats everything for first_published_date.
    _rule_precise_date,
    # Heading / title citations win over docid parsing — they know about
    # addenda and errata that the docid flattens away.
    _rule_heading_ruling,
    _rule_heading_psla,
    _rule_heading_atoid,
    _rule_neutral_citation,
    # Case name only fires if neutral citation didn't — most legally-canonical
    # cite form takes priority.
    _rule_case_name,
    # EM / Bill / Act titles for legislative material that won't carry a
    # ruling code.
    _rule_em_title,
    _rule_act_title,
    # Docid parsing — canonical short code for modern rulings that lack a
    # heading citation (rare) or where heading parsing fails.
    _rule_docid_ruling,
    _rule_docid_psla,
    _rule_docid_atoid,
    _rule_docid_ruling_legacy,
    # Year-only fallbacks.
    _rule_pub_date_fallback,
    _rule_docid_year,
    _rule_body_year,
    # Status applies last; earlier rules can set it, but this catches the
    # common case where no citation was extracted but the banner is plain.
    _rule_status_banner,
)


def derive_metadata(inputs: RuleInputs) -> DerivedMetadata:
    """Run every rule; return a `DerivedMetadata` filled first-match-wins.

    Safe to call with partial inputs — empty headings/body_head just means
    fewer rules fire. Output fields may all be None for a doc the rule set
    can't recognise (EV edited private advice, heterogeneous legislative
    supporting material); callers should tolerate that.
    """
    accum = _EMPTY
    for rule in RULES:
        patch = rule(inputs)
        if patch:
            accum = _merge(accum, patch)
    return accum


# ---------------------------------------------------------------------------
# Backwards compatibility — v1 callers import `human_code_for_doc_id` from
# metadata.py. Keep the docid-only entry point so older code keeps working.


def human_code_for_doc_id(doc_id: str) -> str | None:
    """Compute just human_code using docid rules only (no title / headings)."""
    ins = RuleInputs(doc_id=doc_id)
    for rule in (_rule_docid_ruling, _rule_docid_psla, _rule_docid_atoid, _rule_docid_ruling_legacy):
        patch = rule(ins)
        if patch and patch.get("human_code"):
            return patch["human_code"]
    return None
