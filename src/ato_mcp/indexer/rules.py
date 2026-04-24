"""Template-based metadata classifier.

Empirically, ATO pages fall into a small number of structural templates
(~10 by volume). Rather than running a dozen regexes against free text
hoping one matches, we *classify* each page into a template using its
heading-shape signature, then apply the template's positional extractor.

This is more principled than regex-across-text because the ATO's
publishing stack is consistent within each doc type. A Taxation Ruling
always has ``<h1>Taxation Ruling</h1><h2>TR 2024/3</h2><h3>subtitle</h3>``;
a Practical Compliance Guideline always has
``<h1>Practical Compliance Guideline</h1><h2>PCG 2025/D6</h2><h3>topic</h3>``;
a Decision Impact Statement always has
``<h1>Decision impact statement</h1><h2>Smith v Commissioner</h2>``.
Extraction becomes a positional read rather than a pattern match.

Templates identified:

- ``T_OFFICIAL_PUB``   Rulings / PCG / TA / PS LA / ATO ID / SMSFRB.
                      h1 = type phrase, h2 = citation.
- ``T_CASE_H1``        Court case with party-v-party or Re-X in h1.
- ``T_CASE_H2``        Court case with name in h2 (rarer layout).
- ``T_DIS``            Decision Impact Statement — h1 = DIS phrase,
                      h2 = case name.
- ``T_ACT``            Act / Regulation — h1 = "<Name> Act <Year>".
- ``T_EM``             Explanatory Memorandum / Bill — shape varies;
                      best we do is pull year + bill phrase out.
- ``T_EPA``            Edited private advice — no headings, no citable
                      short form. Leaves human_code NULL.
- ``T_OTHER``          Fallback — docid-only extraction.

Field outputs:

- ``human_code``          Short citation form ("TR 2024/3", "[2024] HCA 41",
                         "PepsiCo Inc v Commissioner of Taxation"). May
                         be NULL for templates without a citable form.
- ``first_published_date`` ISO date if a precise one is visible, else
                          YYYY-01-01 from the year, else None.
- ``citation_year``       Integer year the citation refers to.
- ``variant``             Suffix like "Addendum", "Erratum", "EC".
- ``status``              "draft" / "withdrawn" / "consolidated" / None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Input / output shapes


@dataclass(frozen=True)
class RuleInputs:
    doc_id: str
    title: str | None = None
    headings: tuple[str, ...] = ()
    body_head: str = ""
    category: str | None = None
    pub_date: str | None = None

    @property
    def outer_prefix(self) -> str:
        segs = [s for s in self.doc_id.split("/") if s]
        return segs[0].upper() if segs else ""

    @property
    def inner_body(self) -> str:
        segs = [s for s in self.doc_id.split("/") if s]
        return segs[1] if len(segs) >= 2 else ""

    @property
    def h1(self) -> str:
        return self.headings[0].strip() if self.headings else ""

    @property
    def h2(self) -> str:
        return self.headings[1].strip() if len(self.headings) > 1 else ""

    @property
    def h3(self) -> str:
        return self.headings[2].strip() if len(self.headings) > 2 else ""


@dataclass(frozen=True)
class DerivedMetadata:
    human_code: str | None = None
    human_title: str | None = None
    first_published_date: str | None = None
    citation_year: int | None = None
    variant: str | None = None
    status: str | None = None


_EMPTY = DerivedMetadata()


# ---------------------------------------------------------------------------
# Heading-shape classification
#
# Each shape is a cheap regex-based feature over one heading string. The
# classification dispatch reads shapes, not raw text.


class Shape(str, Enum):
    EMPTY = "EMPTY"
    RULING_TYPE_PHRASE = "RULING_TYPE_PHRASE"     # "Taxation Ruling", "Class Ruling", ...
    GUIDELINE_TYPE_PHRASE = "GUIDELINE_TYPE_PHRASE"  # "Practical Compliance Guideline(s)"
    ALERT_PHRASE = "ALERT_PHRASE"                 # "Taxpayer Alert"
    ATOID_PHRASE = "ATOID_PHRASE"                 # "ATO Interpretative Decision"
    PSLA_PHRASE = "PSLA_PHRASE"                   # "Practice Statement Law Administration"
    SMSFRB_PHRASE = "SMSFRB_PHRASE"               # "SMSF Regulator's Bulletin"
    DIS_PHRASE = "DIS_PHRASE"                     # "Decision impact statement"
    EM_PHRASE = "EM_PHRASE"                       # "Explanatory Memorandum"
    RULING_CITATION = "RULING_CITATION"           # "TR 2024/3" etc.
    ATOID = "ATOID"                               # "ATO ID 2024/3"
    PSLA = "PSLA"                                 # "PS LA 2024/3"
    SMSFRB = "SMSFRB"                             # "SMSFRB 2020/1"
    NEUTRAL_CITATION = "NEUTRAL_CITATION"         # "[2024] HCA 41"
    NAME_V_NAME = "NAME_V_NAME"                   # "Smith v Jones"
    RE_X = "RE_X"                                 # "Re Smith", "In re Smith"
    CASE_NUMBER = "CASE_NUMBER"                   # "Case 9/93", "Case K68"
    ACT_TITLE = "ACT_TITLE"                       # "Income Tax Assessment Act 1997"
    BILL_TITLE = "BILL_TITLE"                     # "... Bill 2024"
    OTHER = "OTHER"


# Regex index used for shape detection only. Recipes that extract values
# (year, number, variant) may re-use these — but dispatch cares about
# *which* shape fired, not the captured groups.

_RULING_SERIES_ALT = "|".join(sorted([
    "SMSFRB", "SMSFR", "SMSFD", "GSTR", "GSTD", "FBTR", "WETR", "WETD",
    "LCR", "SGR", "FTR", "PCG", "LCG", "PRR", "CLR", "COG", "TXD", "TPA",
    "FBT", "GII", "CR", "PR", "TR", "TD", "MT", "TA", "LI", "LG", "WT", "IT",
], key=len, reverse=True))

_RE_RULING_CITATION = re.compile(
    rf"^({_RULING_SERIES_ALT})\s+\d{{1,4}}/D?\d+(?:[A-Z0-9]+)?(?:\s|$|\()"
)
_RE_ATOID = re.compile(r"^ATO\s+ID\s+\d{4}/\d+")
_RE_PSLA = re.compile(r"^PS\s+LA\s+\d{4}/")
_RE_SMSFRB = re.compile(r"^SMSFRB\s+\d{4}/")
_RE_NEUTRAL = re.compile(r"^\[\d{4}\]\s+[A-Z]+\s+\d+")
_RE_NAME_V_NAME = re.compile(
    r"^[A-Z][\w'.&-]*(?:\s+(?:[A-Z][\w'.&-]*|and|of|the|for|on|in|an|Anor|ors?|No|nee))*"
    r"(?:,?\s+(?:Pty\s+)?(?:Ltd|Limited|Inc\.?|LLC|Corp|Co\.?|Plc))?"
    r"\s+(?:v\.?|vs\.?)\s+"
    r"(?:the|a|an)?\s*[A-Za-z][\w'.&-]*",
    re.IGNORECASE,
)
_RE_RE_X = re.compile(
    r"^(?:Re|In\s+re|In\s+the\s+Matter\s+of|Ex\s+parte)\s+[A-Z]",
    re.IGNORECASE,
)
_RE_CASE_NUMBER = re.compile(
    r"^Case\s+[A-Z]?\d+(?:/\d+)?$",
    re.IGNORECASE,
)
_RE_ACT_TITLE = re.compile(
    r"^(?:[A-Z][\w]*\s+)*(?:Act|Regulations?|Code|Rules)\s+(?:19|20)\d{2}"
    r"(?:\s*\(Cth\))?\s*$"
)
_RE_BILL_TITLE = re.compile(r"\bBill\s+(?:19|20)\d{2}\b")

_TYPE_PHRASES = {
    Shape.RULING_TYPE_PHRASE: {
        "taxation ruling", "class ruling", "product ruling",
        "law companion ruling", "gst ruling", "gst determination",
        "taxation determination", "superannuation guarantee ruling",
        "fuel tax ruling", "fringe benefits tax ruling",
        "income tax ruling", "miscellaneous taxation ruling",
        "law companion guideline", "wine equalisation tax ruling",
        "wine equalisation tax determination",
        "superannuation guarantee determination",
        "smsf ruling", "smsf determination", "ruling compendium",
        "goods and services tax ruling", "goods and services tax determination",
    },
    Shape.GUIDELINE_TYPE_PHRASE: {
        "practical compliance guideline", "practical compliance guidelines",
    },
    Shape.ALERT_PHRASE: {
        "taxpayer alert",
    },
    Shape.ATOID_PHRASE: {
        "ato interpretative decision",
    },
    Shape.PSLA_PHRASE: {
        "practice statement law administration",
        "ato practice statement law administration",
        "law administration practice statement",
    },
    Shape.SMSFRB_PHRASE: {
        "smsf regulator's bulletin", "smsf regulators bulletin",
    },
    Shape.DIS_PHRASE: {
        "decision impact statement", "decision impact statements",
    },
    Shape.EM_PHRASE: {
        "explanatory memorandum", "supplementary explanatory memorandum",
    },
}


def shape_of(heading: str) -> Shape:
    """Classify a heading string into a coarse structural shape.

    Order matters: citations before type phrases, because a heading like
    ``"TR 2024/3"`` might technically match ``"Taxation Ruling"`` (it
    doesn't, but the principle holds for shorter strings).
    """
    if not heading or not heading.strip():
        return Shape.EMPTY
    t = heading.strip()
    t_lower = t.lower()

    # Citations — the most specific shapes.
    if _RE_NEUTRAL.match(t):
        return Shape.NEUTRAL_CITATION
    if _RE_ATOID.match(t):
        return Shape.ATOID
    if _RE_PSLA.match(t):
        return Shape.PSLA
    if _RE_SMSFRB.match(t):
        return Shape.SMSFRB
    if _RE_RULING_CITATION.match(t):
        return Shape.RULING_CITATION
    # Structural phrases (exact or close-prefix match on a known set).
    # We match the *start* of the heading since some have trailing qualifiers
    # ("Taxation Ruling" vs "Taxation Ruling - TR 2024/3").
    for sh, phrases in _TYPE_PHRASES.items():
        for phrase in phrases:
            if t_lower == phrase or t_lower.startswith(phrase + " ") or t_lower.startswith(phrase + "\n"):
                return sh
    # Acts / Bills.
    if _RE_ACT_TITLE.match(t):
        return Shape.ACT_TITLE
    if _RE_BILL_TITLE.search(t):
        return Shape.BILL_TITLE
    # Case-name forms.
    if _RE_RE_X.match(t):
        return Shape.RE_X
    if _RE_CASE_NUMBER.match(t):
        return Shape.CASE_NUMBER
    if _RE_NAME_V_NAME.match(t) and len(t) < 200:
        return Shape.NAME_V_NAME
    return Shape.OTHER


# ---------------------------------------------------------------------------
# Templates
#
# Each template has:
# - matches(ins): does this template apply to this doc?
# - extract(ins) -> DerivedMetadata: how to read the fields
#
# Classification iterates templates in priority order; first match wins.
# Extraction is positional (headings[N] reads) with small helper regexes
# to pull year / number / variant tokens out of already-classified strings.


class Template(str, Enum):
    OFFICIAL_PUB = "OFFICIAL_PUB"
    CASE_H1 = "CASE_H1"
    CASE_H2 = "CASE_H2"
    DIS = "DIS"
    ACT = "ACT"
    BILL_EM = "BILL_EM"
    SMSFRB = "SMSFRB"
    EPA = "EPA"
    OTHER = "OTHER"


def classify(ins: RuleInputs) -> Template:
    """Pick a template based on heading shapes + category hints.

    We scan every heading (up to headings[:6]) rather than only h1/h2, so
    noise like a URL as h0 — an extraction artefact on some scraped pages
    — doesn't mask the real citation/type-phrase pair living at h1/h2.

    Priority order (most specific first):
    1. SMSFRB — bulletin shape anywhere in headings.
    2. OFFICIAL_PUB — a type-phrase heading and a citation heading both
       appear in the first several headings (in any order).
    3. DIS — Decision Impact Statement phrase + a name-v-name.
    4. CASE_H1 / CASE_H2 — court-case name forms.
    5. ACT — Act / Regulation title at h1.
    6. BILL_EM — EM / bill title.
    7. EPA — explicit Edited_private_advice category.
    8. OTHER — fallback to docid-only extraction.
    """
    shapes: list[Shape] = [shape_of(h) for h in ins.headings[:6]]
    has = lambda s: s in shapes  # noqa: E731 — tiny helper
    any_citation = any(s in (Shape.RULING_CITATION, Shape.ATOID, Shape.PSLA) for s in shapes)
    any_type_phrase = any(s in (
        Shape.RULING_TYPE_PHRASE, Shape.GUIDELINE_TYPE_PHRASE, Shape.ALERT_PHRASE,
        Shape.ATOID_PHRASE, Shape.PSLA_PHRASE,
    ) for s in shapes)

    # SMSFRB — citation shape anywhere, OR the bulletin phrase.
    if has(Shape.SMSFRB) or has(Shape.SMSFRB_PHRASE):
        return Template.SMSFRB

    # OFFICIAL_PUB — both a type phrase and a citation somewhere. A
    # missing type phrase is tolerated if a citation appears (covers pages
    # where the H1 is a URL artefact).
    if any_citation and (any_type_phrase or True):
        return Template.OFFICIAL_PUB

    # DIS — DIS phrase at any depth, with a case name nearby.
    if has(Shape.DIS_PHRASE) or (
        ins.category == "Decision_impact_statements" and has(Shape.NAME_V_NAME)
    ):
        return Template.DIS

    # Court case — name in h1 or h2, or any case-shape in heading list.
    if shapes and shapes[0] in (Shape.NAME_V_NAME, Shape.RE_X, Shape.NEUTRAL_CITATION, Shape.CASE_NUMBER):
        return Template.CASE_H1
    if len(shapes) >= 2 and shapes[1] == Shape.NAME_V_NAME and ins.category == "Cases":
        return Template.CASE_H2
    if ins.category == "Cases":
        # Any heading has a case shape — route to CASE_H1, extractor will
        # pick the right one.
        if any(s in (Shape.NAME_V_NAME, Shape.RE_X, Shape.NEUTRAL_CITATION,
                     Shape.CASE_NUMBER) for s in shapes):
            return Template.CASE_H1
        # Fallback for Cases — even an OTHER-shaped heading is usually the
        # case's identifier (court name + parties concatenated).
        return Template.CASE_H1

    # Legislation — Act / Regulation / Code.
    if shapes and shapes[0] == Shape.ACT_TITLE:
        return Template.ACT
    if has(Shape.ACT_TITLE) and ins.category == "Legislation_and_supporting_material":
        return Template.ACT

    # Bill / EM.
    if has(Shape.BILL_TITLE) or has(Shape.EM_PHRASE):
        return Template.BILL_EM

    if ins.category == "Edited_private_advice":
        return Template.EPA

    return Template.OTHER


# ---------------------------------------------------------------------------
# Extractors


_RE_CITATION_TOKEN = re.compile(
    rf"^({_RULING_SERIES_ALT})\s+(?P<year>\d{{1,4}})/(?P<draft>D?)(?P<num>\d+)(?P<suffix>[A-Z0-9]*)"
)
_RE_ATOID_TOKEN = re.compile(r"^ATO\s+ID\s+(?P<year>\d{4})/(?P<num>\d+)(?P<suffix>[A-Z0-9]*)")
_RE_PSLA_TOKEN = re.compile(r"^PS\s+LA\s+(?P<year>\d{4})/(?P<draft>D?)(?P<num>\d+)(?P<suffix>[A-Z0-9]*)")
_RE_SMSFRB_TOKEN = re.compile(r"^SMSFRB\s+(?P<year>\d{4})/(?P<num>\d+)")
_RE_NEUTRAL_TOKEN = re.compile(r"^\[(?P<year>\d{4})\]\s+(?P<court>[A-Z]+)\s+(?P<num>\d+)")
_RE_ACT_YEAR = re.compile(r"\b(?P<year>(?:19|20)\d{2})\b")
_RE_BILL_YEAR = re.compile(r"\bBill\s+(?P<year>(?:19|20)\d{2})\b")
_RE_WITHDRAWN = re.compile(r"\(\s*withdrawn\s*\)", re.IGNORECASE)
_RE_REPLACED = re.compile(r"\(\s*(?:replaced|superseded)\b", re.IGNORECASE)
_RE_PRECISE_DATE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1)}
_RE_OLD_REPORT = re.compile(r"\((?P<year>1[89]\d{2}|20\d{2})\)\s+(?:L\.?R\.?|AC|QB|KB|Ch|CLR|ALR|ATC|ATR|FCR|HL|PC|NSWLR|VR|QR|SASR)")


def _clean_citation_with_variant(raw: str) -> tuple[str, str | None, str | None]:
    """Normalise a citation heading to the ATO display form.

    Example: ``'LCR 2019/2EC'`` -> ``('LCR 2019/2EC', 'EC', None)``.
    The ``(Withdrawn)`` marker is stripped. Suffix letters (A=Addendum,
    EC=Erratum Compendium / Consolidated, ER=Erratum, DC=Draft Compendium,
    W=Withdrawn notice) are preserved in the returned display form AND
    surfaced separately as the ``variant`` for programmatic use.
    """
    cleaned = _RE_WITHDRAWN.sub("", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    m = re.match(
        rf"^({_RULING_SERIES_ALT}|ATO\s+ID|PS\s+LA|SMSFRB)\s+(\d{{1,4}})/(D?)(\d+)([A-Z]{{1,2}}\d*)?$",
        cleaned,
    )
    if m:
        series = m.group(1)
        year = m.group(2)
        draft = m.group(3)
        num = m.group(4)
        suffix = m.group(5) or ""
        display = f"{series} {year}/{draft}{num}{suffix}"
        return display, (suffix or None), None
    return cleaned, None, None


def _year_from_token(token: str) -> int | None:
    m = _RE_CITATION_TOKEN.match(token) or _RE_ATOID_TOKEN.match(token) or \
        _RE_PSLA_TOKEN.match(token) or _RE_SMSFRB_TOKEN.match(token) or \
        _RE_NEUTRAL_TOKEN.match(token)
    if m:
        y = m.groupdict().get("year")
        if y:
            try:
                return int(y) if len(y) == 4 else (1900 + int(y))
            except ValueError:
                return None
    return None


def _status_from_text(text: str) -> str | None:
    if not text:
        return None
    if _RE_WITHDRAWN.search(text):
        return "withdrawn"
    if "/D" in text and re.search(r"/D\d", text):
        return "draft"
    if "EC" in text and re.search(r"\d/\d+EC\b", text):
        return "consolidated"
    return None


def _precise_date(text: str) -> str | None:
    m = _RE_PRECISE_DATE.search(text)
    if not m:
        return None
    day, month_name, year = m.groups()
    month = _MONTH[month_name.lower()]
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def _human_title(ins: RuleInputs, *, include_h2: bool = True) -> str | None:
    """Compose a human-readable title from headings.

    Default: ``h1 — h2 — h3`` if all three are present and non-duplicate;
    deduplicated along the way so a doc whose h1 and h2 are identical
    doesn't render as ``"X — X — Y"``.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for h in ins.headings[:3]:
        t = " ".join((h or "").split())
        if not t or t in seen:
            continue
        parts.append(t)
        seen.add(t)
    return " — ".join(parts) if parts else None


# --- Template: Official Publication -----------------------------------------


def _extract_official_pub(ins: RuleInputs) -> DerivedMetadata:
    """Rulings, PCG, TA, PS LA, ATO ID — uniform shape.

    The citation is whichever heading matched RULING_CITATION / ATOID /
    PSLA during classification. We re-scan to locate it because a URL
    artefact at h0 may push the real citation to h3.
    """
    citation_heading: str | None = None
    for h in ins.headings[:6]:
        s = shape_of(h)
        if s in (Shape.RULING_CITATION, Shape.ATOID, Shape.PSLA):
            citation_heading = h
            break
    if citation_heading is None:
        # Classifier routed us here but no citation shape matched — fall
        # back to docid.
        return _extract_other(ins)
    cleaned, variant, _ = _clean_citation_with_variant(citation_heading)
    year = _year_from_token(cleaned)
    status = _status_from_text(citation_heading)
    precise = _precise_date(ins.body_head[:600])
    first_pub = precise or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=cleaned or None,
        human_title=_human_title(ins),
        first_published_date=first_pub,
        citation_year=year,
        variant=variant,
        status=status,
    )


# --- Templates: Court Cases -------------------------------------------------


def _case_name_from(heading: str) -> str | None:
    t = " ".join(heading.split())
    # Reject fragments that look too long to be a case name.
    if not t or len(t) > 200:
        return None
    # Trim trailing "[YYYY] COURT N..." if it's attached to the name.
    t = re.sub(r"\s*\[\d{4}\].*$", "", t).strip()
    # Normalise "v." -> "v".
    t = re.sub(r"\bv\.\s+", "v ", t)
    return t


def _extract_case_h1(ins: RuleInputs) -> DerivedMetadata:
    """Court case — case identifier in one of the first few headings.

    The heading that carries the identifier varies:
    - Modern docs: h1 is the case name.
    - Legacy docs: h1 is the court name and h2 / a later heading has the
      case name, OR h1 is a concatenated "COURT — PARTY v PARTY — JUDGE"
      string (a single heading with em-dash separators).
    - Board of Review cases: h1 is "Case 9/93" or "Case K68".

    Strategy: scan the first few headings, extract the best-looking case
    identifier, and use that as human_code. Year comes from a neutral
    citation or old-report citation if present in title/headings/body_head.
    """
    human_code: str | None = None
    year: int | None = None

    # Pass 1: direct heading match on a case-specific shape.
    for h in ins.headings[:5]:
        s = shape_of(h)
        if s == Shape.NEUTRAL_CITATION:
            m = _RE_NEUTRAL_TOKEN.match(h.strip())
            if m:
                human_code = f"[{m['year']}] {m['court']} {m['num']}"
                year = int(m["year"])
                break
        if s == Shape.NAME_V_NAME:
            human_code = _case_name_from(h)
            break
        if s == Shape.RE_X:
            human_code = _case_name_from(h)
            break
        if s == Shape.CASE_NUMBER:
            human_code = " ".join(h.split())
            break

    # Pass 2: split em-dash-joined headings ("COURT — PARTY v PARTY — JUDGE")
    # on " — " and look for a case-name sub-part.
    if human_code is None:
        for h in ins.headings[:3]:
            for part in re.split(r"\s+—\s+", h):
                part = " ".join(part.split())
                ps = shape_of(part)
                if ps == Shape.NAME_V_NAME and part != h:
                    human_code = _case_name_from(part)
                    break
                if ps == Shape.NEUTRAL_CITATION:
                    m = _RE_NEUTRAL_TOKEN.match(part)
                    if m:
                        human_code = f"[{m['year']}] {m['court']} {m['num']}"
                        year = int(m["year"])
                        break
            if human_code:
                break

    # Pass 3: the Cases bucket's ultimate fallback — use the first non-URL
    # heading as-is. Not ideal but beats leaving human_code NULL for a
    # bucket where EVERY doc has SOME identifier.
    if human_code is None and ins.category == "Cases":
        for h in ins.headings[:3]:
            clean = " ".join(h.split())
            if clean and not clean.startswith("/law/view/") and len(clean) < 200:
                human_code = clean
                break

    # Year fallback: neutral citation, then old report, scanned in meta
    # only (never body) to avoid bleeding in cited cases.
    if year is None:
        for src in [ins.title or "", *ins.headings[:5]]:
            m = _RE_NEUTRAL_TOKEN.search(src)
            if m:
                year = int(m["year"])
                break
    if year is None:
        old = _RE_OLD_REPORT.search(ins.body_head[:400])
        if old:
            year = int(old["year"])

    precise = _precise_date(ins.body_head[:600])
    first_pub = precise or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=human_code,
        human_title=_human_title(ins),
        first_published_date=first_pub,
        citation_year=year,
    )


def _extract_case_h2(ins: RuleInputs) -> DerivedMetadata:
    """Court case layouts where h2 carries the case name."""
    human_code = _case_name_from(ins.h2)
    neutral = _RE_NEUTRAL_TOKEN.search(ins.body_head[:500])
    year = int(neutral["year"]) if neutral else None
    if year is None:
        old = _RE_OLD_REPORT.search(ins.body_head[:500])
        if old:
            year = int(old["year"])
    precise = _precise_date(ins.body_head[:600])
    first_pub = precise or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=human_code,
        human_title=_human_title(ins),
        first_published_date=first_pub,
        citation_year=year,
    )


def _extract_dis(ins: RuleInputs) -> DerivedMetadata:
    """Decision Impact Statement — h1 is DIS phrase, h2 is the case name."""
    case_name = _case_name_from(ins.h2) if ins.h2 else _case_name_from(ins.h1)
    # Try to pull year from body first few lines — DIS docs list the
    # underlying case's citation early.
    neutral = _RE_NEUTRAL_TOKEN.search(ins.body_head[:1200])
    year = int(neutral["year"]) if neutral else None
    precise = _precise_date(ins.body_head[:600])
    first_pub = precise or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=case_name,
        human_title=_human_title(ins),
        first_published_date=first_pub,
        citation_year=year,
    )


# --- Template: Legislation / Act --------------------------------------------


def _extract_act(ins: RuleInputs) -> DerivedMetadata:
    """Act / Regulation / Code title in h1.

    human_code = h1 (the Act's full name, which IS its canonical citation).
    citation_year = the year in the Act's name (Income Tax Assessment
    Act 1997 -> 1997 — authoritative for this doc type).
    """
    title = " ".join(ins.h1.split())
    ym = _RE_ACT_YEAR.search(title)
    year = int(ym["year"]) if ym else None
    return DerivedMetadata(
        human_code=title or None,
        human_title=_human_title(ins),
        first_published_date=f"{year}-01-01" if year else None,
        citation_year=year,
    )


# --- Template: Bill / Explanatory Memorandum --------------------------------


def _extract_bill_em(ins: RuleInputs) -> DerivedMetadata:
    """Bill / EM — no canonical short code but we can give a useful label.

    Prefer the Bill title in h2 if present; otherwise h1. Year from the
    "Bill YYYY" token.
    """
    source = ins.h2 if _RE_BILL_YEAR.search(ins.h2) else ins.h1
    title = " ".join(source.split())
    ym = _RE_BILL_YEAR.search(title) or _RE_ACT_YEAR.search(title)
    year = int(ym["year"]) if ym else None
    label = None
    if title:
        if "Explanatory" not in title and ym:
            label = f"EM to {title}"
        else:
            label = title
    return DerivedMetadata(
        human_code=label,
        human_title=_human_title(ins),
        first_published_date=f"{year}-01-01" if year else None,
        citation_year=year,
    )


# --- Template: SMSF Regulator's Bulletin ------------------------------------


def _extract_smsfrb(ins: RuleInputs) -> DerivedMetadata:
    for h in ins.headings[:4]:
        m = _RE_SMSFRB_TOKEN.search(h)
        if m:
            year = int(m["year"])
            return DerivedMetadata(
                human_code=f"SMSFRB {m['year']}/{m['num']}",
                human_title=_human_title(ins),
                first_published_date=f"{year}-01-01",
                citation_year=year,
            )
    # Fall through — no citation found; use docid body if possible.
    return _extract_other(ins)


# --- Templates: EPA and OTHER (fallback) ------------------------------------


_DOCID_YEAR4_RE = re.compile(
    rf"^({_RULING_SERIES_ALT})(?P<year>(?:19|20)\d{{2}})(?P<draft>D?)(?P<num>\d+)$"
)
_DOCID_YEAR2_RE = re.compile(
    rf"^({_RULING_SERIES_ALT})(?P<year>[89]\d)(?P<num>\d+)$"
)
_DOCID_PSLA_RE = re.compile(r"^PSLA(?P<year>\d{4})(?P<num>\d+)$")
_DOCID_PSLA_DRAFT_RE = re.compile(r"^PSD(?P<year>\d{4})D?(?P<num>\d+)$")
_DOCID_ATOID_RE = re.compile(r"^(?:ATOID|AID)(?P<year>\d{4})(?P<num>\d+)$")


def _extract_from_docid(ins: RuleInputs) -> tuple[str | None, int | None, str | None]:
    body = ins.inner_body
    m = _DOCID_YEAR4_RE.match(body)
    if m:
        series = m.group(1)
        y = int(m["year"])
        draft = m["draft"] or ""
        return f"{series} {m['year']}/{draft}{m['num']}", y, ("draft" if draft else None)
    m = _DOCID_PSLA_RE.match(body)
    if m:
        y = int(m["year"])
        return f"PS LA {m['year']}/{m['num']}", y, None
    m = _DOCID_PSLA_DRAFT_RE.match(body)
    if m:
        y = int(m["year"])
        return f"PS LA {m['year']}/D{m['num']}", y, "draft"
    m = _DOCID_ATOID_RE.match(body)
    if m:
        y = int(m["year"])
        return f"ATO ID {m['year']}/{m['num']}", y, None
    m = _DOCID_YEAR2_RE.match(body)
    if m:
        series = m.group(1)
        y = 1900 + int(m["year"])
        return f"{series} {m['year']}/{m['num']}", y, None
    return None, None, None


def _extract_epa(ins: RuleInputs) -> DerivedMetadata:
    """Edited private advice — no citable short form; leave human_code NULL.

    We can still pull a year from the docid body (EV/1012345678901 has
    no year; most EPA docids don't), or from pub_date if one was scraped.
    """
    year = None
    if ins.pub_date and len(ins.pub_date) >= 4 and ins.pub_date[:4].isdigit():
        year = int(ins.pub_date[:4])
    first_pub = ins.pub_date or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=None,
        human_title=None,
        first_published_date=first_pub,
        citation_year=year,
    )


def _extract_other(ins: RuleInputs) -> DerivedMetadata:
    """Fallback — try docid + pub_date; leave NULL if nothing fires."""
    code, year, status = _extract_from_docid(ins)
    if ins.pub_date and len(ins.pub_date) >= 4 and ins.pub_date[:4].isdigit():
        y = int(ins.pub_date[:4])
        if year is None:
            year = y
    precise = _precise_date(ins.body_head[:600])
    first_pub = precise or ins.pub_date or (f"{year}-01-01" if year else None)
    return DerivedMetadata(
        human_code=code,
        human_title=_human_title(ins),
        first_published_date=first_pub,
        citation_year=year,
        status=status,
    )


# ---------------------------------------------------------------------------
# Public API


_EXTRACTORS: dict[Template, Callable[[RuleInputs], DerivedMetadata]] = {
    Template.OFFICIAL_PUB: _extract_official_pub,
    Template.CASE_H1: _extract_case_h1,
    Template.CASE_H2: _extract_case_h2,
    Template.DIS: _extract_dis,
    Template.ACT: _extract_act,
    Template.BILL_EM: _extract_bill_em,
    Template.SMSFRB: _extract_smsfrb,
    Template.EPA: _extract_epa,
    Template.OTHER: _extract_other,
}


def derive_metadata(inputs: RuleInputs) -> DerivedMetadata:
    """Classify the page's template, then run that template's extractor."""
    template = classify(inputs)
    result = _EXTRACTORS[template](inputs)
    # If the template extractor came up empty on human_code, try the docid
    # fallback so e.g. a scraper-damaged OFFICIAL_PUB page (no h2) still
    # surfaces its canonical code.
    if result.human_code is None and template != Template.EPA:
        fallback_code, fb_year, fb_status = _extract_from_docid(inputs)
        if fallback_code:
            result = replace(
                result,
                human_code=fallback_code,
                citation_year=result.citation_year or fb_year,
                status=result.status or fb_status,
            )
    return result


def template_of(inputs: RuleInputs) -> Template:
    """Expose the classification decision (used by tests and for debugging)."""
    return classify(inputs)


# Backwards-compat shim so the old v1 call site keeps working.
def human_code_for_doc_id(doc_id: str) -> str | None:
    code, _year, _status = _extract_from_docid(RuleInputs(doc_id=doc_id))
    return code
