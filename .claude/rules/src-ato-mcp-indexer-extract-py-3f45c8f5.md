---
paths:
  - "src/ato_mcp/indexer/extract.py"
---

# src/ato_mcp/indexer/extract.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Index Builder
Build orchestration, heading-aware chunking with overlap, token estimation, packing, manifest, release.

- [IB-07 L144] Document title is composed from a small number of leading headings (h1=doc_type, h2=code, h3=subject on rulings) via _compose_title with prefix-overlap suppression; falls back to the raw <title> when no leading headings present.
- [IB-06 L167] HTML container is picked from a fallback chain (#LawContent → #lawContents → #contents → #content → article → main → body), absorbing the various wrapper IDs ATO has used over the years.
- [IB-08 L193] extract injects heading id attributes back into the markdown as ' {#anchor}' suffixes so chunks can reference sections directly; markdownify runs with heading_style=ATX, bullets='-', and script/style/iframe stripped.
