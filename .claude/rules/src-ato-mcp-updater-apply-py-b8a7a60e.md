---
paths:
  - "src/ato_mcp/updater/apply.py"
---

# src/ato_mcp/updater/apply.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Update Mechanism
End-user update flow: manifest diff, pack fetch, in-place SQLite patch application, lock.

- [UM-05 L17] apply_update flow: diff manifests → group changed+added docs by pack_sha8 → fetch needed byte ranges → snapshot ato.db to backups/ato.db.prev → mutate documents/chunks/chunks_fts/chunks_vec/title_fts in ONE SQLite transaction → touch meta.last_update_at → write installed_manifest.json LAST (pointer-after-data so a partial run can't claim a successful update).
- [UM-06 L18] Rollback path: copy backups/ato.db.prev back over ato.db and drop the new manifest. The serve process picks up the rollback automatically via its mtime-based connection refresh; doctor --rollback wraps this in a safety check.
