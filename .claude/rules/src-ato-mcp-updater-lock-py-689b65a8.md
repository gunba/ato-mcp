---
paths:
  - "src/ato_mcp/updater/lock.py"
---

# src/ato_mcp/updater/lock.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Update Mechanism
End-user update flow: manifest diff, pack fetch, in-place SQLite patch application, lock.

- [UM-01 L8] Single-writer guard: apply_update wraps work in an exclusive flock; the serve process opens DB read-only and never takes the lock, so reading + updating coexist. Stale lock files from crashes are benign — flock drops the advisory lock when the owning process exits, even if the file remains.
- [UM-02 L46] Cross-platform locking: fcntl.flock(LOCK_EX | LOCK_NB) on Linux/macOS, msvcrt.locking on Windows; both are non-blocking — second writer fails immediately with LockError rather than queueing.
