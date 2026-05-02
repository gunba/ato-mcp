"""flock-based single-writer guard for ``ato-mcp update``.

The ``serve`` process opens the DB read-only and does not take the lock. Only
one ``update`` process may hold it at a time. Stale lock files from crashes
are handled naturally: ``flock`` drops the advisory lock when the owning
process exits, even if the file remains.
"""
# [UM-01] apply takes flock; serve opens DB read-only and never takes the lock — read + update coexist. Stale files benign because flock drops on process exit.
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

from ..util import paths


class LockError(RuntimeError):
    pass


@contextlib.contextmanager
def exclusive_lock(path: Path | None = None) -> Iterator[int]:
    """Acquire an exclusive advisory lock on the LOCK file.

    On Linux/macOS uses fcntl.flock (non-blocking); on Windows uses msvcrt.
    Raises :class:`LockError` if another process holds the lock.
    """
    lock_path = Path(path or paths.lock_path())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _lock_fd(fd)
    except OSError as exc:
        os.close(fd)
        raise LockError(f"another ato-mcp process holds {lock_path}") from exc
    try:
        yield fd
    finally:
        _unlock_fd(fd)
        os.close(fd)


def _lock_fd(fd: int) -> None:
    # [UM-02] Cross-platform: fcntl.flock(LOCK_EX | LOCK_NB) on Linux/macOS, msvcrt.locking on Windows. Non-blocking — second writer fails immediately with LockError.
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except ImportError:  # pragma: no cover - windows
        pass
    import msvcrt  # type: ignore[import-not-found]
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _unlock_fd(fd: int) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    except ImportError:  # pragma: no cover - windows
        pass
    import msvcrt  # type: ignore[import-not-found]
    try:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
