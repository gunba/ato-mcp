from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator, Optional

_DEFAULT_MAX_WORKERS = 4

_executor: Optional[ThreadPoolExecutor] = None


def get_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
	global _executor
	if _executor is None:
		_executor = ThreadPoolExecutor(max_workers=max_workers or _DEFAULT_MAX_WORKERS)
	return _executor


@contextmanager
def thread_pool(max_workers: Optional[int] = None) -> Iterator[ThreadPoolExecutor]:
	executor = ThreadPoolExecutor(max_workers=max_workers or _DEFAULT_MAX_WORKERS)
	try:
		yield executor
	finally:
		executor.shutdown(wait=True)
