from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Union
from urllib.parse import urlencode

import requests

LOGGER = logging.getLogger(__name__)


class AtoBrowseClientError(Exception):
	"""Raised when the ATO browse-content API cannot be reached or parsed."""


@dataclass
class _HttpResponse:
	status_code: int
	text: str

	def json(self) -> Any:
		return json.loads(self.text)


class AtoBrowseClient:
	"""Thin wrapper around the ATO browse-content API."""

	def __init__(
		self,
		base_url: str = "https://www.ato.gov.au/API/v1/law/lawservices/browse-content/",
		timeout: float = 30.0,
		transport: Optional[Callable[[str], _HttpResponse]] = None,
	) -> None:
		self.base_url = base_url.rstrip("?")
		self.timeout = timeout
		self._transport = transport
		self._session = requests.Session() if transport is None else None

	def fetch_nodes(self, query: Union[str, Dict[str, str]]) -> Iterable[Dict[str, Any]]:
		"""Fetch a node list. ``query`` may be dict params or a raw query string."""

		url = self._build_url(query)
		response = self._make_request(url)

		try:
			payload = response.json()
		except ValueError as exc:  # pragma: no cover - network parsing guard
			raise AtoBrowseClientError("ATO response was not valid JSON") from exc

		if not isinstance(payload, list):
			raise AtoBrowseClientError("ATO response payload is not a list")

		return payload

	def _make_request(self, url: str) -> _HttpResponse:
		LOGGER.debug("Fetching %s", url)
		if self._transport is not None:
			return self._transport(url)

		try:  # pragma: no cover - exercised only with live HTTP
			resp = self._session.get(url, timeout=self.timeout)
			resp.raise_for_status()
			return _HttpResponse(status_code=resp.status_code, text=resp.text)
		except requests.RequestException as exc:  # pragma: no cover - live network
			raise AtoBrowseClientError("Failed to reach ATO API") from exc

	def _build_url(self, query: Union[str, Dict[str, str]]) -> str:
		if isinstance(query, str):
			query_string = query.lstrip("?")
		else:
			query_string = urlencode(query, safe=":#")

		if query_string:
			return f"{self.base_url}?{query_string}"
		return self.base_url
