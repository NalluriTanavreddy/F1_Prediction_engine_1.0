"""Shared async HTTP client.

Adapted from BoxBox (https://github.com/NalluriTanavreddy/boxbox)
src/boxbox/utils/http.py.
"""

import httpx

from f1_predict import __version__

DEFAULT_TIMEOUT = 30.0

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": f"f1-predict/{__version__}"},
            follow_redirects=True,
        )
    return _client


async def close_client() -> None:
    """Close the shared client on shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
