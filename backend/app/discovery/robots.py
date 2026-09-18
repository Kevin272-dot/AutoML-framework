"""robots.txt fetching and evaluation.

robots.txt is treated strictly as a TECHNICAL access signal. We never claim that
a permissive robots.txt implies legal permission to scrape; policy status is
reported separately (KNOWN / UNKNOWN / RESTRICTED) based on whether terms and
license documents are locatable.
"""

import asyncio
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


@dataclass
class RobotsResult:
    status: str = "UNKNOWN"  # FOUND | NOT_FOUND | UNKNOWN | ERROR
    allowed: bool | None = None  # None = unknown
    disallowed_paths: list[str] = field(default_factory=list)
    note: str = ""


async def fetch_robots(client: httpx.AsyncClient, robots_url: str) -> tuple[str | None, int | None]:
    try:
        resp = await client.get(robots_url, follow_redirects=True, timeout=15.0)
        if resp.status_code == 200 and "text/plain" in resp.headers.get("content-type", "text/plain"):
            return resp.text, resp.status_code
        return None, resp.status_code
    except httpx.HTTPError:
        return None, None


def evaluate_robots(robots_txt: str | None, status_code: int | None, target_path: str) -> RobotsResult:
    result = RobotsResult()
    if robots_txt is None:
        if status_code == 404:
            result.status = "NOT_FOUND"
            result.allowed = True  # No robots.txt => no technical restrictions declared
            result.note = "No robots.txt published (404)."
        elif status_code is None:
            result.status = "ERROR"
            result.note = "robots.txt could not be fetched."
        else:
            result.status = "UNKNOWN"
            result.note = f"robots.txt returned HTTP {status_code}."
        return result

    result.status = "FOUND"
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())
    useragent = "IntelligentAutoMLBot/0.1"
    result.allowed = parser.can_fetch(useragent, target_path) or parser.can_fetch("*", target_path)
    if not result.allowed:
        for line in robots_txt.splitlines():
            line = line.strip()
            if line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    result.disallowed_paths.append(path)
        result.note = f"robots.txt disallows automated access to {target_path}."
    return result


async def check_url_reachable(client: httpx.AsyncClient, url: str) -> tuple[bool, int | None, str | None]:
    """HEAD first, fall back to GET. Returns (reachable, status_code, content_snippet|None)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, None, "Non-HTTP(S) URL rejected."
    try:
        resp = await client.head(url, follow_redirects=True, timeout=10.0)
        if resp.status_code < 400:
            return True, resp.status_code, None
        resp = await client.get(url, follow_redirects=True, timeout=15.0)
        snippet = resp.text[:2000] if resp.status_code < 400 else None
        return resp.status_code < 400, resp.status_code, snippet
    except httpx.HTTPError:
        return False, None, None


async def probe_api(client: httpx.AsyncClient, api_probe_url: str) -> tuple[bool, str | None]:
    """A documented API endpoint is 'available' when it answers 2xx."""
    try:
        resp = await client.get(api_probe_url, follow_redirects=True, timeout=15.0)
        return resp.status_code < 400, None
    except httpx.HTTPError as exc:
        return False, str(exc.__class__.__name__)


async def gather_with_timeout(coros, timeout: float = 45.0):
    return await asyncio.wait_for(asyncio.gather(*coros), timeout=timeout)
