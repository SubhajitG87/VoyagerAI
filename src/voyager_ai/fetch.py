"""Fetch and extract the readable text of web pages, with safety checks for untrusted URLs."""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from voyager_ai.utils import truncate

logger = logging.getLogger(__name__)

USER_AGENT = "VoyagerAI/0.2 (+https://github.com/SubhajitG87/VoyagerAI)"
MAX_REDIRECTS = 3
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_TYPES = ("text/html", "application/xhtml+xml")

Resolver = Callable[[str], list[str]]


@dataclass
class Page:
    url: str
    title: str
    text: str


def default_resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def is_public_http_url(url: str, resolve: Resolver) -> bool:
    """True for http(s) URLs whose host resolves only to publicly routable addresses."""

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host:
            return False
        addresses = resolve(host)
        if not addresses:
            return False
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%")[0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False
    except Exception:  # unparsable URL, DNS failure or a malformed address: not safe
        return False
    return True


class _Skip(Exception):
    """The URL should not be read; the message says why."""


class PageFetcher:
    def __init__(
        self,
        timeout_s: float = 10.0,
        max_bytes: int = 2_000_000,
        max_chars: int = 4000,
        resolve: Resolver = default_resolve,
        client: httpx.Client | None = None,
        check_robots: bool = True,
    ):
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.resolve = resolve
        self.client = client or httpx.Client(timeout=timeout_s)
        self.check_robots = check_robots
        self._robots: dict[str, RobotFileParser | None] = {}
        self._lock = threading.Lock()

    def fetch(self, url: str, title: str = "") -> Page | None:
        try:
            if not is_public_http_url(url, self.resolve):
                raise _Skip("not a public http(s) URL")
            if self.check_robots and not self._robots_allow(url):
                raise _Skip("disallowed by robots.txt")
            status, content_type, body = self._get(url)
            if status >= 400:
                raise _Skip(f"HTTP {status}")
            if not content_type.lower().startswith(_HTML_TYPES):
                raise _Skip(f"unsupported content type {content_type!r}")
            text = trafilatura.extract(
                body, include_comments=False, include_tables=True, favor_precision=True
            )
            if not text or not text.strip():
                raise _Skip("no readable text")
            return Page(url=url, title=title, text=truncate(text.strip(), self.max_chars))
        except _Skip as exc:
            logger.debug("Skipped %s: %s", url, exc)
        except Exception as exc:
            logger.debug("Could not fetch %s: %s", url, exc, exc_info=True)
        return None

    def _get(self, url: str, html_only: bool = True) -> tuple[int, str, bytes]:
        """GET with manual, checked redirects and a streamed size cap."""

        current = url
        for hop in range(MAX_REDIRECTS + 1):
            if hop and not is_public_http_url(current, self.resolve):
                raise _Skip(f"redirect to non-public URL {current}")
            with self.client.stream(
                "GET",
                current,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=False,
                timeout=self.timeout_s,
            ) as response:
                location = response.headers.get("location")
                if response.status_code in _REDIRECT_STATUSES and location:
                    current = urljoin(current, location)
                    continue
                content_type = response.headers.get("content-type", "")
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > self.max_bytes:
                    raise _Skip("declared size over limit")
                if html_only and response.status_code < 400:
                    if not content_type.lower().startswith(_HTML_TYPES):
                        return response.status_code, content_type, b""
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        raise _Skip("body over size limit")
                return response.status_code, content_type, bytes(body)
        raise _Skip("too many redirects")

    def _robots_allow(self, url: str) -> bool:
        parsed = urlparse(url)
        key = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            known = key in self._robots
            parser = self._robots.get(key)
        if not known:
            parser = self._load_robots(key)
            with self._lock:
                self._robots[key] = parser
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        """Parsed robots.txt, or None (everything allowed) when it is unavailable."""

        robots_url = f"{origin}/robots.txt"
        try:
            if not is_public_http_url(robots_url, self.resolve):
                return None
            status, _, body = self._get(robots_url, html_only=False)
        except Exception as exc:
            logger.debug("No robots.txt for %s: %s", origin, exc)
            return None
        if status >= 400:
            return None
        parser = RobotFileParser()
        parser.parse(body.decode("utf-8", errors="replace").splitlines())
        return parser


class NoopFetcher:
    def fetch(self, url: str, title: str = "") -> Page | None:
        return None
