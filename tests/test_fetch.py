"""Tests for the guarded page fetcher. No sockets: httpx.MockTransport and an injected resolver."""

from __future__ import annotations

import logging
from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest

from voyager_ai.fetch import NoopFetcher, PageFetcher, default_resolve, is_public_http_url

ARTICLE = (
    "<html><head><title>T</title></head><body><article><p>Visa-free for 90 days.</p>"
    "<p>" + "Detail. " * 2000 + "</p></article></body></html>"
)
PUBLIC = "93.184.216.34"


def public(host: str) -> list[str]:
    return [PUBLIC]


def _fetcher(
    handler: Callable[[httpx.Request], httpx.Response],
    resolve: Callable[[str], list[str]] = public,
    **kwargs: object,
) -> PageFetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("check_robots", False)
    return PageFetcher(resolve=resolve, client=client, **kwargs)  # type: ignore[arg-type]


def html_handler(body: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, content=body
        )

    return handler


def test_extracts_and_truncates_text() -> None:
    page = _fetcher(html_handler(ARTICLE)).fetch("https://example.org/visa", "Visa")
    assert page and "Visa-free for 90 days." in page.text and len(page.text) <= 4000
    assert page.url == "https://example.org/visa" and page.title == "Visa"


def test_empty_extraction_returns_none() -> None:
    assert _fetcher(html_handler("<html><body></body></html>")).fetch("https://e.org/") is None


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_private_addresses_are_not_fetched(ip: str) -> None:
    handler = MagicMock()
    assert _fetcher(handler, resolve=lambda h: [ip]).fetch("https://intranet.example/") is None
    handler.assert_not_called()


def test_any_private_address_among_several_blocks() -> None:
    handler = MagicMock()
    fetcher = _fetcher(handler, resolve=lambda h: [PUBLIC, "10.0.0.1"])
    assert fetcher.fetch("https://mixed.example/") is None
    handler.assert_not_called()


def test_is_public_http_url_rules() -> None:
    assert is_public_http_url("https://e.org/", public)
    assert not is_public_http_url("ftp://e.org/", public)
    assert not is_public_http_url("file:///etc/passwd", public)
    assert not is_public_http_url("https:///nohost", public)
    assert not is_public_http_url("not a url", public)

    def boom(host: str) -> list[str]:
        raise OSError("dns")

    assert not is_public_http_url("https://e.org/", boom)
    assert not is_public_http_url("https://e.org/", lambda h: [])
    assert not is_public_http_url("https://e.org/", lambda h: ["0.0.0.0"])
    assert not is_public_http_url("https://e.org/", lambda h: ["224.0.0.1"])
    assert not is_public_http_url("https://e.org/", lambda h: ["not-an-ip"])
    assert not is_public_http_url("https://e.org/", lambda h: ["100.64.0.1"])
    assert not is_public_http_url("https://e.org/", lambda h: ["100.100.100.200"])
    assert not is_public_http_url("https://e.org/", lambda h: ["::ffff:10.0.0.1"])
    assert not is_public_http_url("https://e.org/", lambda h: ["::ffff:127.0.0.1"])


def test_redirect_to_private_address_is_blocked() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://internal.example/"})

    def resolve(host: str) -> list[str]:
        return ["10.0.0.1"] if host == "internal.example" else [PUBLIC]

    assert _fetcher(handler, resolve=resolve).fetch("https://e.org/start") is None
    assert seen == ["https://e.org/start"]


def test_relative_redirect_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return html_handler(ARTICLE)(request)

    page = _fetcher(handler).fetch("https://e.org/old")
    assert page and "Visa-free" in page.text


def test_too_many_redirects_gives_up() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": f"/r{len(calls)}"})

    assert _fetcher(handler).fetch("https://e.org/") is None
    assert len(calls) == 4  # the original request plus three hops


def test_non_html_and_oversized_are_skipped() -> None:
    def handler_with(
        content_type: str = "text/html",
        content_length: int | None = None,
        body: bytes = b"<p>x</p>",
    ) -> Callable[[httpx.Request], httpx.Response]:
        def handler(request: httpx.Request) -> httpx.Response:
            headers = {"content-type": content_type}
            if content_length is not None:
                headers["content-length"] = str(content_length)
            return httpx.Response(200, headers=headers, content=body)

        return handler

    assert _fetcher(handler_with(content_type="application/pdf")).fetch("https://e.org/a") is None
    assert _fetcher(handler_with(content_length=3_000_000)).fetch("https://e.org/b") is None
    assert _fetcher(handler_with(body=b"x" * 2_000_001)).fetch("https://e.org/c") is None


def test_streaming_stops_reading_past_max_bytes() -> None:
    chunks_read = 0

    def body() -> object:
        nonlocal chunks_read
        for _ in range(1000):
            chunks_read += 1
            yield b"x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    assert _fetcher(handler, max_bytes=5000).fetch("https://e.org/") is None
    assert chunks_read < 50


def test_http_errors_and_exceptions_return_none_and_log(caplog: pytest.LogCaptureFixture) -> None:
    def err(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    with caplog.at_level(logging.DEBUG, logger="voyager_ai.fetch"):
        assert _fetcher(err).fetch("https://e.org/") is None
        assert _fetcher(boom).fetch("https://e.org/") is None
    assert caplog.records and all(r.levelno == logging.DEBUG for r in caplog.records)


def _robots_handler(robots: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return robots
        return html_handler(ARTICLE)(request)

    return handler


def test_robots_disallow_is_respected() -> None:
    robots = httpx.Response(200, content="User-agent: *\nDisallow: /")
    assert _fetcher(_robots_handler(robots), check_robots=True).fetch("https://e.org/a") is None


@pytest.mark.parametrize("status", [404, 500])
def test_robots_error_status_allows(status: int) -> None:
    fetcher = _fetcher(_robots_handler(httpx.Response(status)), check_robots=True)
    assert fetcher.fetch("https://e.org/a") is not None


def test_robots_fetch_failure_allows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            raise httpx.ConnectError("nope")
        return html_handler(ARTICLE)(request)

    assert _fetcher(handler, check_robots=True).fetch("https://e.org/a") is not None


def test_robots_is_cached_per_host() -> None:
    robots_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_calls
        if request.url.path == "/robots.txt":
            robots_calls += 1
            return httpx.Response(200, content="User-agent: *\nDisallow: /private")
        return html_handler(ARTICLE)(request)

    fetcher = _fetcher(handler, check_robots=True)
    assert fetcher.fetch("https://e.org/a")
    assert fetcher.fetch("https://e.org/b")
    assert fetcher.fetch("https://e.org/private/x") is None
    assert robots_calls == 1


def test_robots_redirect_to_private_address_is_ignored() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.path == "/robots.txt":
            return httpx.Response(302, headers={"location": "http://internal.example/robots.txt"})
        return html_handler(ARTICLE)(request)

    def resolve(host: str) -> list[str]:
        return ["10.0.0.1"] if host == "internal.example" else [PUBLIC]

    assert _fetcher(handler, resolve=resolve, check_robots=True).fetch("https://e.org/a")
    assert "internal.example" not in seen


def test_noop_fetcher() -> None:
    assert NoopFetcher().fetch("https://e.org/", "t") is None


def test_default_resolve_returns_unique_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    infos = [
        (2, 1, 6, "", ("1.2.3.4", 0)),
        (2, 2, 17, "", ("1.2.3.4", 0)),
        (10, 1, 6, "", ("::1", 0, 0, 0)),
    ]
    monkeypatch.setattr("voyager_ai.fetch.socket.getaddrinfo", lambda *a, **k: infos)
    assert default_resolve("x") == ["1.2.3.4", "::1"]


def test_slow_body_is_abandoned_after_total_deadline() -> None:
    now = [0.0]
    chunks_read = 0

    def body() -> object:
        nonlocal chunks_read
        for _ in range(100):
            chunks_read += 1
            now[0] += 9.0  # a trickle: one chunk every 9 seconds
            yield b"<p>x</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    fetcher = _fetcher(handler, total_timeout_s=20.0, clock=lambda: now[0])
    assert fetcher.fetch("https://e.org/") is None
    assert chunks_read <= 3


def test_requests_ask_for_uncompressed_bodies() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["accept-encoding"])
        return html_handler(ARTICLE)(request)

    assert _fetcher(handler).fetch("https://e.org/")
    assert seen == ["identity"]
