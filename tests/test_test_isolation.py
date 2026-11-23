"""The suite must never reach the network or the developer's own .env file."""

import socket

import pytest

from voyager_ai.config import Settings
from voyager_ai.search import DuckDuckGoSearchClient

BLOCKED = "Network access is blocked in tests"


def test_python_sockets_are_blocked():
    with socket.socket() as sock:
        with pytest.raises(BaseException, match=BLOCKED):
            sock.connect(("127.0.0.1", 9))


def test_unmocked_search_is_blocked():
    with pytest.raises(BaseException, match=BLOCKED):
        DuckDuckGoSearchClient().search("anything")


def test_dotenv_file_is_not_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: calls.append(1))
    Settings.from_env()
    assert calls == []


def test_guards_still_apply_to_unmarked_tests(request):
    assert request.node.get_closest_marker("live") is None
    with socket.socket() as sock, pytest.raises(BaseException, match=BLOCKED):
        sock.connect(("127.0.0.1", 9))
