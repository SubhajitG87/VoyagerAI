"""Suite-wide guards: no network, and no dependence on the developer's own settings."""

from __future__ import annotations

import os
import socket
import sys
import types

import pytest


class NetworkBlocked(BaseException):
    """Deliberately not an Exception, so `except Exception` blocks cannot swallow it."""


def _blocked(*args: object, **kwargs: object) -> None:
    raise NetworkBlocked("Network access is blocked in tests")


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return  # opt-in live tests are meant to reach the network
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    # The search library uses its own HTTP stack, so it is replaced as a whole.
    # Tests that exercise search put their own mock module in sys.modules.
    stub = types.ModuleType("ddgs")
    stub.DDGS = _blocked  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", stub)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", stub)


@pytest.fixture(autouse=True)
def _isolated_settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return  # live tests read the real environment and .env file
    monkeypatch.setattr("voyager_ai.config._load_dotenv_if_available", lambda: None)
    for name in list(os.environ):
        if name.startswith(("VOYAGER_", "OLLAMA_", "OPENAI_COMPATIBLE_")):
            monkeypatch.delenv(name)
