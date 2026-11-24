"""Smoke test for scripts/probe_schema.py: it imports and its --help works, with no network."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_schema.py"


def _load(monkeypatch):
    spec = importlib.util.spec_from_file_location("probe_schema", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up in sys.modules while the script runs.
    monkeypatch.setitem(sys.modules, "probe_schema", module)
    spec.loader.exec_module(module)
    return module


def test_help_lists_the_real_schemas(capsys, monkeypatch):
    probe = _load(monkeypatch)
    with pytest.raises(SystemExit) as info:
        probe.main(["--help"])
    assert info.value.code == 0
    assert "--schemas" in capsys.readouterr().out
    assert [p.name for p in probe.SCHEMA_SETS["real"]] == ["simple", "research", "itinerary"]


def test_real_probe_replies_are_validated_against_their_schema(monkeypatch):
    probe = _load(monkeypatch)
    research = probe.SCHEMA_SETS["real"][1]
    good = '{"summary": "S", "findings": []}'
    assert probe.validate(good, research.schema) is None
    assert "schema violation" in probe.validate('{"summary": "S"}', research.schema)
