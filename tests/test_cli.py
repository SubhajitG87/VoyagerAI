import json

from voyager_ai.cli import main

PLAN = [
    "plan",
    "--from",
    "Mumbai",
    "--destination",
    "Tokyo",
    "--country",
    "Japan",
    "--provider",
    "demo",
    "--no-search",
]


def test_plan_out_creates_parent_directories(tmp_path):
    out = tmp_path / "trip_plans" / "nested" / "tokyo.md"
    assert main([*PLAN, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# VoyagerAI Trip Plan: Tokyo, Japan")


def test_plan_out_unwritable_path_exits_cleanly(tmp_path, capsys):
    assert main([*PLAN, "--out", str(tmp_path)]) == 1
    assert "VoyagerAI error: could not write" in capsys.readouterr().err


def test_discover_prints_eight_destinations(capsys):
    assert main(["discover", "--from", "Mumbai", "--provider", "demo", "--no-search"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 8
