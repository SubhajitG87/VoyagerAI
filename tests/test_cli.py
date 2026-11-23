import argparse
import json
from pathlib import Path

from voyager_ai.cli import main, settings_from_args
from voyager_ai.config import MODEL_PRESETS, NIM_MODELS

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
    markdown = out.read_text(encoding="utf-8")
    assert markdown.startswith("# VoyagerAI Trip Plan: Tokyo, Japan")
    assert markdown.count("### Day ") == 8
    assert "## Budget" in markdown and "## Visa Advisor" in markdown


def test_plan_out_unwritable_path_exits_cleanly(tmp_path, capsys):
    assert main([*PLAN, "--out", str(tmp_path)]) == 1
    assert "VoyagerAI error: could not write" in capsys.readouterr().err


def test_discover_prints_eight_destinations(capsys):
    assert main(["discover", "--from", "Mumbai", "--provider", "demo", "--no-search"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 8


def test_plan_out_directory_is_rejected_before_planning(tmp_path, capsys):
    assert main([*PLAN, "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "VoyagerAI error: could not write" in err
    assert "started" not in err


def test_plan_is_printed_when_the_final_write_fails(tmp_path, capsys, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", refuse)
    assert main([*PLAN, "--out", str(tmp_path / "tokyo.md")]) == 1
    out, err = capsys.readouterr()
    assert out.startswith("# VoyagerAI Trip Plan: Tokyo, Japan")
    assert "VoyagerAI error: could not write" in err


def test_provider_flag_rebuilds_model_map(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("VOYAGER_MODEL_PRESET", "small")
    settings = settings_from_args(argparse.Namespace(provider="nim"))
    assert settings.model_map == NIM_MODELS

    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "nim")
    settings = settings_from_args(argparse.Namespace(provider="ollama"))
    assert settings.model_map == MODEL_PRESETS["small"]


def test_plan_prints_agent_progress(capsys):
    assert main([*PLAN, "--nights", "3"]) == 0
    err = capsys.readouterr().err
    assert "[visa] started" in err and "[itinerary] done: 4 days" in err
    assert err.rstrip().endswith("plan finished")
    visa = next(line for line in err.splitlines() if line.startswith("[visa] done:"))
    assert visa.endswith("sourced") and " findings, " in visa
    # Chunks are not printed: every line is one of the fixed progress lines.
    for line in err.strip().splitlines():
        assert line == "plan finished" or line.split("] ", 1)[1].startswith(
            ("started", "done: ", "failed: ")
        )


def test_failed_agent_line(capsys):
    from voyager_ai.cli import print_event
    from voyager_ai.events import PlanEvent

    print_event(PlanEvent("agent_failed", "hotel", text="model timed out"))
    print_event(PlanEvent("agent_chunk", "hotel", text="{partial"))
    assert capsys.readouterr().err == "[hotel] failed: model timed out\n"


def test_failed_line_falls_back_when_error_is_empty(capsys):
    from voyager_ai.cli import print_event
    from voyager_ai.events import PlanEvent

    print_event(PlanEvent("agent_failed", "hotel", text=""))
    assert capsys.readouterr().err == "[hotel] failed: unknown error\n"


def test_ctrl_c_prints_cancelled_then_exits_hard(capsys, monkeypatch):
    import voyager_ai.cli as cli
    from voyager_ai.planner import TravelPlanner

    class HardExit(Exception):
        pass

    codes = []

    def fake_exit(code):
        codes.append(code)
        raise HardExit

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_hard_exit", fake_exit)
    monkeypatch.setattr(TravelPlanner, "plan_trip", interrupt)
    try:
        main(PLAN)
    except HardExit:
        pass
    assert codes == [130]
    assert "Cancelled." in capsys.readouterr().err
