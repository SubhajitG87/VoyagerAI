"""Command line entry points for VoyagerAI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from voyager_ai.config import MODEL_PRESETS, MODEL_RATIONALE, Settings, load_settings
from voyager_ai.constants import VIBES
from voyager_ai.events import PlanEvent
from voyager_ai.models import Destination, TravelPreferences, ValidationError, VoyagerError
from voyager_ai.planner import TravelPlanner
from voyager_ai.render import render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValidationError, VoyagerError) as exc:
        print(f"VoyagerAI error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voyager-ai",
        description="Multi-agent travel research using local/open-model LLM backends.",
    )
    sub = parser.add_subparsers(required=True)

    discover = sub.add_parser("discover", help="Suggest destinations for a traveller profile.")
    add_common_args(discover)
    discover.add_argument("--avoid", default="", help="Comma-separated destinations to avoid.")
    discover.set_defaults(func=cmd_discover)

    plan = sub.add_parser("plan", help="Create a full trip plan.")
    add_common_args(plan)
    plan.add_argument("--destination", required=True, help="Destination city or region.")
    plan.add_argument("--country", required=True, help="Destination country.")
    plan.add_argument(
        "--why",
        default="Selected by the user.",
        help="Short reason for the destination.",
    )
    plan.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Destination tag. Can be repeated.",
    )
    plan.add_argument("--out", default="", help="Write markdown to this path.")
    plan.set_defaults(func=cmd_plan)

    models = sub.add_parser("models", help="Print the configured model presets.")
    models.set_defaults(func=cmd_models)

    web = sub.add_parser("web", help="Launch the Streamlit web app.")
    web.set_defaults(func=cmd_web)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from", dest="departure_city", required=True, help="Departure city.")
    parser.add_argument("--nationality", default="", help="Traveller nationality.")
    parser.add_argument("--months", type=int, default=6, choices=[3, 6, 12], help="Travel window.")
    parser.add_argument("--nights", type=int, default=7, help="Trip length in nights.")
    parser.add_argument("--vibe", choices=sorted(VIBES), default="city", help="Travel style.")
    parser.add_argument("--direct", action="store_true", help="Prefer direct flights only.")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible", "nim", "demo"],
        default="",
        help="Override VOYAGER_LLM_PROVIDER for this run.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable live web search snippets for this run.",
    )


def preferences_from_args(args: argparse.Namespace) -> TravelPreferences:
    return TravelPreferences(
        departure_city=args.departure_city,
        nationality=args.nationality,
        months=args.months,
        direct_flights=args.direct,
        nights=args.nights,
        vibe=args.vibe,
    )


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    if getattr(args, "provider", ""):
        settings.provider = args.provider
        # The model map depends on the provider (nim has its own models).
        settings.use_preset(settings.model_preset)
    if getattr(args, "no_search", False):
        settings.search_enabled = False
    return settings


def cmd_discover(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    planner = TravelPlanner(settings=settings)
    avoid = [item.strip() for item in args.avoid.split(",") if item.strip()]
    destinations = planner.discover_destinations(preferences_from_args(args), avoid=avoid)
    print(
        json.dumps(
            [destination.to_dict() for destination in destinations],
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    planner = TravelPlanner(settings=settings)
    preferences = preferences_from_args(args)
    destination = Destination(
        name=args.destination,
        country=args.country,
        why=args.why,
        tags=args.tag,
    )

    out_path = _prepare_out_path(args.out)
    trip = planner.plan_trip(preferences, destination, on_event=print_event)
    markdown = render_markdown(trip)
    if out_path:
        path = out_path
        try:
            path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            # Do not throw away minutes of model work because of a bad path.
            print(markdown)
            raise VoyagerError(
                f"could not write {path}: {exc.strerror or exc}. "
                "The plan was printed to standard output instead."
            ) from exc
        print(f"Wrote {path}")
    else:
        print(markdown)
    return 0 if not trip.errors else 2


def print_event(event: PlanEvent) -> None:
    """Print one plan event as a progress line on stderr. Chunks are not printed."""

    line = _event_line(event)
    if line:
        print(line, file=sys.stderr)


def _event_line(event: PlanEvent) -> str:
    agent = event.agent_id
    if event.kind == "agent_started":
        return f"[{agent}] started"
    if event.kind == "agent_failed":
        return f"[{agent}] failed: {event.text}"
    if event.kind == "agent_finished":
        result = event.result
        if result is not None and result.itinerary is not None:
            return f"[{agent}] done: {len(result.itinerary.days)} days"
        report = result.report if result is not None else None
        findings = len(report.findings) if report else 0
        sourced = report.sourced_count if report else 0
        return f"[{agent}] done: {findings} findings, {sourced} sourced"
    if event.kind == "plan_finished":
        return "plan finished"
    return ""


def _prepare_out_path(out: str) -> Path | None:
    """Check the --out target before any model call is made."""

    if not out:
        return None
    path = Path(out).expanduser()
    if path.is_dir():
        raise VoyagerError(f"could not write {path}: it is a directory")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VoyagerError(f"could not write {path}: {exc.strerror or exc}") from exc
    return path


def cmd_models(args: argparse.Namespace) -> int:
    print("Model presets:")
    for preset, mapping in MODEL_PRESETS.items():
        print(f"\n[{preset}]")
        for agent_id, model in mapping.items():
            print(f"  {agent_id:<10} {model}")
    print("\nRationale:")
    for family, rationale in MODEL_RATIONALE.items():
        print(f"  {family:<18} {rationale}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    app_path = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        return subprocess.call(command)
    except FileNotFoundError:
        print("Streamlit is not installed. Run: pip install -e .", file=sys.stderr)
        return 1
