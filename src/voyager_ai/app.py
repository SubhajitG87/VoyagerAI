"""Streamlit UI for VoyagerAI."""

from __future__ import annotations

import json
from html import escape

import streamlit as st

from voyager_ai.config import MODEL_PRESETS, load_settings
from voyager_ai.constants import AGENT_SEQUENCE, AGENTS, CITIES, NATIONALITIES, VIBES
from voyager_ai.models import (
    AgentResult,
    Destination,
    TravelPreferences,
    ValidationError,
    VoyagerError,
)
from voyager_ai.parser import parse_itinerary
from voyager_ai.planner import TravelPlanner


def main() -> None:
    st.set_page_config(page_title="VoyagerAI", page_icon="✈️", layout="wide")
    inject_css()
    init_state()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Multi-agent travel research</div>
          <h1>VoyagerAI</h1>
          <p>
            Discover destinations, research flights, check visas, compare stays, curate
            experiences, and turn it all into a practical itinerary using local/open-model LLMs.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    settings = sidebar_settings()
    planner = TravelPlanner(settings=settings)
    preferences_panel(planner)
    destination_panel(planner)
    trip_panel()


def init_state() -> None:
    defaults: dict[str, object] = {
        "destinations": [],
        "avoid_destinations": [],
        "trip": None,
        "preferences": None,
        "planning_log": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sidebar_settings():
    settings = load_settings()
    with st.sidebar:
        st.header("Runtime")
        provider = st.selectbox(
            "LLM provider",
            ["ollama", "openai-compatible", "demo"],
            index=["ollama", "openai-compatible", "demo"].index(settings.provider)
            if settings.provider in {"ollama", "openai-compatible", "demo"}
            else 0,
        )
        preset = st.selectbox(
            "Model preset",
            list(MODEL_PRESETS),
            index=list(MODEL_PRESETS).index(settings.model_preset)
            if settings.model_preset in MODEL_PRESETS
            else 1,
        )
        settings.provider = provider
        settings.use_preset(preset)
        settings.search_enabled = st.toggle(
            "Use web search snippets",
            value=settings.search_enabled,
        )
        settings.search_max_results = st.slider(
            "Snippets per query",
            1,
            6,
            settings.search_max_results,
        )

        st.caption(f"Default backend: Ollama at `{settings.ollama_base_url}`.")
        st.caption("Use `demo` for a no-network smoke test.")
        with st.expander("Agent models"):
            for agent_id, model in settings.model_map.items():
                st.code(f"{agent_id}: {model}", language="text")
    return settings


def preferences_panel(planner: TravelPlanner) -> None:
    with st.container(border=True):
        st.subheader("1. Traveller Profile")
        with st.form("profile"):
            col1, col2 = st.columns(2)
            with col1:
                departure_city = st.text_input(
                    "Flying from",
                    value="",
                    placeholder="e.g. Mumbai, London, New York",
                    help=(
                        "You can type any city. Suggestions from the original app are listed below."
                    ),
                )
                st.caption(f"Examples: {', '.join(CITIES[:8])}")
            with col2:
                nationality = st.text_input(
                    "Nationality",
                    value="",
                    placeholder="e.g. Indian, British, American",
                )
                st.caption(f"Examples: {', '.join(NATIONALITIES[:8])}")

            col3, col4, col5 = st.columns(3)
            with col3:
                months = st.radio("Travel window", [3, 6, 12], index=1, horizontal=True)
            with col4:
                nights = st.slider("Trip length", 3, 30, 7)
            with col5:
                direct = st.checkbox("Direct flights only", value=False)

            vibe_label_to_id = {label: key for key, label in VIBES.items()}
            vibe_label = st.selectbox("Travel style", list(vibe_label_to_id))
            submitted = st.form_submit_button("Discover destinations", use_container_width=True)

        if submitted:
            try:
                prefs = TravelPreferences(
                    departure_city=departure_city,
                    nationality=nationality,
                    # Both widgets have a default selected; the fallbacks only satisfy typing.
                    months=months or 6,
                    direct_flights=direct,
                    nights=nights,
                    vibe=vibe_label_to_id.get(vibe_label or "", "city"),
                )
                st.session_state.preferences = prefs
                st.session_state.trip = None
                with st.spinner("Finding destinations that fit the profile..."):
                    st.session_state.destinations = planner.discover_destinations(
                        prefs, avoid=st.session_state.avoid_destinations
                    )
                st.success(f"Found {len(st.session_state.destinations)} destinations.")
            except (ValidationError, VoyagerError) as exc:
                st.error(str(exc))


def destination_panel(planner: TravelPlanner) -> None:
    destinations = st.session_state.destinations
    preferences = st.session_state.preferences
    if not destinations or not preferences:
        return

    st.subheader("2. Choose A Destination")
    cols = st.columns(4)
    for index, destination in enumerate(destinations):
        with cols[index % 4]:
            render_destination(destination)
            if st.button(f"Plan {destination.name}", key=f"plan_{destination.name}_{index}"):
                run_trip(planner, preferences, destination)

    col1, col2 = st.columns([1, 3])
    with col1:
        refresh = st.button("Different destinations", use_container_width=True)
    with col2:
        st.caption("Refreshing asks the discovery agent to avoid the previous destination names.")

    if refresh:
        avoid = merge_avoid_list(
            st.session_state.avoid_destinations, [d.name for d in destinations]
        )
        try:
            with st.spinner("Refreshing destination ideas..."):
                fresh = planner.discover_destinations(preferences, avoid=avoid)
        except (ValidationError, VoyagerError) as exc:
            # Keep the current cards on screen so the user can still pick one.
            st.error(str(exc))
            return
        st.session_state.avoid_destinations = avoid
        st.session_state.destinations = fresh
        st.rerun()


# Most destination names sent back to the discovery agent as "avoid these".
MAX_AVOID_DESTINATIONS = 24


def merge_avoid_list(current: list[str], new: list[str]) -> list[str]:
    """Add names to the avoid list without duplicates, keeping only the most recent ones."""

    merged: list[str] = []
    seen: set[str] = set()
    for name in reversed([*current, *new]):
        key = name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(name)
    return list(reversed(merged[:MAX_AVOID_DESTINATIONS]))


def destination_card_html(destination: Destination) -> str:
    """Build the destination card markup. All model-supplied text is escaped."""

    tags = " ".join(f'<span class="pill">{escape(tag)}</span>' for tag in destination.tags)
    return f"""
        <div class="dest-card">
          <div class="dest-title">{escape(destination.flag)} {escape(destination.name)}</div>
          <div class="dest-country">{escape(destination.country)}</div>
          <p>{escape(destination.why)}</p>
          <div>{tags}</div>
        </div>
        """


def render_destination(destination: Destination) -> None:
    st.markdown(destination_card_html(destination), unsafe_allow_html=True)


def run_trip(
    planner: TravelPlanner,
    preferences: TravelPreferences,
    destination: Destination,
) -> None:
    st.session_state.planning_log = {}
    progress_bar = st.progress(0)
    status_box = st.empty()
    completed = set()

    def progress(agent_id: str, status: str, result: AgentResult | None) -> None:
        if status in {"done", "error"}:
            completed.add(agent_id)
        st.session_state.planning_log[agent_id] = status
        progress_bar.progress(len(completed) / len(AGENT_SEQUENCE))
        lines = ["**Planning status**"]
        for current in AGENT_SEQUENCE:
            current_status = st.session_state.planning_log.get(current, "queued")
            label = AGENTS[current]["label"]
            lines.append(f"- **{label}**: {current_status}")
        status_box.markdown("\n".join(lines))

    with st.spinner(f"Running specialist agents for {destination.display_name}..."):
        st.session_state.trip = planner.plan_trip(preferences, destination, progress=progress)
    if st.session_state.trip.errors:
        st.warning("The plan finished with agent warnings. See the research tabs for details.")
    else:
        st.success("Trip plan ready.")


def trip_panel() -> None:
    trip = st.session_state.trip
    if not trip:
        return

    st.subheader("3. Complete Travel Plan")
    st.markdown(
        f"**{trip.destination.display_name}** from **{trip.preferences.departure_city}** | "
        f"{trip.preferences.nights} nights | {trip.preferences.travel_window}"
    )

    markdown = trip.to_markdown()
    st.download_button(
        "Download Markdown",
        data=markdown,
        file_name=f"voyager_ai_{trip.destination.name.lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )

    itinerary_tab, research_tab, raw_tab = st.tabs(["Itinerary", "Research", "Raw Markdown"])
    with itinerary_tab:
        parsed = parse_itinerary(trip.itinerary)
        if parsed.days:
            for day in parsed.days:
                with st.expander(f"Day {day.num} - {day.title}", expanded=day.num == 1):
                    st.markdown(day.body)
            for section in parsed.extras:
                with st.expander(section.title):
                    st.markdown(section.body)
        else:
            st.markdown(trip.itinerary)

    with research_tab:
        for _agent_id, result in trip.results.items():
            with st.expander(
                f"{result.label} ({result.status})",
                expanded=result.error is not None,
            ):
                if result.error:
                    st.error(result.error)
                st.markdown(result.content)
                if result.sources:
                    st.write("Sources considered")
                    for source in result.sources:
                        st.markdown(f"- [{source.title}]({source.url}) - {source.snippet}")

    with raw_tab:
        st.code(markdown, language="markdown")
        st.download_button(
            "Download JSON Summary",
            data=json.dumps(
                {
                    "destination": trip.destination.to_dict(),
                    "errors": trip.errors,
                    "models": {agent_id: result.model for agent_id, result in trip.results.items()},
                },
                indent=2,
            ),
            file_name="voyager_ai_summary.json",
            mime="application/json",
        )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 2.4rem 2rem;
            border-radius: 28px;
            background:
              radial-gradient(circle at top left, rgba(234,196,93,.28), transparent 35%),
              linear-gradient(135deg, #062a38, #087062 48%, #e9d8a6);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 50px rgba(0,0,0,.18);
        }
        .hero h1 {
            font-size: clamp(2.4rem, 6vw, 4.8rem);
            line-height: .95;
            margin: .1rem 0 .7rem;
            letter-spacing: -.05em;
        }
        .hero p {
            max-width: 760px;
            font-size: 1.05rem;
            opacity: .9;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: .16em;
            font-size: .78rem;
            opacity: .78;
            font-weight: 700;
        }
        .dest-card {
            min-height: 230px;
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: .65rem;
            background:
              linear-gradient(180deg, rgba(255,255,255,.96), rgba(244,241,230,.96)),
              radial-gradient(circle at top right, rgba(8,112,98,.16), transparent 40%);
            box-shadow: 0 8px 24px rgba(8, 42, 56, .08);
        }
        .dest-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: .1rem;
        }
        .dest-country {
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .72rem;
            color: #087062;
            font-weight: 700;
        }
        .pill {
            display: inline-block;
            padding: .12rem .45rem;
            border-radius: 999px;
            background: rgba(8,112,98,.10);
            color: #075c51;
            font-size: .72rem;
            font-weight: 700;
            margin: .1rem .18rem .1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
