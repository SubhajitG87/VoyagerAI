# VoyagerAI premium rebuild: design

Date: 14 November 2026
Status: draft, awaiting approval

## Intent

VoyagerAI works today as a Streamlit prototype. The aim is to turn it into a polished personal
and portfolio app: something Subhajit can use for his own trips and link to publicly, where the
look, the behaviour and the trustworthiness of the output all hold up to a close look.

It does not need to stay local-only. Hosted open-model inference is fine.

## Goals

- A custom, designed web UI that no longer reads as a Streamlit demo.
- Output that can be trusted: claims carry their sources, and anything the model produced from
  its own knowledge is labelled as such.
- A structured trip (days, time slots, places, costs) so the app can show a map, a budget table
  and proper exports.
- A plan run that feels alive: research agents work in parallel and stream as they go.
- Trips are saved and each has its own shareable URL.
- A public link that shows finished work instantly and cannot be used to burn the owner's
  inference quota.
- The existing CLI keeps working.

## Non-goals

- Booking, payments, or any affiliate flow.
- User accounts or multi-user auth.
- Live fare or hotel-rate APIs in the first version (see Constraints).
- A native mobile app. The web UI should be responsive, nothing more.
- Translations.

## Anti-goals

These would make the project a failure even if everything technically works:

- It still looks like a template.
- Visa rules or prices are presented as fact with no source and no warning.
- It depends on a paid proprietary model SDK.
- A stranger with the link can run up model calls on the owner's key.

## Constraints

- Inference stays on open models through open or free endpoints (Ollama locally, NVIDIA NIM
  hosted, anything OpenAI-compatible). No paid proprietary SDKs.
- NVIDIA's free hosted endpoint is rate limited (roughly 40 requests a minute, account-wide,
  not a published guarantee) and is offered for development and prototyping. The app must back
  off on HTTP 429 and must keep the provider swappable by config.
- Amadeus closed its free self-service API on 17 July 2026, so there is no free fare API to
  build on right now. Flight and hotel figures stay as sourced estimates with deep links to
  Google Flights, Skyscanner and Booking.com. A provider interface is left in place so a real
  API can be added later.
- Free container hosting has thinned out in 2026. Hugging Face Docker Spaces now need a paid
  plan, Fly.io has no free tier, and Render's free service sleeps after 15 minutes and has no
  persistent disk. The app must therefore run from one Docker image with no assumption of a
  persistent disk on the host.
- Python 3.10 or newer. Support for 3.9 is dropped (it is past end of life and the code
  already fails to import on it).

## Approaches considered

1. Keep Streamlit and push it further. Fastest and all Python, but layout and interaction
   stay limited and it will always be recognisable as Streamlit.
2. FastAPI with server-rendered pages (Jinja templates, HTMX, hand-written CSS). Full control
   of the look, live streaming over server-sent events, one Python app and one deployment, no
   JavaScript build chain. Very rich interactions such as drag-to-reorder cost more effort.
3. FastAPI backend with a React or Next.js frontend. Highest ceiling for interaction, at the
   price of a second codebase in another language and two things to deploy.

Chosen: approach 2. Subhajit picked it. It gives a designed product while staying in Python,
and the existing planner sits behind it with little change.

## Decisions made by Subhajit

- Target: polished personal and portfolio app, not a paid product. Hosted inference is allowed.
- UI stack: FastAPI with Jinja, HTMX and custom CSS.
- Public access: visitors browse pre-generated sample trips. Running a live plan needs a
  passcode that he holds.

## Calls made in the design (not yet confirmed line by line)

These were proposed and not objected to. They are mine, and open to correction.

- No accounts. A trip is identified by a short random id in its URL. "My trips" lists what is
  stored.
- Storage is SQLite behind a small repository layer. Sample trips live in the repo as JSON and
  are loaded at startup, so they survive on a host with no persistent disk. On such a host,
  trips created with the passcode last until the next restart. Locally they persist.
- NVIDIA NIM is the default provider when hosted, through the existing OpenAI-compatible
  client. Ollama stays the local default. Model names for NIM are picked from the live
  catalogue at build time and kept in config, since the catalogue changes often.
- The look follows the "Ember" direction: warm charcoal base, coral kept for key actions, soft
  teal as the second accent, serif headlines, soft corners, moderate glass, gentle motion.
  Home and gallery pages are dark, trip pages are light.
- The Streamlit app is removed once the new UI covers everything it does.
- Host choice is left until the last phase, because free tiers keep changing. The design only
  requires a single Docker image.

## How it works

### Engine (the existing `voyager_ai` package)

- Research agents (flight, visa, hotel, experience) run concurrently. The itinerary agent
  waits for all four, and receives their full output. The current 1,100-character cut is
  removed.
- Research reads pages, not just snippets. For each agent the search results are ranked
  (official government domains first for visa), the top few pages are fetched and reduced to
  text, and that text goes into the prompt.
- Each research agent returns structured output: a short summary plus a list of findings.
  Every finding has the claim, a source URL if one supports it, and a marker saying whether it
  is sourced or from model knowledge.
- The itinerary agent returns structured output too:
  - days, each with a number, a title and morning, afternoon and evening slots
  - each slot has a title, a description, an optional place name and an optional cost estimate
  - practical tips
  - a budget with line items as low to high ranges and a total
- Structured output is enforced by schema on each provider: Ollama's `format`, NIM's
  `guided_json`. The current `json_object` mode on the OpenAI-compatible client is replaced,
  because it allows any JSON at all.
- The planner emits events as it works (agent started, text chunk, agent finished, agent
  failed, plan finished). The CLI prints them; the web app streams them.
- LLM and search calls retry with backoff. Search failures are recorded on the result and
  shown, not swallowed.
- Markdown output for the CLI is rendered from the structured trip.

### Web app

Pages:

- Home: short pitch, the sample trip gallery, and the planning form (departure city,
  nationality, travel window or an optional start date, nights, style, direct flights).
- Destinations: eight suggestion cards, with "show me others".
- Planning view: four research cards filling in side by side as they stream, then the
  itinerary. A refresh reconnects to the same run and replays what has happened so far.
- Trip page: header with the trip facts, tabs for Itinerary, Flights, Visa, Stays and Budget.
  The itinerary tab shows the day list beside a map with numbered pins for the open day.
  Research tabs show findings with their source links and a visible label on anything
  unsourced. Export row at the bottom.
- My trips: saved trips, newest first.
- Passcode prompt: shown when someone without a valid session tries to start a live plan.

Behaviour:

- Live planning (discovery and plan runs) requires the passcode, set by an environment
  variable. A correct passcode sets a signed session cookie. Wrong attempts are rate limited.
  With no passcode configured (local use), live planning is open.
- A plan runs as a background task keyed by trip id, inside the single app process. Events
  are saved as they happen so the stream can be replayed.
- Places in the itinerary are geocoded with OpenStreetMap's Nominatim, at most one request a
  second, with results cached in SQLite. The map is Leaflet on OpenStreetMap tiles with
  attribution. A place that cannot be geocoded simply has no pin.
- HTMX, its SSE extension and Leaflet are vendored into the repo, not loaded from a CDN.
- Exports: PDF (server-rendered from a print template), calendar file (asks for a start date
  if the trip has none), Markdown, and the trip URL as the share link.
- "Refine this day": a text box on a day that reruns the itinerary agent for that day only,
  with the instruction and the rest of the trip as context.

## Phases

Each phase leaves the app working and gets its own implementation plan.

0. Repair. Fix the dead search import, the `--out` crash, the model-override loss, and the
   research truncation. Drop Python 3.9. Tidy the README, the prompt typo, the Makefile and
   requirements files, and formatting.
1. Engine. Structured research and itinerary, page fetching with sources, parallel agents,
   events, retries, NIM provider preset. CLI updated to match.
2. Web app. FastAPI, templates, styling, all pages above, SQLite, passcode gate, sample trips.
   A mockup of the home page and the trip page is shown for approval before the pages are
   built.
3. Finish. Map and geocoding, exports, refine-a-day, accessibility pass, Dockerfile, hosting,
   README with screenshots. Streamlit removed.

## Testing

- Unit tests keep using the demo provider and never call a real model or the network.
- Route tests cover each page, the passcode gate and the event stream.
- Schema tests check that demo and recorded outputs match the structured trip format.
- One opt-in smoke test runs a real plan when an API key is present. It is skipped in CI.
- CI runs ruff, the format check, mypy and pytest on Python 3.10 and 3.12.

## Left to the builder

- Module layout inside the package, and whether concurrency uses asyncio or threads.
- Exact colours, type choices and spacing within the Ember direction, subject to the mockup
  review in phase 2.
- Which NIM models to use, provided they support schema-constrained output.
- Which library reduces fetched pages to text, and which renders the PDF.
- Which three or four sample trips to generate.

## Open items

- Hosting provider, decided in phase 3 against whatever free options exist then.
- NVIDIA describes the free hosted endpoint as being for development and prototyping. A
  passcode-gated personal demo fits that loosely. If it stops fitting, the provider is a
  config change.
