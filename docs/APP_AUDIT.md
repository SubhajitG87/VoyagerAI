# VoyagerAI App Audit

## What The Original App Does

`voyager_ai.jsx` is a single-file React travel planner. It gives the user a polished form, suggests
destinations, and then sequentially calls five LLM-powered agents:

- Flight Scout researches airlines, fares, route duration, and booking tactics.
- Visa Advisor researches entry requirements for the traveller nationality.
- Hotel Finder recommends neighbourhoods and hotel tiers.
- Experience Curator finds tours and activities.
- Itinerary Architect combines the research into a day-by-day plan.

The UI includes destination cards, agent progress cards, markdown rendering, a final itinerary
view, and copy/restart actions.

## Important Issues Found

- The Anthropic API call is embedded directly in frontend code. That leaks API usage to the browser
  and has no server-side secret management.
- The original `fetch("https://api.anthropic.com/v1/messages")` call does not include an API key or
  Anthropic browser-safety headers, so it will fail in normal browser execution.
- The Experience Curator system prompt uses a plain string:
  `"You are a GetYourGuide experience curator. Focus on: ${vibe}. Use ## section headings."`
  This sends the literal `${vibe}` instead of the selected travel style.
- Anthropic web search tool handling is incorrect. The app maps `tool_use` blocks to `tool_result`
  using `b.output`, but Anthropic tool-use blocks do not work that way for client-side tool calls.
- The app trusts model JSON too optimistically during destination discovery. Invalid JSON, partial
  JSON, duplicate destinations, or missing fields are only weakly handled.
- Visa and price outputs are high-risk stale data unless backed by source links and a verification
  workflow.
- `source.unsplash.com` image URLs can fail or change behaviour; image failure is handled visually,
  but the app has no deterministic local fallback assets.
- The whole flow is sequential and long-running. A failed early agent can interrupt the entire plan.

## What The Python Port Changes

- Moves LLM calls into a Python provider layer.
- Defaults to local/open-model backends instead of Anthropic.
- Adds per-agent model mapping via `.env`.
- Adds optional external search snippets before LLM synthesis instead of relying on provider-native
  tool calls.
- Adds deterministic `demo` mode for tests and UI smoke checks.
- Adds input validation for trip length, travel window, style, and destination payloads.
- Deduplicates destination suggestions.
- Continues planning when one non-final agent fails and records agent-level warnings.
- Adds parsers and tests for JSON extraction and itinerary structure.
- Adds packaging, CI, `.env.example`, Makefile, README, license, and examples.

## Suggested Next Improvements

1. Add official-source verification for visas.
   Build a dedicated source ranking step that prefers government domains and flags unsupported
   claims before the Visa Advisor response is accepted.

2. Add real flight and hotel APIs.
   Search snippets are helpful, but booking data should come from structured providers when possible.
   Consider Amadeus, Duffel, Skyscanner partners, Booking.com affiliates, or SerpAPI-like wrappers
   depending on budget and terms.

3. Parallelise the independent agents.
   Flight, visa, hotel, and experience research can run concurrently. The itinerary agent should
   wait for their results.

4. Add persistent trip projects.
   Store generated plans in SQLite or Postgres with run metadata, model versions, sources, and
   user edits.

5. Add source confidence scoring.
   Each agent should output claim/source pairs so the UI can show which claims are verified,
   inferred, or unsupported.

6. Add export formats.
   Markdown is ready now. Good next exports are PDF, DOCX, calendar `.ics`, and a shareable HTML
   itinerary.

7. Add map-aware planning.
   Geocode venues and cluster each day by neighbourhood to avoid unrealistic transit jumps.

8. Add budget localisation.
   Let users choose home currency, comfort level, and daily spend style; convert estimates with a
   live exchange-rate provider.

9. Add user preference memory.
   Store preferred airlines, hotel chains, walking tolerance, dietary needs, accessibility needs,
   and pace.

10. Add structured tests for prompt outputs.
   Use golden demo fixtures and schema checks to prevent regressions in destination JSON, source
   formatting, and day-by-day itinerary structure.
