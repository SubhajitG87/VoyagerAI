# VoyagerAI

[![CI](https://github.com/SubhajitG87/VoyagerAI/actions/workflows/ci.yml/badge.svg)](https://github.com/SubhajitG87/VoyagerAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://codecov.io/gh/SubhajitG87/VoyagerAI/branch/main/graph/badge.svg)](https://codecov.io/gh/SubhajitG87/VoyagerAI)

VoyagerAI is a multi-agent travel research application. It began as a single-file React
prototype (not included in this repository) that:

1. Collects a traveller profile: departure city, nationality, travel window, trip length,
   direct-flight preference, and travel style.
2. Suggests eight destination ideas.
3. Runs five specialist agents: Flight Scout, Visa Advisor, Hotel Finder, Experience Curator,
   and Itinerary Architect.
4. Presents a polished itinerary with supporting research.

This Python version keeps the same product idea but makes it a GitHub-ready project with:

- A Streamlit web app.
- A CLI for repeatable runs and automation.
- A provider abstraction that replaces Anthropic calls with local/open-model backends.
- Per-agent model selection.
- DuckDuckGo search outside the LLM: the top results are read, and findings carry
  the URL they came from (see "How sources work").
- Schema-constrained structured output with retries, a structured itinerary with a budget, and
  research agents that run in parallel with progress events.
- Validation, JSON extraction, tests, CI, and packaging.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Run the app in deterministic, offline demo mode first (no model, search or network):

```bash
VOYAGER_LLM_PROVIDER=demo streamlit run src/voyager_ai/app.py
```

Or via the CLI:

```bash
VOYAGER_LLM_PROVIDER=demo voyager-ai discover --from Mumbai --nationality Indian --vibe food
```

## Local LLM Setup With Ollama

Install Ollama, start it, and pull whichever preset you want to use:

```bash
ollama serve
ollama pull qwen3:14b
ollama pull mistral-small3.1:24b
```

Then run:

```bash
streamlit run src/voyager_ai/app.py
```

The app talks to Ollama through `/api/chat` and uses Ollama structured outputs for every agent's
JSON (destinations, research reports and the itinerary). See the official Ollama docs for the chat API and structured output format:

- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

## Providers

Set `VOYAGER_LLM_PROVIDER` (or pass `--provider` on the CLI) to one of:

| Provider | `VOYAGER_LLM_PROVIDER` | Env vars it needs | Notes |
| --- | --- | --- | --- |
| Ollama (default) | `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434`), `VOYAGER_OLLAMA_NUM_CTX`, `VOYAGER_OLLAMA_THINK` | Local models; uses the model presets below. |
| OpenAI-compatible | `openai-compatible` | `OPENAI_COMPATIBLE_BASE_URL` (default `http://localhost:1234/v1`), `OPENAI_COMPATIBLE_API_KEY` (optional) | LM Studio, vLLM, llama.cpp server; uses the model presets below. |
| NVIDIA NIM | `nim` | `NVIDIA_API_KEY`, `NVIDIA_BASE_URL` (default `https://integrate.api.nvidia.com/v1`) | Hosted open models. Picks its own model per agent and ignores `VOYAGER_MODEL_PRESET`; per-agent overrides still apply. |
| Demo | `demo` | none | Deterministic offline responses. |

The NIM model ids in `NIM_MODELS` in `src/voyager_ai/config.py` are unprobed placeholders: they
have not been checked against the live NIM catalogue. Run `scripts/probe_schema.py` with
`NVIDIA_API_KEY` set to see which models accept structured output, and update `NIM_MODELS` (or
use the per-agent overrides) before relying on it.

## Model Strategy

The project ships three presets:

| Preset | Best for | Models |
| --- | --- | --- |
| `small` | Laptops / quick demo | `qwen3:8b`, `phi4-mini` |
| `balanced` | Good default quality | `qwen3:14b`, `mistral-small3.1:24b` |
| `quality` | Workstations | `qwen3:30b`, `mistral-small3.1:24b`, `llama3.3:70b` |

Why these defaults:

- Qwen3 is strong for agentic work, multilingual travel ideation, structured output, and broad
  reasoning. Ollama lists Qwen3 variants including `8b`, `14b`, and `30b`.
- Mistral Small 3.1 has a 128K context window and is a good fit for synthesising sourced travel
  research into compact summaries.
- Llama 3.3 70B is reserved for the `quality` preset because it is much larger, but useful when
  you want stronger reasoning for the visa and final itinerary agents.
- Phi-4-mini is the lightweight fallback for constrained hardware.

References:

- [Qwen3 on Ollama](https://ollama.com/library/qwen3)
- [Mistral Small 3.1 on Ollama](https://ollama.com/library/mistral-small3.1)
- [Llama 3.3 on Ollama](https://ollama.com/library/llama3.3)
- [Phi-4-mini on Ollama](https://ollama.com/library/phi4-mini)

## Configuration

Copy `.env.example` to `.env` and edit as needed.

```bash
VOYAGER_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
VOYAGER_MODEL_PRESET=balanced
VOYAGER_ENABLE_SEARCH=true
VOYAGER_OLLAMA_NUM_CTX=16384
```

Other variables (defaults in brackets; values below the minimum are raised to it):

| Variable | Meaning |
| --- | --- |
| `VOYAGER_LLM_RETRIES` [3; 5 on `nim`; min 1] | Transport attempts per model call, for HTTP 429, 5xx and connection errors. A 429 waits at least 15 s (longer if `Retry-After` says so, up to 60 s). This does not cover invalid structured output: a reply that does not match the schema gets exactly one repair attempt, which is not configurable. |
| `VOYAGER_OLLAMA_THINK` [`false`] | Ollama "think" flag: `false`, `true`, or `default` to leave it to the model. |
| `VOYAGER_SCHEMA_MODE` [provider default] | Structured-output mode: `json_schema`, `nvext_guided`, `vllm_guided` or `json_object`. |
| `VOYAGER_SEARCH_PROVIDER` [`duckduckgo`] | Web search provider: `duckduckgo`, or `none` to turn search off (as does `VOYAGER_ENABLE_SEARCH=false`). |
| `VOYAGER_SEARCH_RETRIES` [2, min 1] | Attempts per web search. |
| `VOYAGER_SEARCH_MAX_RESULTS` [4, min 1] | Search results per query. Each research agent runs 3 queries, and its prompt shows at most 12 sources. |
| `VOYAGER_FETCH_PAGES` [3, 0 turns it off] | Top search results each research agent reads in full. |
| `VOYAGER_PAGE_CHARS` [4000, min 500] | Characters of readable text kept per fetched page. |
| `VOYAGER_LLM_CONCURRENCY` [0 = automatic] | Model calls allowed at once: 1 on Ollama, 4 elsewhere. |
| `VOYAGER_REQUEST_TIMEOUT_S` [180] | Request timeout in seconds. |

`VOYAGER_OLLAMA_NUM_CTX` sets Ollama's context window in tokens (minimum 4096). The itinerary
agent reads all four research reports. With a smaller window the reports are trimmed to fit, so
lowering it saves memory at the cost of detail in the final itinerary.

The itinerary agent's reply budget grows with trip length (320 tokens a day plus 1200 for the tips
and budget, at least 4200). A reply cut off at that limit is retried once with a request for
shorter days and 50% more tokens, as far as the Ollama window allows. Trips longer than about two weeks need the default 16384-token window or more; with a
smaller one the reply budget is capped and the last days, tips or budget may be cut off.

On OpenAI-compatible servers the app cannot see the context size. If your server is loaded with
a small context, set `VOYAGER_RESEARCH_CONTEXT_CHARS` (for example `1100` for a 4096-token
window) to limit how much of each report is passed on.

Per-agent overrides:

```bash
VOYAGER_AGENT_MODEL_DISCOVERY=qwen3:14b
VOYAGER_AGENT_MODEL_FLIGHT=mistral-small3.1:24b
VOYAGER_AGENT_MODEL_VISA=mistral-small3.1:24b
VOYAGER_AGENT_MODEL_HOTEL=qwen3:14b
VOYAGER_AGENT_MODEL_EXPERIENCE=qwen3:14b
VOYAGER_AGENT_MODEL_ITINERARY=mistral-small3.1:24b
```

OpenAI-compatible local servers are supported too:

```bash
VOYAGER_LLM_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1
OPENAI_COMPATIBLE_API_KEY=
```

This works with local servers that expose `/v1/chat/completions`, such as LM Studio, vLLM, or
llama.cpp server.

## How sources work

Each research agent searches the web and reads the top few result pages
(`VOYAGER_FETCH_PAGES`). Its prompt shows at most 12 sources: the pages it read first, then
search snippets. The model is asked to mark each finding `sourced` with the URL it came from, or
`model` when no source supports it. A finding is shown as sourced only when the model marked it
`sourced` and its URL is one of the sources shown in that prompt. Anything else (a finding the
model marked `model`, one with no URL, or one citing a URL it was not shown) is labelled
`unverified: model knowledge` in the app and in the Markdown output, with no link. With search
disabled, or when no search results come back, every finding gets that label.

Prices, the budget and the itinerary carry a note that they are estimates to check with the
booking links. Model and search text is escaped before it is shown, so it cannot add its own
links.

The app checks only that the cited URL was one of the sources shown to the model. It does not check that the
page says what the claim says, and a sourced URL may be one of the results the agent only saw as
a snippet rather than read in full. Treat a sourced finding as a place to check, and an
unverified one as the model's own knowledge.

## CLI Usage

Discover destinations:

```bash
voyager-ai discover \
  --from Mumbai \
  --nationality Indian \
  --months 6 \
  --nights 7 \
  --vibe culture
```

Plan a specific trip:

```bash
voyager-ai plan \
  --from Mumbai \
  --nationality Indian \
  --destination Tokyo \
  --country Japan \
  --nights 7 \
  --vibe food \
  --out trip_plans/tokyo.md
```

List model presets:

```bash
voyager-ai models
```

## Development

```bash
pip install -e ".[dev]"
make test
make lint
```

The test suite uses deterministic demo responses and does not call real LLM or search services;
it blocks network access outright.

One opt-in smoke test runs a real plan (London to Lisbon, 3 nights, food). It is skipped unless
`VOYAGER_LIVE_PROVIDER` is set (`nim`, `ollama` or `openai-compatible`), so CI never runs it:

```bash
VOYAGER_LIVE_PROVIDER=nim NVIDIA_API_KEY=... python -m pytest -m live
```

It reads your real environment and `.env` and calls a real model and the network. It forces
search on (`VOYAGER_ENABLE_SEARCH`) and page fetching on (at least one page, raising
`VOYAGER_FETCH_PAGES` if it is lower), and asserts there were no agent errors, a four-day
itinerary, and at least one sourced finding. With `nim` it skips if `NVIDIA_API_KEY` is empty.

## Safety Notes

VoyagerAI is a research assistant, not a booking engine or immigration authority.

- Verify visa, passport, health, and entry rules on official government sites before booking.
- Treat flight, hotel, and activity prices as estimates unless verified on the booking provider.
- Live search and page fetching can fail, rate-limit, or return stale pages. Findings that do not
  cite a URL from the search results are labelled unverified; a cited URL only means the model
  pointed at a search result, not that the page was read or supports the claim.
- Local models can hallucinate. For high-stakes decisions, add a source-verification step before
  accepting the itinerary.

## Project Layout

```text
src/voyager_ai/
  app.py          Streamlit web UI
  cli.py          Command line interface
  config.py       Environment and model preset configuration
  llm.py          Ollama, OpenAI-compatible, NIM, and demo LLM clients
  planner.py      Multi-agent orchestration (parallel research, then itinerary)
  prompts.py      Specialist prompts
  schemas.py      JSON schemas for structured output
  structured.py   Typed research reports and itinerary, parsed and checked
  retry.py        Retry with backoff for model and search calls
  search.py       DuckDuckGo search provider
  fetch.py        Page fetching and readable-text extraction
  events.py       Progress events emitted while a plan runs
  render.py       Markdown rendering with source labels
  links.py        Booking deep links
  demo.py         Offline demo responses
  models.py       Domain models and validation
  parser.py       Extracts JSON from model output and strips reasoning
  constants.py    Cities, nationalities, travel styles, agent metadata
scripts/          probe_schema.py: checks which NIM models accept structured output
tests/            Unit tests (network blocked) and the opt-in live smoke test
docs/             Audit notes and improvement suggestions
examples/         Example request payloads
```

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository and create a feature branch.
2. **Set up** a development environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run the tests** before and after your changes:

   ```bash
   python -m pytest --cov=src --cov-report=term-missing
   ```

   The live smoke test is skipped by default (see Development).

4. **Lint & type-check** your code:

   ```bash
   ruff check src tests
   mypy --config-file mypy.ini src
   ```

5. **Write or update tests** for any new functionality. All new code must pass CI,
   maintain or increase coverage, and follow the existing code style.

6. Open a **pull request** with a clear description of what your change does
   and why it's needed.

### Code of Conduct

Please be respectful and constructive. Review the [LICENSE](LICENSE) and note
that this project is maintained by volunteers.

## License

VoyagerAI is released under the [MIT License](LICENSE).

