# VoyagerAI
# VoyagerAI

[![CI](https://github.com/voyager-ai/voyager-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/voyager-ai/voyager-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://codecov.io/gh/voyager-ai/voyager-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/voyager-ai/voyager-ai)

VoyagerAI is a multi-agent travel research application. The original
`voyager_ai.jsx` prototype was a single React component that:

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
- Optional DuckDuckGo search snippets outside the LLM.
- Validation, JSON parsing, itinerary parsing, tests, CI, and packaging.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Run the app in deterministic demo mode first:

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

The app talks to Ollama through `/api/chat` and uses Ollama structured outputs for destination
JSON. See the official Ollama docs for the chat API and structured output format:

- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

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
```

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

The test suite uses deterministic demo responses and does not call real LLM or search services.

## Safety Notes

VoyagerAI is a research assistant, not a booking engine or immigration authority.

- Verify visa, passport, health, and entry rules on official government sites before booking.
- Treat flight, hotel, and activity prices as estimates unless verified on the booking provider.
- Live search snippets can fail, rate-limit, or return stale pages; the app surfaces uncertainty in
  prompts instead of hiding it.
- Local models can hallucinate. For high-stakes decisions, add a source-verification step before
  accepting the itinerary.

## Project Layout

```text
src/voyager_ai/
  app.py          Streamlit web UI
  cli.py          Command line interface
  config.py       Environment and model preset configuration
  llm.py          Ollama, OpenAI-compatible, and demo LLM clients
  planner.py      Multi-agent orchestration
  prompts.py      Specialist prompts and JSON schema
  search.py       Optional DuckDuckGo snippet provider
  models.py       Domain models and validation
  parser.py       JSON and itinerary markdown parsing
  constants.py    Cities, nationalities, travel styles, agent metadata
tests/            Unit tests
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

