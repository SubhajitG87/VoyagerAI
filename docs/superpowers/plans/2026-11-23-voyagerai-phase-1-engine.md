# VoyagerAI Phase 1 (Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each agent return structured, sourced output. Research agents read real pages, run in parallel and stream events. LLM and search calls retry, and NVIDIA NIM is added as a hosted provider. The CLI and the Streamlit app keep working on top of the new engine.

**Architecture:** The `voyager_ai` package keeps its shape, plus seven small new modules:

| Module | Holds |
| --- | --- |
| `retry` | retry helper |
| `schemas` | JSON schemas |
| `structured` | validated trip types |
| `fetch` | page reader |
| `events` | plan events |
| `render` | Markdown output |
| `links` | booking deep links |

Concurrency uses threads. A `ThreadPoolExecutor` runs the four research agents, and a semaphore caps simultaneous LLM calls. Events are always delivered on the thread that called `plan_trip`, so consumers need no locking. Phase 2's FastAPI app will call `plan_trip` from a background thread.

**Tech Stack:** Python 3.10+, httpx, ddgs, trafilatura (new: page text extraction), Streamlit, pytest, jsonschema (new, dev only), ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-14-voyagerai-premium-design.md` (Phases, item 1; "How it works → Engine")

## Global Constraints

- Python 3.10 or newer.
- "Inference stays on open models through open or free endpoints (Ollama locally, NVIDIA NIM hosted, anything OpenAI-compatible). No paid proprietary SDKs."
- "The app must back off on HTTP 429 and must keep the provider swappable by config."
- "Flight and hotel figures stay as sourced estimates with deep links to Google Flights, Skyscanner and Booking.com. A provider interface is left in place so a real API can be added later."
- "Unit tests keep using the demo provider and never call a real model or the network."
- "One opt-in smoke test runs a real plan when an API key is present. It is skipped in CI."
- "CI runs ruff, the format check, mypy and pytest on Python 3.10 and 3.12."
- The existing CLI keeps working: same subcommands and flags, same exit codes (0 ok, 1 error, 2 finished with agent errors).
- Ruff line length stays 100.
- New runtime dependency allowed: `trafilatura>=2.0`. New dev dependency: `jsonschema>=4.0`. Nothing else.
- Work happens on a branch named `phase-1-engine`, cut from `main`.

## Review Focus

1. **A model cites a URL it was never given.** The finding is shown as unverified and the URL is dropped. This is pinned in Task 4 and end to end in Task 8.
2. **Ollama on a laptop with several different per-agent models.** Running them in parallel would load several large models at once. On Ollama, LLM calls default to one at a time; only search and page fetching overlap. Pinned in Task 9.
3. **A stream fails after some chunks were already emitted.** There is no retry, because a retry would duplicate text. The agent gets an `agent_failed` event after its chunks, and the plan carries on. Pinned in Task 2 (client) and Task 9 (planner).
4. **Itinerary JSON is invalid or cut off** (long trip, small context). The planner makes one repair attempt. If that also fails, the itinerary agent fails with a readable error, and the research still appears in the Markdown output. Pinned in Task 8.
5. **A search result points at a private or loopback address, redirects to one, or serves a huge file or a PDF.** It is not fetched, and the plan carries on. Pinned in Task 6.

---

### Task 1: Retries with backoff, and Ollama thinking off

**Files:**
- Create: `src/voyager_ai/retry.py`
- Modify: `src/voyager_ai/llm.py` (import httpx at module level; status classification; `think`; strip reasoning)
- Modify: `src/voyager_ai/parser.py` (add `strip_reasoning`)
- Modify: `src/voyager_ai/config.py` (`llm_retries`, `ollama_think`)
- Test: `tests/test_retry.py` (new), `tests/test_llm_clients.py`, `tests/test_parser.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `RetryableError(message: str, retry_after: float | None = None)`, an `Exception` subclass with a `.retry_after` attribute.
  - `retry_call(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, sleep: Callable[[float], None] = time.sleep, random: Callable[[], float] = random.random) -> T`. It retries only on `RetryableError` and re-raises the last one when attempts run out.
    - Delay without a hint: `min(max_delay, base_delay * 2 ** (n - 1)) * (1 + 0.25 * random())`, where `n` is the attempt that just failed.
    - Delay with `retry_after`: `min(max_delay, retry_after)`, with no jitter.
  - `parse_retry_after(value: str | None, now: datetime | None = None) -> float | None`. It accepts seconds or an HTTP date and returns `None` for anything else.
  - `strip_reasoning(text: str) -> str` in `parser.py`. It removes `<think>…</think>` blocks. If an opening `<think>` has no closing tag, it returns `""`.
  - `OllamaClient(base_url, timeout_s=180, num_ctx=16384, think="false", retries=3, sleep=time.sleep)`
  - `OpenAICompatibleClient(base_url, api_key="", timeout_s=180, retries=3, sleep=time.sleep)`
  - `RETRYABLE_STATUSES = {429, 500, 502, 503, 504}` in `llm.py`. These statuses, plus `httpx.ConnectError` and `httpx.RemoteProtocolError`, become `RetryableError` inside the client. `httpx.ReadTimeout` is not retried, so a 180 s timeout never triples. After the last attempt, every failure surfaces as `LLMError`, keeping the existing messages.
  - `Settings.llm_retries: int = 3`, from `VOYAGER_LLM_RETRIES` with a minimum of 1.
  - `Settings.ollama_think: str = "false"`, from `VOYAGER_OLLAMA_THINK`. It accepts `false`, `true` or `default`; anything else means `false`.

- [ ] **Step 1: Create the branch.** Run: `git checkout main && git pull && git checkout -b phase-1-engine`

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_retry.py
def test_retries_then_succeeds():
    sleeps, calls = [], iter([RetryableError("x"), RetryableError("x"), "ok"])
    def fn():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item
    assert retry_call(fn, sleep=sleeps.append, random=lambda: 0) == "ok"
    assert sleeps == [1.0, 2.0]

def test_honours_retry_after_and_caps_it():
    for hint, expected in ((7, 7.0), (120, 30.0)):
        sleeps = []
        with pytest.raises(RetryableError):
            retry_call(_always(RetryableError("429", retry_after=hint)), attempts=2,
                       sleep=sleeps.append, random=lambda: 0)
        assert sleeps == [expected]

def test_other_errors_are_not_retried():
    fn = MagicMock(side_effect=ValueError("bad"))
    with pytest.raises(ValueError):
        retry_call(fn, sleep=lambda s: None)
    assert fn.call_count == 1

def test_gives_up_after_attempts():
    fn = MagicMock(side_effect=RetryableError("x"))
    with pytest.raises(RetryableError):
        retry_call(fn, attempts=3, sleep=lambda s: None)
    assert fn.call_count == 3

def test_parse_retry_after():
    now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:10 GMT", now=now) == 10.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
```

Add these to `tests/test_llm_clients.py`. Convert the existing tests there from patching `sys.modules["httpx"]` to patching `voyager_ai.llm.httpx.post`.

```python
def test_ollama_retries_429_using_retry_after(): ...
    # post returns a response with status_code 429 and headers {"Retry-After": "3"}, then a 200
    # with {"message": {"content": "ok"}}; client built with sleep=sleeps.append
    # assert result.text == "ok" and sleeps == [3.0] and post.call_count == 2

def test_ollama_does_not_retry_400(): ...
    # a single 400 response -> LLMError, post.call_count == 1

def test_ollama_sends_think_false_by_default():
    # payload["think"] is False; with think="true" -> True; with think="default" -> "think" not in payload

def test_openai_compatible_retries_503_then_succeeds(): ...
    # 503 then 200 -> text returned, post.call_count == 2

def test_reasoning_is_stripped_from_content():
    # Ollama content "<think>plan</think>\n{\"a\": 1}" -> result.text == '{"a": 1}'
```

```python
# tests/test_parser.py
def test_strip_reasoning():
    assert strip_reasoning("<think>a\nb</think>\n{\"x\": 1}") == '{"x": 1}'
    assert strip_reasoning("<think>never closed") == ""
    assert strip_reasoning("plain") == "plain"
```

```python
# tests/test_config.py
def test_retry_and_think_settings(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_RETRIES", "0")
    monkeypatch.setenv("VOYAGER_OLLAMA_THINK", "maybe")
    s = Settings.from_env()
    assert s.llm_retries == 1 and s.ollama_think == "false"
```

- [ ] **Step 3: Run them to make sure they fail.** Run: `python -m pytest tests/test_retry.py tests/test_llm_clients.py tests/test_parser.py tests/test_config.py -q`. Expected: import errors and failures.

- [ ] **Step 4: Implement** `retry.py`, `strip_reasoning`, the client changes and the settings as listed under Interfaces. Each client wraps its single HTTP attempt in a closure and passes it to `retry_call(attempts=self.retries, sleep=self.sleep)`. `build_llm_client` passes `settings.llm_retries` and `settings.ollama_think`. Apply `strip_reasoning` to the content before the empty-response check.

- [ ] **Step 5: Run the whole suite.** Run: `python -m pytest -q && ruff check src tests && ruff format --check src tests && mypy --config-file mypy.ini src`. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: retry LLM calls with backoff and Retry-After, turn off Ollama thinking"
```

---

### Task 2: Streaming chunks from every client

**Files:**
- Modify: `src/voyager_ai/llm.py`
- Test: `tests/test_llm_clients.py`

**Interfaces:**
- Consumes: Task 1's `retry_call` and `RetryableError`.
- Produces:
  - `ChunkSink = Callable[[str], None]` in `llm.py`.
  - `BaseLLMClient.complete(..., on_chunk: ChunkSink | None = None) -> LLMResponse`. When `on_chunk` is given, the client streams and calls it with each content piece. `LLMResponse.text` is still the full text, reasoning stripped.
  - Ollama streaming: `httpx.stream("POST", f"{base}/api/chat", json={..., "stream": True}, timeout=...)`. Each line of `iter_lines()` is JSON; it appends `message.content` and stops at `"done": true`.
  - OpenAI-compatible streaming: `"stream": True`. It reads `iter_lines()` and handles only lines starting with `data: `. `data: [DONE]` ends the stream, and the piece is `choices[0].delta.content`. Blank lines and lines starting with `:` are ignored.
  - Retry rule: a failure before the first chunk is retried like a non-streaming call. A failure after the first chunk raises `LLMError("<provider> stream interrupted after <n> characters: <error>")` without retrying.
  - `DemoLLMClient` with `on_chunk` sends its text in 3 pieces whose concatenation equals the text.

- [ ] **Step 1: Write the failing tests.** Mock `voyager_ai.llm.httpx.stream` with a context manager whose `__enter__` returns a response with `status_code = 200` and `iter_lines` set.

```python
def test_ollama_streams_chunks():
    # lines: '{"message":{"content":"Hel"},"done":false}', '{"message":{"content":"lo"},"done":true}'
    # assert chunks == ["Hel", "lo"] and result.text == "Hello"

def test_openai_streams_sse_chunks():
    # lines: ': keepalive', 'data: {"choices":[{"delta":{"content":"Hel"}}]}', '',
    #        'data: {"choices":[{"delta":{"content":"lo"}}]}', 'data: [DONE]'
    # assert chunks == ["Hel", "lo"] and result.text == "Hello"

def test_stream_interrupted_after_first_chunk_is_not_retried():
    # iter_lines yields one data line, then raises httpx.ReadError("reset")
    # assert LLMError raised, "interrupted after 3 characters" in str(err), stream.call_count == 1

def test_stream_503_before_any_chunk_is_retried():
    # first stream context returns status_code 503, second streams "ok"
    # assert result.text == "ok" and stream.call_count == 2

def test_demo_client_streams_three_pieces():
    chunks = []
    result = DemoLLMClient().complete(agent_id="flight", model="demo", system="", user="u",
                                      on_chunk=chunks.append)
    assert len(chunks) == 3 and "".join(chunks) == result.text
```

- [ ] **Step 2: Run them to make sure they fail.** Run: `python -m pytest tests/test_llm_clients.py -q`. Expected: FAIL, because `complete()` gets an unexpected keyword `on_chunk`.

- [ ] **Step 3: Implement** streaming as listed under Interfaces. Count emitted characters in the attempt closure, so the retry wrapper can tell "before the first chunk" from "after".

- [ ] **Step 4: Run the whole suite and checks** (same command as Task 1, Step 5). Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: stream LLM output chunks from Ollama, OpenAI-compatible and demo clients"
```

---

### Task 3: Schema modes and the NIM provider

**Files:**
- Modify: `src/voyager_ai/llm.py`, `src/voyager_ai/config.py`, `src/voyager_ai/cli.py` (`--provider` choices), `src/voyager_ai/app.py` (provider select box), `.env.example`
- Create: `scripts/probe_schema.py`
- Test: `tests/test_llm_clients.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `SchemaMode = Literal["json_schema", "nvext_guided", "vllm_guided", "json_object"]` in `llm.py`.
  - `apply_schema(payload: dict[str, Any], schema: dict[str, Any], mode: SchemaMode, name: str) -> dict[str, Any]`. It returns a new dict and doesn't mutate its input.
    - `json_schema` sets `response_format = {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}`.
    - `nvext_guided` sets `nvext = {"guided_json": schema}`.
    - `vllm_guided` sets a top-level `guided_json = schema`.
    - `json_object` sets `response_format = {"type": "json_object"}`.
  - `OpenAICompatibleClient(..., schema_mode: SchemaMode = "json_schema")`. The existing fallback to `json_object` on HTTP 400 or 422 now applies to every mode except `json_object`.
  - In `config.py`:
    - `NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"`
    - `NIM_MODELS: dict[str, str]`, one model per agent id.
    - `NIM_SCHEMA_MODE: str`
    - `NIM_MODELS` and `NIM_SCHEMA_MODE` are filled in at Step 6.
  - New settings:
    - `Settings.nvidia_api_key`, from `NVIDIA_API_KEY`.
    - `Settings.nvidia_base_url`, from `NVIDIA_BASE_URL` (default `NIM_BASE_URL`).
    - `Settings.schema_mode`, from `VOYAGER_SCHEMA_MODE`. The default `""` means the provider default: `json_schema` for openai-compatible, `NIM_SCHEMA_MODE` for nim.
  - `resolve_model_map(preset: str, env: Mapping[str, str] | None = None, provider: str = "ollama") -> dict[str, str]`. For `provider == "nim"`, it starts from `NIM_MODELS` whatever the preset, then applies the same per-agent overrides. `Settings.from_env` and `Settings.use_preset` pass the provider through.
  - `build_llm_client` with `provider == "nim"` returns an `OpenAICompatibleClient` on `nvidia_base_url` with `nvidia_api_key`. If the key is empty, it raises `LLMError("Set NVIDIA_API_KEY to use the nim provider.")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_schema_modes():
    base, schema = {"model": "m"}, {"type": "object"}
    assert apply_schema(base, schema, "json_schema", "d")["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "d", "schema": schema}}
    assert apply_schema(base, schema, "nvext_guided", "d")["nvext"] == {"guided_json": schema}
    assert apply_schema(base, schema, "vllm_guided", "d")["guided_json"] == schema
    assert apply_schema(base, schema, "json_object", "d")["response_format"] == {"type": "json_object"}
    assert base == {"model": "m"}

def test_nvext_mode_falls_back_to_json_object_on_400(): ...
    # client(schema_mode="nvext_guided"); 400 then 200; second payload has response_format json_object
    # and no "nvext" key

def test_build_nim_requires_key():
    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        build_llm_client(Settings(provider="nim"))

def test_build_nim_client():
    client = build_llm_client(Settings(provider="nim", nvidia_api_key="k"))
    assert client.base_url == NIM_BASE_URL and client.schema_mode == NIM_SCHEMA_MODE

def test_nim_model_map_keeps_overrides():
    env = {"VOYAGER_AGENT_MODEL_VISA": "custom/visa"}
    model_map = resolve_model_map("small", env, provider="nim")
    assert model_map["visa"] == "custom/visa"
    assert model_map["flight"] == NIM_MODELS["flight"]
```

- [ ] **Step 2: Run them to make sure they fail.** Run: `python -m pytest tests/test_llm_clients.py tests/test_config.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement** `apply_schema`, the NIM settings, the model map and the client builder. In Step 6 the probe will set `NIM_MODELS` to the first candidate of each role and `NIM_SCHEMA_MODE = "json_schema"`; until then, these placeholder values let the tests run.

- [ ] **Step 4: Write `scripts/probe_schema.py`.** Arguments:
  - `--base-url`, default `NIM_BASE_URL`.
  - `--model`, repeatable.
  - `--mode`, repeatable, default all four modes in the order `json_schema`, `nvext_guided`, `vllm_guided`, `json_object`.
  - The API key comes from `NVIDIA_API_KEY`, falling back to `OPENAI_COMPATIBLE_API_KEY`.

  For each model and mode, it sends one request through `OpenAICompatibleClient` with the fallback off. It uses system `"Reply with JSON only."`, user `"Name one city for a short trip and a number of days."`, and this schema:

  `{"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city", "days"], "additionalProperties": false}`.

  It prints one line per pair: `PASS`, or `FAIL <reason>`. `PASS` means HTTP 200 and the reply validates against the schema with `jsonschema`. Add a `fallback: bool = True` constructor flag to `OpenAICompatibleClient` for this.

- [ ] **Step 5: Run the probe against the live catalogue.** This needs the owner's `NVIDIA_API_KEY`. If the key isn't set in the environment, stop and ask the owner to run the command and paste the output.

  ```bash
  python scripts/probe_schema.py \
    --model nvidia/nemotron-3.5-lightning-30b-a3b --model z-ai/glm-5.3-flash \
    --model google/gemma-4-31b-it --model mistralai/mistral-large-4-1t-a52b \
    --model moonshotai/kimi-k3.1-pro --model nvidia/nemotron-3.5-super-vl-120b-a12b
  ```

  These candidates come from `GET https://integrate.api.nvidia.com/v1/models` on 23 November 2026.

- [ ] **Step 6: Record the choice in `config.py`.**
  - `NIM_SCHEMA_MODE` is the first mode, in probe order, that passes for every model chosen below.
  - Fast roles (`discovery`, `flight`, `hotel`, `experience`): the first passing model among `nemotron-3.5-lightning-30b-a3b`, `glm-5.3-flash`, `gemma-4-31b-it`.
  - Careful roles (`visa`, `itinerary`): the first passing model among `mistral-large-4-1t-a52b`, `kimi-k3.1-pro`, `nemotron-3.5-super-vl-120b-a12b`.
  - Put a comment above `NIM_MODELS` with the date and "picked with scripts/probe_schema.py".
  - If no mode passes for at least one fast and one careful model, stop and ask the owner.
  - Add `NVIDIA_API_KEY=` and a commented `VOYAGER_LLM_PROVIDER=nim` to `.env.example`. Add `nim` to the CLI `--provider` choices and to the Streamlit provider select box.

- [ ] **Step 7: Run the whole suite and checks.** Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: add NVIDIA NIM provider and selectable schema modes, with a probe script"
```

---

### Task 4: Structured research and itinerary types

**Files:**
- Create: `src/voyager_ai/schemas.py` (move `DESTINATION_JSON_SCHEMA` here from `prompts.py` and re-export it from `prompts.py` so the planner import still works)
- Create: `src/voyager_ai/structured.py`
- Create: `src/voyager_ai/demo.py`
- Create: `tests/fixtures/visa_report.json`, `tests/fixtures/itinerary_3_nights.json`
- Modify: `pyproject.toml`, `requirements-dev.txt` (`jsonschema>=4.0`)
- Test: `tests/test_structured.py` (new), `tests/test_schemas.py` (new)

**Interfaces:**
- Produces, in `schemas.py`:
  - `RESEARCH_SCHEMA`: an object with `summary` (string) and `findings` (array, 0 to 12 items). Each finding has `topic` (string), `claim` (string), `basis` (enum `sourced` or `model`) and `source_url` (`["string", "null"]`).
  - `ITINERARY_SCHEMA`: an object with:
    - `days` (array). Each day has `number` (integer), `title`, and `morning`, `afternoon`, `evening`. Each slot has `title`, `description`, `place` (`["string", "null"]`) and `cost` (`{"low": number, "high": number}` or null).
    - `tips` (array of strings).
    - `budget` (`{"items": [{"label": string, "low": number, "high": number}]}`). There's no total field: the total is computed.
  - Every object lists all its properties in `required` and sets `"additionalProperties": false`. No `oneOf`, `anyOf` or `$ref`, so that Ollama `format` and vLLM guided decoding both accept the schemas.
- Produces, in `structured.py`:
  - `Basis = Literal["sourced", "model"]`
  - `normalize_url(url: str) -> str`: lowercases the scheme and host, drops the fragment, and strips a trailing `/` unless the path is just `/`.
  - `Finding(topic: str, claim: str, basis: Basis, source_url: str | None = None)`
  - `ResearchReport(summary: str, findings: list[Finding])`, with `from_payload(cls, payload: Any, allowed_urls: Collection[str]) -> ResearchReport`.
    - A finding's final basis is `"sourced"` exactly when `normalize_url(source_url)` matches one of the normalized `allowed_urls`; it then keeps the allowed URL's original spelling. Otherwise the basis is `"model"` and `source_url` is `None`.
    - Findings without a claim are dropped.
    - A missing or blank summary raises `ValidationError("The research report has no summary.")`.
    - `sourced_count` is a property.
  - `CostRange(low: float, high: float)`, with `from_payload(cls, payload: Any) -> CostRange`. It swaps `low` and `high` if they're reversed. It raises `ValidationError` for negative or non-numeric values.
  - `Slot(title: str, description: str, place: str | None = None, cost: CostRange | None = None)`. An invalid cost becomes `None`.
  - `DayPlan(number: int, title: str, morning: Slot, afternoon: Slot, evening: Slot)`
  - `BudgetItem(label: str, cost: CostRange)` and `Budget(items: list[BudgetItem], currency: str = "USD")`. `Budget.total` is a property returning `CostRange` with the sums. Invalid items are skipped.
  - `Itinerary(days: list[DayPlan], tips: list[str], budget: Budget, warnings: list[str])`, with `from_payload(cls, payload: Any, expected_days: int) -> Itinerary`:
    - Days are sorted by `number`, then renumbered from 1.
    - More days than expected are truncated to `expected_days`.
    - Fewer days add the warning `"The itinerary covers {n} of {expected_days} days."`
    - A missing slot becomes `Slot("Free time", "Nothing planned.")` and adds the warning `"Day {number} had no {slot} plan."`
    - No days at all raises `ValidationError("The itinerary has no days.")`.
- Produces, in `demo.py` (wired into the demo client in Task 8):
  - `demo_research_payload(agent_id: str, destination: str) -> dict[str, Any]`, with 3 findings, all with basis `model`.
  - `demo_itinerary_payload(days: int) -> dict[str, Any]`, with exactly `days` days, 2 tips and 5 budget items.

- [ ] **Step 1: Write the fixtures.**
  - `visa_report.json`: `{"allowed_urls": [...2 official URLs...], "payload": {...}}`, a realistic Japan visa report with 4 findings, 2 of them sourced from the listed URLs.
  - `itinerary_3_nights.json`: a realistic 4-day Lisbon payload that matches `ITINERARY_SCHEMA`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_structured.py
ALLOWED = ["https://www.mofa.go.jp/j_info/visit/visa/"]

def _report(**finding):
    base = {"topic": "Entry", "claim": "Visa-free for 90 days.", "basis": "sourced",
            "source_url": None}
    return ResearchReport.from_payload({"summary": "S", "findings": [{**base, **finding}]}, ALLOWED)

def test_known_url_is_sourced_and_keeps_allowed_spelling():
    f = _report(source_url="HTTPS://www.MOFA.go.jp/j_info/visit/visa").findings[0]
    assert f.basis == "sourced" and f.source_url == ALLOWED[0]

def test_invented_url_is_downgraded():
    f = _report(source_url="https://made-up.example/visa").findings[0]
    assert f.basis == "model" and f.source_url is None

def test_model_claim_with_known_url_counts_as_sourced():
    assert _report(basis="model", source_url=ALLOWED[0]).findings[0].basis == "sourced"

def test_finding_without_claim_is_dropped():
    assert _report(claim="  ").findings == []

def test_report_requires_summary():
    with pytest.raises(ValidationError, match="no summary"):
        ResearchReport.from_payload({"summary": "", "findings": []}, [])

def test_itinerary_sorts_renumbers_and_truncates():
    it = Itinerary.from_payload(_itin(numbers=[2, 1, 3]), expected_days=2)
    assert [d.number for d in it.days] == [1, 2] and it.days[0].title == "Day titled 1"

def test_short_itinerary_warns():
    it = Itinerary.from_payload(_itin(numbers=[1, 2, 3]), expected_days=5)
    assert "The itinerary covers 3 of 5 days." in it.warnings

def test_missing_slot_becomes_free_time():
    it = Itinerary.from_payload(_itin(numbers=[1], drop="evening"), expected_days=1)
    assert it.days[0].evening == Slot("Free time", "Nothing planned.")
    assert "Day 1 had no evening plan." in it.warnings

def test_budget_total_is_computed():
    budget = Budget([BudgetItem("Flights", CostRange(800, 1200)),
                     BudgetItem("Hotels", CostRange(700, 1400))])
    assert budget.total == CostRange(1500, 2600)

def test_cost_range_swaps_and_rejects_negative():
    assert CostRange.from_payload({"low": 50, "high": 20}) == CostRange(20, 50)
    with pytest.raises(ValidationError):
        CostRange.from_payload({"low": -1, "high": 5})

def test_itinerary_without_days_is_rejected():
    with pytest.raises(ValidationError, match="no days"):
        Itinerary.from_payload({"days": [], "tips": [], "budget": {"items": []}}, 3)
```

`_itin(numbers, drop=None)` is a local helper. It builds a schema-shaped payload whose day `n` is titled `f"Day titled {n}"`, and it omits the slot named in `drop`.

```python
# tests/test_schemas.py
@pytest.mark.parametrize("schema", [DESTINATION_JSON_SCHEMA, RESEARCH_SCHEMA, ITINERARY_SCHEMA])
def test_schemas_are_valid(schema):
    Draft202012Validator.check_schema(schema)

def test_demo_payloads_match_schemas():
    for agent in ("flight", "visa", "hotel", "experience"):
        validate(demo_research_payload(agent, "Tokyo, Japan"), RESEARCH_SCHEMA)
    validate(demo_itinerary_payload(31), ITINERARY_SCHEMA)
    assert len(demo_itinerary_payload(31)["days"]) == 31

def test_fixtures_match_schemas_and_parse_cleanly():
    visa = _load("visa_report.json")
    validate(visa["payload"], RESEARCH_SCHEMA)
    assert ResearchReport.from_payload(visa["payload"], visa["allowed_urls"]).sourced_count == 2
    itin = _load("itinerary_3_nights.json")
    validate(itin, ITINERARY_SCHEMA)
    assert Itinerary.from_payload(itin, expected_days=4).warnings == []
```

- [ ] **Step 3: Run them to make sure they fail.** Run: `python -m pytest tests/test_structured.py tests/test_schemas.py -q`. Expected: import errors.

- [ ] **Step 4: Implement** `schemas.py`, `structured.py` and `demo.py` as listed under Interfaces. Reuse `clean_text` from `utils.py` for all strings.

- [ ] **Step 5: Run the whole suite and checks.** Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add research and itinerary schemas with validated structured types"
```

---

### Task 5: Search failures recorded, retried, and official sources first

**Files:**
- Modify: `src/voyager_ai/search.py`, `src/voyager_ai/models.py` (`AgentResult.warnings`), `src/voyager_ai/planner.py` (use `SearchOutcome`), `src/voyager_ai/config.py` (`search_retries`)
- Test: `tests/test_search_client.py`, `tests/test_planner_demo.py`

**Interfaces:**
- Consumes: Task 1's `retry_call` and `RetryableError`.
- Produces:
  - `SearchError(VoyagerError)`.
  - `SearchOutcome(results: list[SearchResult], errors: list[str])`.
  - `SearchClient.search(query, max_results) -> list[SearchResult]` now raises `SearchError` instead of returning `[]` on failure. A missing `ddgs` package raises `SearchError("The ddgs package is not installed; run pip install ddgs.")`.
  - `SearchClient.search_many(queries, max_results) -> SearchOutcome`. It catches `SearchError` per query and records `f"Search failed for {query!r}: {error}"`.
  - `DuckDuckGoSearchClient(retries: int = 2, sleep=time.sleep)`. Any exception from `DDGS` becomes `RetryableError` inside `retry_call`, and the final failure becomes `SearchError`.
  - `Settings.search_retries: int = 2`, from `VOYAGER_SEARCH_RETRIES` with a minimum of 1.
  - `is_official_domain(url: str) -> bool`. It's true when the host:
    - ends in `.gov` or `.mil`, or
    - matches `(^|\.)(gov|gouv|gob|govt|go|gc)\.[a-z]{2}$`, or
    - is, or ends with `.` plus, one of `canada.ca`, `europa.eu`, `admin.ch`, `government.nl`.
  - `rank_results(agent_id: str, results: list[SearchResult]) -> list[SearchResult]`. For `visa`, it's a stable sort with official domains first. For other agents, it returns the input order unchanged.
  - `AgentResult.warnings: list[str] = field(default_factory=list)`. The planner puts `outcome.errors` into it, and `TripPlan.to_markdown` prints each as `> Search warning: {w}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_search_many_records_failures_and_keeps_other_results():
    client = FakeSearch({"q1": [r("a")], "q2": SearchError("rate limited"), "q3": [r("b")]})
    outcome = client.search_many(["q1", "q2", "q3"])
    assert [x.url for x in outcome.results] == ["https://a", "https://b"]
    assert outcome.errors == ["Search failed for 'q2': rate limited"]

def test_ddg_retries_then_succeeds(monkeypatch):
    # stub ddgs module whose DDGS().text raises once, then returns one result
    # DuckDuckGoSearchClient(retries=2, sleep=sleeps.append) -> 1 result, len(sleeps) == 1

def test_missing_ddgs_is_an_error(monkeypatch):
    # _load_ddgs_module returns None -> SearchError with "pip install ddgs"

@pytest.mark.parametrize("url,expected", [
    ("https://www.gov.uk/foreign-travel-advice", True),
    ("https://www.mofa.go.jp/j_info/visit/visa/", True),
    ("https://travel.state.gov/content/travel.html", True),
    ("https://www.canada.ca/en/immigration", True),
    ("https://indianvisaonline.gov.in/", True),
    ("https://govisa.example.com/", False),
    ("https://visa-gov.com/", False),
    ("https://go.com/", False),
])
def test_is_official_domain(url, expected):
    assert is_official_domain(url) is expected

def test_visa_ranking_puts_official_first_and_is_stable():
    ranked = rank_results("visa", [r("blog1"), r_url("https://www.gov.uk/x"), r("blog2")])
    assert [x.url for x in ranked] == ["https://www.gov.uk/x", "https://blog1", "https://blog2"]
    assert rank_results("hotel", ranked[::-1]) == ranked[::-1]

def test_search_warnings_reach_the_trip_markdown():
    # planner with a search client whose every query raises SearchError("down")
    # assert trip.results["flight"].warnings and "> Search warning:" in trip.to_markdown()
```

`r(name)` builds `SearchResult(name, f"https://{name}", "")`. `FakeSearch` overrides `search` from a dict.

- [ ] **Step 2: Run them to make sure they fail.** Expected: FAIL.

- [ ] **Step 3: Implement** the changes listed under Interfaces. Update existing tests that expected search errors to return `[]`.

- [ ] **Step 4: Run the whole suite and checks.** Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: retry search, record search failures on the result, rank official visa sources first"
```

---

### Task 6: Read pages, not just snippets

**Files:**
- Create: `src/voyager_ai/fetch.py`
- Modify: `src/voyager_ai/search.py` (`format_search_context` takes pages)
- Modify: `pyproject.toml`, `requirements.txt` (`trafilatura>=2.0`)
- Test: `tests/test_fetch.py` (new), `tests/test_search_client.py`

**Interfaces:**
- Produces:
  - `Page(url: str, title: str, text: str)`.
  - `USER_AGENT = "VoyagerAI/0.2 (+https://github.com/SubhajitG87/VoyagerAI)"`.
  - `Resolver = Callable[[str], list[str]]`. `default_resolve(host)` uses `socket.getaddrinfo` and returns the unique IP strings.
  - `is_public_http_url(url: str, resolve: Resolver) -> bool`. It's false for non-http(s) schemes, a missing host, a resolve failure, or any resolved address that is private, loopback, link-local, multicast, reserved or unspecified (`ipaddress` properties).
  - `PageFetcher(timeout_s: float = 10.0, max_bytes: int = 2_000_000, max_chars: int = 4000, resolve: Resolver = default_resolve, client: httpx.Client | None = None, check_robots: bool = True)`, with `fetch(url: str, title: str = "") -> Page | None`:
    - Redirects are followed by hand, at most 3 hops, and every hop is checked with `is_public_http_url`.
    - The content type must start with `text/html` or `application/xhtml+xml`.
    - The fetch is skipped when `Content-Length` exceeds `max_bytes`, or when the streamed body passes `max_bytes`.
    - Text comes from `trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True)`, truncated with `utils.truncate` to `max_chars`. An empty result returns `None`.
    - Robots: it fetches `/robots.txt` once per host (cached on the instance behind a `threading.Lock`) and parses it with `urllib.robotparser`. If robots.txt can't be fetched or returns 4xx or 5xx, fetching is allowed.
    - Every failure is logged at `debug` level and returns `None`. Nothing raises.
  - `NoopFetcher`, with the same `fetch` signature, always returning `None`.
  - `format_search_context(results: list[SearchResult], pages: Sequence[Page] = (), limit: int = 12) -> str`:
    - Pages come first, as `Source [i]: {title}\nURL: {url}\n{text}`.
    - Then results whose URL isn't a page, as `Source [i]: {title}\nURL: {url}\nSnippet: {snippet}`, up to `limit` sources in total.
    - It ends with this exact line: `Cite a source by copying its URL exactly into source_url. If no source supports a claim, set basis to "model" and source_url to null.`
    - With no results and no pages, it keeps the current "No live web snippets were available..." text.

- [ ] **Step 1: Write the failing tests.** Use `httpx.Client(transport=httpx.MockTransport(handler))`, which opens no sockets, and `resolve=lambda host: ["93.184.216.34"]`.

```python
ARTICLE = "<html><head><title>T</title></head><body><article><p>Visa-free for 90 days.</p>" \
          + "<p>" + "Detail. " * 2000 + "</p></article></body></html>"

def test_extracts_and_truncates_text():
    page = _fetcher(html_handler(ARTICLE)).fetch("https://example.org/visa", "Visa")
    assert page and "Visa-free for 90 days." in page.text and len(page.text) <= 4000

@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1"])
def test_private_addresses_are_not_fetched(ip):
    handler = MagicMock()
    assert _fetcher(handler, resolve=lambda h: [ip]).fetch("https://intranet.example/") is None
    handler.assert_not_called()

def test_redirect_to_private_address_is_blocked():
    # first response 302 Location http://internal.example/, resolve maps internal.example -> 10.0.0.1
    # assert fetch(...) is None

def test_non_html_and_oversized_are_skipped():
    assert _fetcher(handler_with(content_type="application/pdf")).fetch("https://e.org/a") is None
    assert _fetcher(handler_with(content_length=3_000_000)).fetch("https://e.org/b") is None
    assert _fetcher(handler_with(body=b"x" * 2_000_001)).fetch("https://e.org/c") is None

def test_robots_disallow_is_respected_and_404_allows():
    # robots.txt "User-agent: *\nDisallow: /" -> None
    # robots.txt 404 -> Page returned

def test_context_lists_pages_then_snippets_with_cite_rule():
    ctx = format_search_context([r("a"), r("b")], pages=[Page("https://a", "A", "Body A")])
    assert ctx.index("Body A") < ctx.index("https://b")
    assert ctx.count("https://a") == 1
    assert ctx.rstrip().endswith('set basis to "model" and source_url to null.')
```

- [ ] **Step 2: Run them to make sure they fail.** Run: `python -m pytest tests/test_fetch.py tests/test_search_client.py -q`. Expected: import errors.

- [ ] **Step 3: Add `trafilatura>=2.0`, reinstall** (`pip install -e ".[dev]"`), **and implement** `fetch.py` and the new `format_search_context`.

- [ ] **Step 4: Run the whole suite and checks.** Expected: all pass, with mypy clean (`trafilatura` has no stubs, so `ignore_missing_imports` already covers it).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: read the top search result pages, with private-address, size and robots checks"
```

---

### Task 7: Structured prompts

**Files:**
- Modify: `src/voyager_ai/prompts.py` (add new builders; the old `agent_prompt` stays until Task 8 deletes it)
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: Task 4's `ResearchReport` and `Finding`.
- Produces:
  - `research_prompt(agent_id: str, preferences: TravelPreferences, destination: Destination, context: str) -> AgentPrompt`, for flight, visa, hotel and experience.
    - The system text names the role, keeps today's per-agent topic list, and ends with `Return only JSON that matches the schema. Put each fact in its own finding.`
    - The visa system text keeps `Prioritise official government sources.`
    - The experience system text contains `preferences.vibe_description` (the original `${vibe}` bug must not come back).
    - Temperatures and max tokens stay as today.
  - `render_report_for_prompt(report: ResearchReport | None, missing: str) -> str`. It returns `missing` when `report` is `None`. Otherwise it gives the summary, then one line per finding: `- [sourced] {claim} ({url})` or `- [model] {claim}`.
  - `itinerary_prompt(preferences: TravelPreferences, destination: Destination, reports: Mapping[str, ResearchReport | None], research_chars: int, max_tokens: int) -> AgentPrompt`.
    - Each rendered report is truncated to `research_chars`. The placeholders stay `No {agent} research was available.`
    - The long-trip brevity line (over 10 days) stays.
    - The budget instruction names these exact labels: `Return economy flights`, `Mid-range hotel ({nights} nights)`, `Food`, `Activities`, `Local transport`, all in USD.
  - `ITINERARY_TOKENS_PER_DAY = 320` (it was 260, but JSON costs more per day). `ITINERARY_FIXED_TOKENS`, `ITINERARY_MAX_TOKENS`, `itinerary_max_tokens`, `itinerary_tokens_for_context` and `research_chars_for_context` keep their signatures.

- [ ] **Step 1: Write the failing tests** (update the existing token-budget assertions)

```python
def test_month_long_trip_budget():
    assert itinerary_max_tokens(31) == 11120
    assert itinerary_tokens_for_context(31, 16384) == 11120
    assert itinerary_tokens_for_context(31, 8192) == 6214
    assert research_chars_for_context(16384, 11120) == 3564

def test_experience_prompt_uses_the_travel_style():
    prefs = TravelPreferences("Mumbai", vibe="food")
    assert prefs.vibe_description in research_prompt("experience", prefs, DEST, "").system

def test_visa_prompt_names_nationality_and_official_sources():
    p = research_prompt("visa", TravelPreferences("Mumbai", nationality="Indian"), DEST, "")
    assert "Indian" in p.user and "official government sources" in p.system

def test_report_rendering_marks_basis():
    report = ResearchReport("S", [Finding("Entry", "Visa-free.", "sourced", "https://g.gov"),
                                  Finding("Fees", "None.", "model")])
    text = render_report_for_prompt(report, "missing")
    assert "- [sourced] Visa-free. (https://g.gov)" in text and "- [model] None." in text

def test_itinerary_prompt_placeholders_truncation_and_budget_labels():
    prompt = itinerary_prompt(TravelPreferences("Mumbai", nights=7), DEST,
                              {"flight": ResearchReport("F" * 5000, [])}, research_chars=1200,
                              max_tokens=4200)
    assert "No visa research was available." in prompt.user
    assert "F" * 1197 + "..." in prompt.user and "F" * 1201 not in prompt.user
    assert "Mid-range hotel (7 nights)" in prompt.user and prompt.max_tokens == 4200
```

- [ ] **Step 2: Run them to make sure they fail.** Expected: FAIL.

- [ ] **Step 3: Implement** the new builders.

- [ ] **Step 4: Run the whole suite and checks.** Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add structured research and itinerary prompts"
```

---

### Task 8: Structured results through the planner, Markdown renderer and CLI

**Files:**
- Modify: `src/voyager_ai/models.py` (`AgentResult` and `TripPlan` fields; remove `TripPlan.to_markdown`)
- Modify: `src/voyager_ai/planner.py` (still sequential), `src/voyager_ai/llm.py` (demo client returns JSON via `demo.py`), `src/voyager_ai/prompts.py` (delete the old `agent_prompt` and its builders), `src/voyager_ai/config.py` (`fetch_pages`, `page_chars`), `src/voyager_ai/cli.py`, `src/voyager_ai/app.py` (minimal: show `render_markdown(trip)`)
- Create: `src/voyager_ai/render.py`, `src/voyager_ai/links.py`
- Test: `tests/test_planner_structured.py` (new), `tests/test_render.py` (new), `tests/test_planner_demo.py`, `tests/test_planner_context.py`, `tests/test_cli.py`, `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: Tasks 4 to 7.
- Produces:
  - `AgentResult(agent_id: str, model: str, status: str = "done", report: ResearchReport | None = None, itinerary: Itinerary | None = None, error: str | None = None, warnings: list[str] = [], sources: list[SearchResult] = [], raw_text: str = "")`. The `content` field is removed.
  - `TripPlan.itinerary -> Itinerary | None`. `TripPlan.errors` is unchanged.
  - New settings:
    - `Settings.fetch_pages: int = 3`, from `VOYAGER_FETCH_PAGES`, minimum 0; 0 turns fetching off.
    - `Settings.page_chars: int = 4000`, from `VOYAGER_PAGE_CHARS`, minimum 500.
  - `TravelPlanner(settings=None, llm_client=None, search_client=None, fetcher: PageFetcher | NoopFetcher | None = None)`. The default fetcher is `NoopFetcher` when the provider is demo, search is off, or `fetch_pages == 0`. Otherwise it's `PageFetcher(max_chars=settings.page_chars)`.
  - `TravelPlanner.research(agent_id: str, preferences, destination, on_chunk: ChunkSink | None = None) -> AgentResult`:
    1. Search, then `rank_results`.
    2. Fetch the first `fetch_pages` results.
    3. If there were results but no page could be read, add the warning `No pages could be read; this agent used search snippets only.`
    4. Build the prompt with `format_search_context(results, pages)`.
    5. Parse with `allowed_urls` = every result URL.
  - `TravelPlanner.build_itinerary(preferences, destination, research: Mapping[str, AgentResult], on_chunk: ChunkSink | None = None) -> AgentResult`. It uses `itinerary_prompt` with each result's `report` (or `None`), plus the existing `_itinerary_tokens` and `_research_chars`.
  - Private `_complete_structured(agent_id, prompt, schema, parse: Callable[[Any], T], on_chunk) -> tuple[T, str]`:
    - It calls the LLM with `json_schema=schema`, then `extract_json`, then `parse`.
    - On `ValidationError`, it calls once more with the user text plus `\n\nYour previous reply could not be used: {error}. Reply again with only JSON that matches the schema.`
    - A second `ValidationError` propagates.
  - `plan_trip(preferences, destination, progress=None) -> TripPlan` keeps its Phase 0 signature and order in this task. A failed agent gets `status="error"` and `error=str(exc)`.
  - `render.py`:
    - `UNSOURCED_LABEL = "unverified: model knowledge"`
    - `VISA_NOTE = "Check these rules on the official government site before you book."`
    - `render_markdown(plan: TripPlan) -> str`. Order:
      1. Trip facts.
      2. `## Itinerary`, with `### Day N - Title`, then `**Morning** - {title}: {description} ({place}) - USD {low}-{high}` lines, leaving out empty parts.
      3. `## Practical Tips`.
      4. `## Budget`, as a table `| Item | Low (USD) | High (USD) |` with a `| **Total** |` row.
      5. Itinerary warnings.
      6. `## Flight Scout` and the other research sections: summary, then findings as `- **{topic}:** {claim} ([source]({url}))` or `- **{topic}:** {claim} _({UNSOURCED_LABEL})_`. `VISA_NOTE` goes directly under the visa heading. Booking links go under Flight Scout and Hotel Finder. Then `### Sources Considered`, warnings as `> Warning: ...`, and errors as `> Agent error: ...`.
      7. Amounts are formatted as whole dollars with thousands separators.
  - `links.py`:
    - `deep_links(preferences: TravelPreferences, destination: Destination) -> dict[str, str]`, with keys `"Google Flights"`, `"Skyscanner"` and `"Booking.com"`:
      - `https://www.google.com/travel/flights?q=` plus the URL-encoded `Flights from {departure_city} to {destination.display_name}`.
      - `https://www.skyscanner.net/` (Skyscanner's deep links need airport codes the app doesn't collect yet).
      - `https://www.booking.com/searchresults.html?ss=` plus the URL-encoded `destination.display_name`.
    - `PriceProvider` protocol, with `quotes(preferences, destination) -> list[PriceQuote]`, where `PriceQuote(label: str, cost: CostRange, url: str)`.
    - `NoPrices`, which returns `[]`.
    - The renderer prints quotes under the booking links when there are any. This is the spec's seam for a future fare API.
  - CLI `plan` writes `render_markdown(trip)`. `DemoLLMClient` returns `json.dumps(demo_research_payload(...))` for research agents and `json.dumps(demo_itinerary_payload(days))` for the itinerary, parsing `days` from the prompt as it does today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_planner_structured.py
def test_demo_plan_is_structured_end_to_end():
    trip = _demo_planner().plan_trip(TravelPreferences("Mumbai", nights=30), DEST)
    assert not trip.errors and len(trip.itinerary.days) == 31
    assert trip.results["visa"].report.findings

def test_invented_source_is_downgraded_end_to_end():
    # search client returns one result https://www.gov.uk/x for the visa agent; scripted LLM
    # returns a visa report citing https://www.gov.uk/x and https://fake.example/y
    bases = [f.basis for f in trip.results["visa"].report.findings]
    assert bases == ["sourced", "model"]

def test_invalid_json_gets_one_repair_attempt():
    # scripted LLM: itinerary replies "not json", then a valid payload
    # assert trip.itinerary is not None and llm.calls["itinerary"] == 2
    # assert "could not be used" in llm.last_user["itinerary"]

def test_itinerary_failure_keeps_research_in_markdown():
    # scripted LLM: itinerary replies "not json" twice
    md = render_markdown(trip)
    assert trip.results["itinerary"].status == "error" and "## Visa Advisor" in md
    assert "> Agent error:" in md

def test_unreadable_pages_add_a_warning():
    # search returns 2 results, fetcher returns None for both
    assert ("No pages could be read; this agent used search snippets only."
            in trip.results["flight"].warnings)
```

```python
# tests/test_render.py
def test_markdown_labels_unsourced_findings_and_adds_visa_note(): ...
    # contains f"_({UNSOURCED_LABEL})_", "([source](https://www.gov.uk/x))", VISA_NOTE
def test_budget_table_has_computed_total(): ...
    # items 800-1200 and 700-1400 -> "| **Total** | 1,500 | 2,600 |"
def test_deep_links(): ...
    # deep_links(TravelPreferences("Mumbai"), Destination("Tokyo", "Japan", "x"))["Google Flights"]
    # == "https://www.google.com/travel/flights?q=Flights%20from%20Mumbai%20to%20Tokyo%2C%20Japan"
```

Update `tests/test_planner_context.py`, so `LongResearchLLM` returns a valid research payload whose summary is `"R" * 9000`, and keep its truncation assertions against the summary. Update `tests/test_cli.py` and `tests/test_app_smoke.py` for `render_markdown`.

- [ ] **Step 2: Run them to make sure they fail.** Expected: FAIL.

- [ ] **Step 3: Implement** the changes listed under Interfaces. Delete the Phase 0 `agent_prompt`, `_flight_prompt`, `_visa_prompt`, `_hotel_prompt`, `_experience_prompt` and `_itinerary_prompt`, and delete their now-dead tests.

- [ ] **Step 4: Run the whole suite and checks, then smoke the CLI**

Run: `python -m pytest -q && ruff check src tests && ruff format --check src tests && mypy --config-file mypy.ini src`

Run: `VOYAGER_LLM_PROVIDER=demo voyager-ai plan --from Mumbai --destination Tokyo --country Japan --nights 3 --out /tmp/claude-p1/tokyo.md && grep -c "### Day" /tmp/claude-p1/tokyo.md`

Expected: everything passes, the CLI prints `Wrote …`, and `grep` prints `4`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: carry structured, source-checked results through the planner, renderer and CLI"
```

---

### Task 9: Parallel research and plan events

**Files:**
- Create: `src/voyager_ai/events.py`
- Modify: `src/voyager_ai/planner.py`, `src/voyager_ai/config.py` (`llm_concurrency`), `src/voyager_ai/cli.py`, `src/voyager_ai/app.py` (`run_trip` uses events)
- Test: `tests/test_planner_events.py` (new), `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 2's `ChunkSink` and Task 8's planner methods.
- Produces:
  - `EventKind = Literal["agent_started", "agent_chunk", "agent_finished", "agent_failed", "plan_finished"]`.
  - `PlanEvent(kind: EventKind, agent_id: str | None = None, text: str = "", result: AgentResult | None = None, plan: TripPlan | None = None)`, a frozen dataclass. `text` holds the chunk for `agent_chunk` and the error for `agent_failed`. It has `to_dict(self) -> dict[str, Any]`, which is JSON-safe: kind, agent_id, text, and for results the status, model, error, warnings, finding count and sourced count. It doesn't hold the full plan; Phase 2 stores the plan separately.
  - `EventSink = Callable[[PlanEvent], None]`.
  - `TravelPlanner.plan_trip(preferences, destination, on_event: EventSink | None = None) -> TripPlan`. This replaces `progress`, and the CLI and app are updated:
    - The four research agents run in a `ThreadPoolExecutor(max_workers=4)`. The itinerary runs after all four finish.
    - Every LLM call is wrapped in a `threading.Semaphore(settings.effective_llm_concurrency)`.
    - Workers put events on a `queue.Queue`. The calling thread drains it and calls `on_event`, so `on_event` only ever runs on the caller's thread.
    - Per agent, events come in this order: `agent_started`, any number of `agent_chunk`, then exactly one of `agent_finished` or `agent_failed`.
    - `plan_finished` comes exactly once, last, with `plan` set, even when agents fail.
    - If the caller is interrupted (for example `KeyboardInterrupt`), the executor is shut down with `cancel_futures=True` and the exception re-raised.
  - `Settings.llm_concurrency: int = 0`, from `VOYAGER_LLM_CONCURRENCY`, minimum 0.
  - `Settings.effective_llm_concurrency` is a property: `llm_concurrency` if above 0, else `1` for `ollama` and `4` for any other provider.
  - CLI stderr lines, exact:
    - `[{agent}] started`
    - `[{agent}] done: {n} findings, {k} sourced`, or `[itinerary] done: {d} days`
    - `[{agent}] failed: {error}`
    - `plan finished`
    - Chunks are not printed.

- [ ] **Step 1: Write the failing tests**

```python
def test_event_order_and_thread():
    events, threads = [], set()
    def sink(e):
        events.append(e); threads.add(threading.get_ident())
    _demo_planner().plan_trip(PREFS, DEST, on_event=sink)
    assert threads == {threading.get_ident()}
    assert [e.kind for e in events].count("plan_finished") == 1 and events[-1].kind == "plan_finished"
    for agent in ("flight", "visa", "hotel", "experience", "itinerary"):
        kinds = [e.kind for e in events if e.agent_id == agent]
        assert kinds[0] == "agent_started" and kinds[-1] == "agent_finished"
        assert set(kinds[1:-1]) <= {"agent_chunk"}

def test_research_agents_run_concurrently_on_hosted_providers():
    # LLM whose research calls wait on threading.Barrier(4, timeout=2); provider "openai-compatible"
    # plan completes with no errors (a sequential planner would raise BrokenBarrierError)

def test_ollama_runs_one_llm_call_at_a_time():
    # LLM tracking max simultaneous calls with a lock and a short sleep; provider "ollama"
    assert llm.max_in_flight == 1

def test_failed_agent_emits_failed_after_its_chunks_and_plan_continues():
    # LLM: visa agent calls on_chunk("partial") then raises LLMError("stream interrupted")
    kinds = [e.kind for e in events if e.agent_id == "visa"]
    assert kinds == ["agent_started", "agent_chunk", "agent_failed"]
    assert "stream interrupted" in [e for e in events if e.kind == "agent_failed"][0].text
    assert trip.results["itinerary"].status == "done"

def test_events_serialise_to_json():
    _demo_planner().plan_trip(PREFS, DEST, on_event=lambda e: json.dumps(e.to_dict()))

def test_effective_concurrency_defaults():
    assert Settings(provider="ollama").effective_llm_concurrency == 1
    assert Settings(provider="nim").effective_llm_concurrency == 4
    assert Settings(provider="ollama", llm_concurrency=2).effective_llm_concurrency == 2
```

```python
# tests/test_cli.py
def test_plan_prints_agent_progress(capsys, monkeypatch):
    # demo provider, plan --nights 3
    err = capsys.readouterr().err
    assert "[visa] started" in err and "[itinerary] done: 4 days" in err
    assert err.rstrip().endswith("plan finished")
```

- [ ] **Step 2: Run them to make sure they fail.** Expected: FAIL.

- [ ] **Step 3: Implement** `events.py` and the parallel `plan_trip` as listed under Interfaces. In `app.py`, `run_trip` updates the status box and progress bar from events.

- [ ] **Step 4: Run the whole suite and checks.** Run the event tests five times with `python -m pytest tests/test_planner_events.py -q --count=5` if `pytest-repeat` is installed, or in a shell loop if not, to check for flakiness. Expected: all pass every time.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: run research agents in parallel and emit plan events"
```

---

### Task 10: Streamlit shows the structured trip

**Files:**
- Modify: `src/voyager_ai/app.py`, `src/voyager_ai/parser.py` (delete `parse_itinerary`, `ItineraryDay`, `ItinerarySection`, `ParsedItinerary`, `DAY_RE` and `HEADING_RE`)
- Test: `tests/test_app_smoke.py`, `tests/test_parser.py` (delete the itinerary-parsing tests)

**Interfaces:**
- Consumes: Task 4's types, and Task 8's `render_markdown`, `deep_links`, `UNSOURCED_LABEL` and `VISA_NOTE`.
- Produces, in the trip panel:
  - Tabs: `Itinerary`, `Research`, `Budget`, `Markdown`.
  - Itinerary: one expander per day, titled `Day {n} - {title}`, with three lines (Morning, Afternoon, Evening), then Practical Tips, then itinerary warnings via `st.warning`.
  - Research: one expander per agent, titled `{label} ({status})`. It shows the summary, then findings, with an unsourced finding suffixed `({UNSOURCED_LABEL})` in italics. `VISA_NOTE` appears via `st.info` in the visa expander. Booking links go in the flight and hotel expanders, and warnings and errors via `st.warning` and `st.error`.
  - Budget: `st.table` of `{"Item", "Low (USD)", "High (USD)"}` rows plus a Total row.
  - Markdown: the `render_markdown` output in `st.code`, plus the existing Download Markdown button.

- [ ] **Step 1: Write the failing test**

```python
def test_full_demo_flow_shows_structured_trip(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Mumbai")
    app.button[0].click().run()
    next(b for b in app.button if b.label == "Plan Lisbon").click().run()
    assert not app.exception
    assert any(e.label.startswith("Day 1 - ") for e in app.expander)
    assert any(UNSOURCED_LABEL in m.value for m in app.markdown)
    assert any(VISA_NOTE in i.value for i in app.info)
```

- [ ] **Step 2: Run it to make sure it fails.** Expected: FAIL.

- [ ] **Step 3: Implement** the trip panel as listed under Interfaces, and delete the dead parser code and its tests.

- [ ] **Step 4: Run the whole suite and checks, then look at the app.** Run: `VOYAGER_LLM_PROVIDER=demo streamlit run src/voyager_ai/app.py`, plan a demo trip, and check each tab renders. Expected: all checks pass and every tab shows content.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: show the structured itinerary, sourced research and budget in Streamlit"
```

---

### Task 11: Opt-in live smoke test, docs and version

**Files:**
- Create: `tests/test_live_smoke.py`
- Modify: `tests/conftest.py`, `pyproject.toml` (pytest marker, version `0.2.0`), `src/voyager_ai/__init__.py` (`__version__ = "0.2.0"`), `README.md`, `.env.example`

**Interfaces:**
- Produces:
  - A pytest marker `live`, registered in `[tool.pytest.ini_options] markers = ["live: calls a real model and the network; opt in with VOYAGER_LIVE_PROVIDER"]`.
  - In `conftest.py`, `_no_network` and `_isolated_settings` skip their blocking for tests marked `live`.
  - `test_live_plan`:
    - It is skipped unless `VOYAGER_LIVE_PROVIDER` is set (`nim`, `ollama` or `openai-compatible`).
    - It plans London to Lisbon, 3 nights, food.
    - It asserts that `trip.errors == {}`, that `len(trip.itinerary.days) == 4`, and that at least one research finding has basis `sourced`.

- [ ] **Step 1: Write the test and the conftest change.** Run: `python -m pytest -q`. Expected: everything passes and the live test is reported as skipped.

- [ ] **Step 2: Update the README and `.env.example`.**
  - Add a provider table (Ollama, OpenAI-compatible, NIM), each with the env vars it needs.
  - Document the new variables: `VOYAGER_LLM_RETRIES`, `VOYAGER_OLLAMA_THINK`, `VOYAGER_SCHEMA_MODE`, `VOYAGER_SEARCH_RETRIES`, `VOYAGER_FETCH_PAGES`, `VOYAGER_PAGE_CHARS`, `VOYAGER_LLM_CONCURRENCY`, `NVIDIA_API_KEY` and `NVIDIA_BASE_URL`.
  - Add a "How sources work" section: claims carry the URL they came from, and anything else is labelled `unverified: model knowledge`.
  - Add the live test command: `VOYAGER_LIVE_PROVIDER=nim NVIDIA_API_KEY=... python -m pytest -m live`.
  - Update the project layout for the new modules.

- [ ] **Step 3: Run the live test once by hand**, if a key or a local Ollama is available. Run: `VOYAGER_LIVE_PROVIDER=nim python -m pytest -m live -v`. Expected: PASS. If it can't be run, say so in the PR description.

- [ ] **Step 4: Run the whole suite and checks.** Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: add opt-in live smoke test; docs and version 0.2.0 for the new engine"
```
