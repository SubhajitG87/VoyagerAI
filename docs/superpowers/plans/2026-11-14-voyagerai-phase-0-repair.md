# VoyagerAI Phase 0 (Repair) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing app do what it already claims: live search runs, model overrides stick, the itinerary agent sees the research, the CLI example works, and tooling agrees with itself.

**Architecture:** No structural change. Each task is a small fix inside the existing `voyager_ai` package with a test that pins it. The Streamlit app stays as the UI for now.

**Tech Stack:** Python 3.10+, httpx, ddgs, Streamlit, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-14-voyagerai-premium-design.md` (Phases, item 0)

## Global Constraints

- Python 3.10 or newer. Support for 3.9 is dropped.
- No new runtime dependencies in this phase.
- No paid proprietary model SDKs.
- Tests never call a real model or the network.
- The existing CLI keeps working.
- Ruff line length stays 100.
- Work happens on a branch named `phase-0-repair`, cut from `main`. Nothing is pushed.

## Review Focus

1. Search library present but the call fails (offline, rate limited): the plan run continues with no snippets and a warning is logged naming the error. Pinned in Task 2.
2. `--out` points at a directory or an unwritable location: exit code 1 with a one-line `VoyagerAI error:` message, no traceback. Pinned in Task 5.
3. A per-agent override variable is set but blank: it is ignored and the preset's model is used. Pinned in Task 3.
4. An unknown preset name: falls back to `balanced`. Pinned in Task 3.
5. Research longer than the context budget, or missing because an agent failed: the text is cut with `...` at the limit, and a missing section gets its placeholder line. Pinned in Task 4.

---

### Task 1: Drop Python 3.9 and make the tooling agree

**Files:**
- Modify: `pyproject.toml` (lines 10, 19, 48)
- Modify: `mypy.ini` (line 2)
- Modify: `.github/workflows/ci.yml` (matrix, new format step)
- Modify: `Makefile` (`test`, `lint`)
- Modify: `requirements-dev.txt`
- Modify: `README.md` (lines 1-8)
- Modify: any file `ruff format` rewrites under `src/` and `tests/`

**Interfaces:**
- Produces: `make test` runs pytest; `make lint` runs ruff check, ruff format check and mypy.

- [ ] **Step 1: Create the branch**

Run: `git checkout -b phase-0-repair`

- [ ] **Step 2: Edit the config files**

- `pyproject.toml`: `requires-python = ">=3.10"`; delete the `Programming Language :: Python :: 3.9` classifier; `target-version = "py310"`.
- `mypy.ini`: `python_version = 3.10`.
- `.github/workflows/ci.yml`: matrix `["3.10", "3.12"]`; add a step after Lint named `Format check` running `ruff format --check src tests`.
- `Makefile`: `test` runs `python3 -m pytest`; `lint` runs `ruff check src tests`, `ruff format --check src tests`, `mypy --config-file mypy.ini src` on three lines.
- `requirements-dev.txt`: add `pytest-cov>=5.0` and `mypy>=1.0,<2.0`.
- `README.md`: delete the duplicate `# VoyagerAI` on line 2; Python badge reads `3.10+` (`python-3.10%2B-blue`); both Codecov URLs use `SubhajitG87/VoyagerAI`.

- [ ] **Step 3: Apply lint fixes and formatting**

Run: `ruff check --fix src tests && ruff format src tests`

- [ ] **Step 4: Verify**

Run: `ruff check src tests && ruff format --check src tests && mypy --config-file mypy.ini src && python -m pytest -q`
Expected: `All checks passed!`, `18 files already formatted`, `Success: no issues found`, `42 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: require Python 3.10, align Makefile, CI and dev requirements"
```

---

### Task 2: Restore live web search

**Files:**
- Modify: `src/voyager_ai/search.py:41-63`
- Test: `tests/test_search_client.py`

**Interfaces:**
- Produces: `_load_ddgs_module() -> ModuleType | None` in `search.py`. Returns the `ddgs` module, else `duckduckgo_search`, else `None`.

- [ ] **Step 1: Write the failing tests** (in `TestDuckDuckGoSearchClient`; the second replaces the existing `test_missing_ddgs_package_returns_empty`)

```python
def test_imports_ddgs_when_not_preloaded(self) -> None:
    mock_module = MagicMock()
    mock_module.DDGS.return_value.__enter__.return_value.text.return_value = [
        {"title": "T", "href": "https://example.com/t", "body": "B"}
    ]
    with patch.dict("sys.modules"):
        sys.modules.pop("ddgs", None)
        sys.modules.pop("duckduckgo_search", None)
        with patch(
            "voyager_ai.search.importlib.import_module", return_value=mock_module
        ) as importer:
            results = DuckDuckGoSearchClient().search("q")
    importer.assert_called_once_with("ddgs")
    self.assertEqual([r.url for r in results], ["https://example.com/t"])

def test_missing_ddgs_package_returns_empty(self) -> None:
    with patch.dict("sys.modules", {"ddgs": None, "duckduckgo_search": None}):
        with self.assertLogs("voyager_ai.search", level="WARNING"):
            self.assertEqual(DuckDuckGoSearchClient().search("anything"), [])

def test_search_failure_logs_warning(self) -> None:
    module = self._patch_ddgs([])
    module.DDGS.return_value.__enter__.return_value.text.side_effect = RuntimeError(
        "rate limited"
    )
    with self.assertLogs("voyager_ai.search", level="WARNING") as logs:
        self.assertEqual(DuckDuckGoSearchClient().search("q"), [])
    self.assertIn("rate limited", logs.output[0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_search_client.py -q`
Expected: `test_imports_ddgs_when_not_preloaded` and `test_search_failure_logs_warning` FAIL.

- [ ] **Step 3: Implement**

Add `_load_ddgs_module()`: try `importlib.import_module` for `"ddgs"` then `"duckduckgo_search"`, catching `ImportError`. `DuckDuckGoSearchClient.search` calls it in place of the `sys.modules` loop. In the `except Exception` around the search call, bind the exception and log `logger.warning("Web search failed for %r: %s", query, exc)` before returning `[]`. Remove the unused `sys` import.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_search_client.py -q && python -c "from voyager_ai.search import _load_ddgs_module; print(_load_ddgs_module().__name__)"`
Expected: all pass, then `ddgs`.

- [ ] **Step 5: Commit**

```bash
git add src/voyager_ai/search.py tests/test_search_client.py
git commit -m "fix: import ddgs for real so live search runs"
```

---

### Task 3: Per-agent model overrides survive a preset change

**Files:**
- Modify: `src/voyager_ai/config.py:89-103`
- Modify: `src/voyager_ai/app.py:79-81`
- Create: `tests/test_config.py`
- Create: `tests/test_app_smoke.py`

**Interfaces:**
- Produces: `resolve_model_map(preset: str, env: Mapping[str, str] | None = None) -> dict[str, str]` in `config.py`. `env` defaults to `os.environ`. Unknown preset means `balanced`. For each agent id, `VOYAGER_AGENT_MODEL_<ID>` then `VOYAGER_MODEL_<ID>` overrides the preset when non-blank after stripping.
- Produces: `Settings.use_preset(self, preset: str) -> None`. Sets `model_preset` (falling back to `balanced` if unknown) and `model_map = resolve_model_map(preset)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
from voyager_ai.config import MODEL_PRESETS, Settings, resolve_model_map


def test_override_survives_preset_change():
    env = {"VOYAGER_AGENT_MODEL_VISA": "custom-visa"}
    model_map = resolve_model_map("small", env)
    assert model_map["visa"] == "custom-visa"
    assert model_map["flight"] == "phi4-mini"


def test_blank_override_is_ignored():
    assert resolve_model_map("small", {"VOYAGER_AGENT_MODEL_VISA": "  "})["visa"] == "phi4-mini"


def test_unknown_preset_falls_back_to_balanced():
    assert resolve_model_map("nope", {}) == MODEL_PRESETS["balanced"]


def test_use_preset_keeps_env_override(monkeypatch):
    monkeypatch.setenv("VOYAGER_AGENT_MODEL_VISA", "custom-visa")
    settings = Settings()
    settings.use_preset("quality")
    assert settings.model_preset == "quality"
    assert settings.model_map["visa"] == "custom-visa"
    assert settings.model_map["itinerary"] == "llama3.3:70b"
```

`tests/test_app_smoke.py`:

```python
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "voyager_ai" / "app.py"


def test_sidebar_shows_env_override(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    monkeypatch.setenv("VOYAGER_AGENT_MODEL_VISA", "custom-visa")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert "visa: custom-visa" in [block.value for block in app.sidebar.code]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_app_smoke.py -q`
Expected: FAIL (`cannot import name 'resolve_model_map'`; sidebar shows `visa: mistral-small3.1:24b`).

- [ ] **Step 3: Implement**

Add `resolve_model_map` and `Settings.use_preset` as specified. `Settings.from_env` builds its map through `resolve_model_map`. In `app.py`, replace the two lines that set `model_preset` and `model_map` with `settings.use_preset(preset)`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_config.py tests/test_app_smoke.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voyager_ai/config.py src/voyager_ai/app.py tests/test_config.py tests/test_app_smoke.py
git commit -m "fix: keep per-agent model overrides when the preset changes"
```

---

### Task 4: The itinerary agent sees the research

**Files:**
- Modify: `src/voyager_ai/prompts.py:191-195`
- Modify: `src/voyager_ai/config.py` (`Settings`, `from_env`)
- Modify: `src/voyager_ai/llm.py:43-45, 70, 188`
- Modify: `.env.example`, `README.md` (Configuration section)
- Create: `tests/test_prompts.py`
- Test: `tests/test_llm_clients.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `RESEARCH_CONTEXT_CHARS = 12000` in `prompts.py`, the per-agent cap on research passed to the itinerary agent (was 1100).
- Produces: `Settings.ollama_num_ctx: int = 16384`, read from `VOYAGER_OLLAMA_NUM_CTX`, floor 2048, bad values use the default.
- Produces: `OllamaClient(base_url: str = "http://localhost:11434", timeout_s: int = 180, num_ctx: int = 16384)`, which sends `options["num_ctx"]`. Raising the cap without this would let Ollama silently drop the start of the prompt.

- [ ] **Step 1: Write the failing tests**

`tests/test_prompts.py`:

```python
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.prompts import RESEARCH_CONTEXT_CHARS, agent_prompt


def _itinerary_user(prior):
    prefs = TravelPreferences("Mumbai")
    destination = Destination("Tokyo", "Japan", "Chosen by test.")
    return agent_prompt("itinerary", prefs, destination, "", prior).user


def test_itinerary_prompt_keeps_research_beyond_1100_chars():
    hotel = "A" * 3000 + " LUXURY-TIER-MARKER"
    assert "LUXURY-TIER-MARKER" in _itinerary_user({"hotel": hotel})


def test_itinerary_prompt_caps_oversized_research():
    user = _itinerary_user({"flight": "B" * (RESEARCH_CONTEXT_CHARS + 500)})
    assert "B" * RESEARCH_CONTEXT_CHARS not in user
    assert "B" * (RESEARCH_CONTEXT_CHARS - 3) + "..." in user


def test_itinerary_prompt_placeholder_for_missing_research():
    assert "No visa research was available." in _itinerary_user({})
```

Add to `TestOllamaClient` in `tests/test_llm_clients.py`:

```python
def test_num_ctx_sent_in_options(self) -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_s=10, num_ctx=8192)
    self._mock_response({"message": {"content": "ok"}})
    client.complete(agent_id="itinerary", model="m", system="", user="x")
    options = self.mock_httpx.post.call_args[1]["json"]["options"]
    self.assertEqual(options["num_ctx"], 8192)
```

Add to `tests/test_config.py`:

```python
def test_ollama_num_ctx_from_env(monkeypatch):
    monkeypatch.setenv("VOYAGER_OLLAMA_NUM_CTX", "8192")
    assert Settings.from_env().ollama_num_ctx == 8192


def test_ollama_num_ctx_bad_value_uses_default(monkeypatch):
    monkeypatch.setenv("VOYAGER_OLLAMA_NUM_CTX", "lots")
    assert Settings.from_env().ollama_num_ctx == 16384
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_prompts.py tests/test_llm_clients.py tests/test_config.py -q`
Expected: the new tests FAIL (import error for `RESEARCH_CONTEXT_CHARS`, unexpected keyword `num_ctx`, missing attribute `ollama_num_ctx`).

- [ ] **Step 3: Implement**

As the Interfaces block specifies. `build_llm_client` passes `settings.ollama_num_ctx`. Add `VOYAGER_OLLAMA_NUM_CTX=16384` to `.env.example` with a one-line comment, and one line to the README Configuration block.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/voyager_ai/prompts.py src/voyager_ai/config.py src/voyager_ai/llm.py .env.example README.md tests/
git commit -m "fix: pass full research to the itinerary agent and set Ollama context size"
```

---

### Task 5: `plan --out` works into a new folder and fails cleanly

**Files:**
- Modify: `src/voyager_ai/cli.py:142-145`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `cmd_plan` creates missing parent directories for `--out`. Any `OSError` while creating or writing is raised as `VoyagerError("could not write <path>: <reason>")`, which `main` already prints as `VoyagerAI error: ...` with exit code 1.

- [ ] **Step 1: Write the failing tests**

```python
import json

from voyager_ai.cli import main

PLAN = [
    "plan", "--from", "Mumbai", "--destination", "Tokyo", "--country", "Japan",
    "--provider", "demo", "--no-search",
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: the two `--out` tests FAIL with `FileNotFoundError` and `IsADirectoryError`.

- [ ] **Step 3: Implement** as the Interfaces block specifies.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voyager_ai/cli.py tests/test_cli.py
git commit -m "fix: create output folders for plan --out and report write errors cleanly"
```

---

### Task 6: Prompt typo and unescaped model text

**Files:**
- Modify: `src/voyager_ai/prompts.py:49`
- Modify: `src/voyager_ai/app.py:185-197`
- Test: `tests/test_prompts.py`, `tests/test_app_smoke.py`

**Interfaces:**
- Produces: `destination_card_html(destination: Destination) -> str` in `app.py`, returning the card markup with name, country, why, tags and flag all passed through `html.escape`. `render_destination` calls it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prompts.py`:

```python
from voyager_ai.prompts import discovery_prompt


def test_discovery_prompt_uses_product_name():
    system = discovery_prompt(TravelPreferences("Mumbai")).system
    assert "VoyagerAI's travel discovery expert" in system
```

Add to `tests/test_app_smoke.py`:

```python
from voyager_ai.app import destination_card_html
from voyager_ai.models import Destination


def test_destination_card_escapes_model_text():
    card = destination_card_html(
        Destination(name="<b>X</b>", country="Y", why="z", tags=["<i>"], flag="<script>")
    )
    assert "<script>" not in card
    assert "<b>X</b>" not in card
    assert "&lt;script&gt;" in card
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_prompts.py tests/test_app_smoke.py -q`
Expected: FAIL (`VoyageAI's` in the prompt; `destination_card_html` not defined).

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run the whole gate**

Run: `ruff check src tests && ruff format --check src tests && mypy --config-file mypy.ini src && python -m pytest -q`
Expected: all clean, all tests pass.

- [ ] **Step 5: Run the README example end to end**

Run: `VOYAGER_LLM_PROVIDER=demo voyager-ai plan --from Mumbai --nationality Indian --destination Tokyo --country Japan --nights 7 --vibe food --out trip_plans/tokyo.md`
Expected: `Wrote trip_plans/tokyo.md`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add src/voyager_ai/prompts.py src/voyager_ai/app.py tests/test_prompts.py tests/test_app_smoke.py
git commit -m "fix: correct product name in discovery prompt and escape destination card text"
```
