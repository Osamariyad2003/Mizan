"""Provider backends: selection, request shape, refusals, errors.

No network. The Gemini tests build requests with the *real* SDK types and hand
them to a fake transport, so a wrong field name or an invalid value fails here
rather than at the first real call.
"""

from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

from rag.config import Config
from rag.llm import (
    CREDENTIAL_HINT,
    GEMINI_DEFAULT_MODEL,
    ClaudeBackend,
    Completion,
    GeminiBackend,
    OpenRouterBackend,
    MissingCredentials,
    credentials_present,
    resolve_provider,
)

ALL_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "OPENROUTER_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI")


@pytest.fixture(autouse=True)
def isolate_provider_overrides(monkeypatch):
    """Developer .env choices must not change backend contract tests."""
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_LLM_PROVIDER", raising=False)


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)
    # credentials_present() also looks for an `ant auth login` profile on disk.
    monkeypatch.setattr("os.path.isdir", lambda _p: False)
    config = Config(index_dir=tmp_path)
    config.llm_provider = "auto"
    config.model = ""
    return config


# --------------------------------------------------------------------------
# provider selection
# --------------------------------------------------------------------------

def test_gemini_key_selects_gemini(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert resolve_provider(no_keys) == "gemini"


def test_google_api_key_also_selects_gemini(no_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert resolve_provider(no_keys) == "gemini"


def test_anthropic_key_selects_claude(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert resolve_provider(no_keys) == "claude"


def test_openrouter_key_selects_openrouter(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert resolve_provider(no_keys) == "openrouter"


def test_gemini_wins_when_both_keys_are_present(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert resolve_provider(no_keys) == "gemini"


def test_explicit_provider_overrides_available_keys(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    no_keys.llm_provider = "claude"
    assert resolve_provider(no_keys) == "claude"


def test_no_credentials_is_an_actionable_error(no_keys):
    with pytest.raises(MissingCredentials) as caught:
        resolve_provider(no_keys)
    assert "GEMINI_API_KEY" in str(caught.value) and "ANTHROPIC_API_KEY" in str(caught.value)


def test_unknown_provider_name_is_rejected(no_keys):
    no_keys.llm_provider = "llama"
    with pytest.raises(ValueError, match="Unknown RAG_LLM_PROVIDER"):
        resolve_provider(no_keys)


def test_vertex_ai_counts_as_gemini_credentials(no_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    assert credentials_present()["gemini"] is True


class _OpenRouterResponse:
    def __init__(self, payload=None, lines=None):
        self.payload = payload
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()

    def __iter__(self):
        return iter(self.lines)


def test_openrouter_complete_sends_bearer_key_and_parses_usage(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    captured = {}

    def fake_open(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _OpenRouterResponse({
            "model": "openai/gpt-4o-mini",
            "choices": [{"message": {"content": "answer [1]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        })

    monkeypatch.setattr("rag.llm.urlopen", fake_open)
    backend = OpenRouterBackend(no_keys)
    result = backend.complete(["system"], [{"role": "user", "content": "question"}])
    body = json.loads(captured["request"].data)
    assert captured["request"].headers["Authorization"] == "Bearer secret"
    assert body["messages"][0] == {"role": "system", "content": "system"}
    assert body["stream"] is False
    assert result.text == "answer [1]"
    assert result.usage == {"input_tokens": 11, "output_tokens": 4}


def test_openrouter_stream_skips_metadata_and_stops_at_done(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    lines = [
        b": OPENROUTER PROCESSING\n",
        b'data: {"choices":[{"delta":{"content":"Article "}}]}\n',
        b'data: {"choices":[{"delta":{"content":"61"}}]}\n',
        b"data: [DONE]\n",
        b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n',
    ]
    monkeypatch.setattr("rag.llm.urlopen", lambda *_a, **_k: _OpenRouterResponse(lines=lines))
    backend = OpenRouterBackend(no_keys)
    assert "".join(backend.stream([], [{"role": "user", "content": "q"}])) == "Article 61"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "rejected the API key"),
        (404, "does not offer model"),
        (429, "rate limit or credit limit"),
        (503, "server error"),
    ],
)
def test_openrouter_http_failures_are_actionable(no_keys, monkeypatch, status, message):
    from urllib.error import HTTPError

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    error = HTTPError("https://openrouter.ai", status, "failed", {}, None)
    monkeypatch.setattr("rag.llm.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(error))
    backend = OpenRouterBackend(no_keys)
    with pytest.raises(RuntimeError, match=message):
        backend.complete([], [{"role": "user", "content": "q"}])


@pytest.mark.xfail(strict=True, reason="OpenRouter SSE error payloads with HTTP 200 are silently ignored")
def test_openrouter_stream_surfaces_in_band_error_payload(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    lines = [
        b'data: {"error":{"code":429,"message":"quota exceeded"}}\n',
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr("rag.llm.urlopen", lambda *_a, **_k: _OpenRouterResponse(lines=lines))
    backend = OpenRouterBackend(no_keys)
    with pytest.raises(RuntimeError, match="quota exceeded"):
        list(backend.stream([], [{"role": "user", "content": "q"}]))


# --------------------------------------------------------------------------
# Gemini: request shape
# --------------------------------------------------------------------------

def _response(text="answer", finish="FinishReason.STOP", blocked=None, usage=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=SimpleNamespace(block_reason=blocked),
        usage_metadata=usage,
    )


class FakeModels:
    def __init__(self, response=None, chunks=None, error=None):
        self.last: dict = {}
        self._response = response or _response()
        self._chunks = chunks or []
        self._error = error

    def generate_content(self, **kwargs):
        self.last = kwargs
        if self._error:
            raise self._error
        return self._response

    def generate_content_stream(self, **kwargs):
        self.last = kwargs
        if self._error:
            raise self._error
        return iter(self._chunks)


@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """A GeminiBackend whose transport is fake but whose types are real."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def build(response=None, chunks=None, error=None, **config_kwargs):
        models = FakeModels(response, chunks, error)
        monkeypatch.setattr(
            "google.genai.Client", lambda *a, **k: SimpleNamespace(models=models)
        )
        # No retry sleeps unless a test is specifically about retrying.
        config_kwargs.setdefault("retry_attempts", 0)
        config_kwargs.setdefault("model", "")
        backend = GeminiBackend(Config(index_dir=tmp_path, **config_kwargs))
        return backend, models

    return build


def test_gemini_uses_the_current_default_model(gemini):
    backend, _ = gemini()
    assert backend.model == GEMINI_DEFAULT_MODEL == "gemini-3.8-flash"


def test_gemini_request_carries_system_prompt_and_limits(gemini):
    backend, models = gemini(max_tokens=1234)
    backend.complete(["rules here", "<context>excerpt</context>"], [{"role": "user", "content": "q"}])
    config = models.last["config"]
    assert "rules here" in config.system_instruction
    assert "<context>excerpt</context>" in config.system_instruction
    assert config.max_output_tokens == 1234
    assert models.last["model"] == GEMINI_DEFAULT_MODEL


def test_gemini_maps_assistant_turns_to_the_model_role(gemini):
    backend, models = gemini()
    backend.complete(
        ["sys"],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
    )
    contents = models.last["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "reply"


@pytest.mark.parametrize(
    "effort, level",
    [("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("xhigh", "HIGH"), ("max", "HIGH")],
)
def test_effort_maps_onto_gemini_thinking_levels(gemini, effort, level):
    backend, models = gemini(effort=effort)
    backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert models.last["config"].thinking_config.thinking_level.name == level


def test_gemini_returns_text_and_usage(gemini):
    usage = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20,
        cached_content_token_count=64, thoughts_token_count=7,
    )
    backend, _ = gemini(response=_response(text="  grounded answer [1] ", usage=usage))
    done = backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert done.text == "grounded answer [1]"
    assert done.usage == {
        "input_tokens": 100, "output_tokens": 20,
        "cached_tokens": 64, "thinking_tokens": 7,
    }
    assert done.refused is False and done.stop_reason == "end_turn"


def test_gemini_safety_finish_is_reported_as_a_refusal(gemini):
    backend, _ = gemini(response=_response(text=None, finish="FinishReason.SAFETY"))
    done = backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert done.refused and done.stop_reason == "refusal" and "SAFETY" in done.refusal_detail


def test_gemini_blocked_prompt_is_reported_as_a_refusal(gemini):
    backend, _ = gemini(response=_response(blocked="OTHER"))
    assert backend.complete(["sys"], [{"role": "user", "content": "q"}]).refused


def test_gemini_truncation_is_visible_in_the_stop_reason(gemini):
    backend, _ = gemini(response=_response(finish="FinishReason.MAX_TOKENS"))
    assert backend.complete(["sys"], [{"role": "user", "content": "q"}]).stop_reason == "max_tokens"


def test_gemini_streams_text_and_skips_empty_chunks(gemini):
    chunks = [SimpleNamespace(text="Arti"), SimpleNamespace(text=None), SimpleNamespace(text="cle 61")]
    backend, _ = gemini(chunks=chunks)
    assert "".join(backend.stream(["sys"], [{"role": "user", "content": "q"}])) == "Article 61"


# --------------------------------------------------------------------------
# Gemini: errors
# --------------------------------------------------------------------------

def _client_error(code: int):
    from google.genai import errors

    return errors.ClientError(code, {"error": {"message": "nope", "code": code}})


def test_unknown_model_error_names_a_current_model(gemini):
    backend, _ = gemini(error=_client_error(404))
    with pytest.raises(RuntimeError, match="gemini-3.8-flash"):
        backend.complete(["sys"], [{"role": "user", "content": "q"}])


def test_bad_key_error_points_at_the_env_var(gemini):
    backend, _ = gemini(error=_client_error(403))
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        backend.complete(["sys"], [{"role": "user", "content": "q"}])


def test_quota_error_is_explained(gemini):
    backend, _ = gemini(error=_client_error(429))
    with pytest.raises(RuntimeError, match="quota|rate limit"):
        backend.complete(["sys"], [{"role": "user", "content": "q"}])


def test_missing_gemini_key_is_caught_before_any_request(monkeypatch, tmp_path):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingCredentials, match="GEMINI_API_KEY"):
        GeminiBackend(Config(index_dir=tmp_path))


# --------------------------------------------------------------------------
# Claude: unchanged behaviour through the same interface
# --------------------------------------------------------------------------

class _FakeMessages:
    def __init__(self):
        self.last: dict = {}

    def create(self, **kwargs):
        self.last = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="claude answer [1]")],
            model="claude-opus-5",
            stop_reason="end_turn",
            stop_details=None,
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=64, cache_creation_input_tokens=0,
            ),
        )


@pytest.fixture
def claude(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    messages = _FakeMessages()
    monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: SimpleNamespace(messages=messages))
    return ClaudeBackend(Config(index_dir=tmp_path, stream=False)), messages


def test_claude_still_gets_adaptive_thinking_and_effort(claude):
    backend, messages = claude
    backend.complete(["rules", "<context/>"], [{"role": "user", "content": "q"}])
    assert messages.last["thinking"] == {"type": "adaptive"}
    assert messages.last["output_config"] == {"effort": "medium"}
    assert messages.last["model"] == "claude-opus-5"


def test_claude_puts_a_cache_breakpoint_after_the_context(claude):
    backend, messages = claude
    backend.complete(["rules", "<context/>"], [{"role": "user", "content": "q"}])
    system = messages.last["system"]
    assert [part["text"] for part in system] == ["rules", "<context/>"]
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral"}


def test_claude_usage_uses_the_neutral_key_names(claude):
    backend, _ = claude
    done = backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert done.usage["cached_tokens"] == 64
    assert done.usage["input_tokens"] == 10 and done.usage["output_tokens"] == 5


# --------------------------------------------------------------------------
# generation is provider-agnostic
# --------------------------------------------------------------------------

def test_answers_and_citations_work_on_any_backend(monkeypatch, tmp_path):
    import rag.generate as generate_mod
    from test_rag import _hit

    class Stub:
        name, model = "stub", "stub-1"

        def complete(self, system_parts, messages, max_tokens=None, effort=None):
            assert "<context>" in "\n".join(system_parts)  # context reached the model
            return Completion(text="Yes [2], see also [1].", model=self.model,
                              usage={"input_tokens": 1}, stop_reason="end_turn")

    monkeypatch.setattr(generate_mod, "get_backend", lambda _cfg: Stub())
    answer = generate_mod.generate_answer(
        "q", [_hit("alpha"), _hit("beta")], Config(index_dir=tmp_path)
    )
    assert answer.cited_indices == [2, 1]
    assert answer.model == "stub-1" and answer.grounded


def test_a_refusal_from_either_provider_becomes_a_plain_message(monkeypatch, tmp_path):
    import rag.generate as generate_mod
    from test_rag import _hit

    class Refuser:
        name, model = "stub", "stub-1"

        def complete(self, *a, **k):
            return Completion(text="", model="stub-1", refused=True, refusal_detail="SAFETY")

    monkeypatch.setattr(generate_mod, "get_backend", lambda _cfg: Refuser())
    answer = generate_mod.generate_answer("q", [_hit("alpha")], Config(index_dir=tmp_path))
    assert answer.stop_reason == "refusal" and "SAFETY" in answer.text


# --------------------------------------------------------------------------
# .env file loading
# --------------------------------------------------------------------------

def test_env_file_sets_a_key(tmp_path, monkeypatch):
    from rag.config import load_env_file

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=abc123\n", encoding="utf-8")
    assert load_env_file(path) == {"GEMINI_API_KEY": "abc123"}
    assert credentials_present()["gemini"] is True


def test_env_file_handles_comments_quotes_and_export(tmp_path, monkeypatch):
    from rag.config import load_env_file

    for var in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        'A_KEY="quoted value"\n'
        "export B_KEY=exported\n"
        "  C_KEY = spaced \n"
        "not a setting line\n",
        encoding="utf-8",
    )
    loaded = load_env_file(path)
    assert loaded == {"A_KEY": "quoted value", "B_KEY": "exported", "C_KEY": "spaced"}


def test_a_real_environment_variable_wins_over_the_file(tmp_path, monkeypatch):
    from rag.config import load_env_file

    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    load_env_file(tmp_path / ".env")
    import os

    assert os.environ["GEMINI_API_KEY"] == "from-environment"


def test_an_empty_value_is_not_treated_as_a_key(tmp_path, monkeypatch):
    """An unfilled template must not look like a configured key."""
    from rag.config import load_env_file

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    assert load_env_file(tmp_path / ".env") == {}
    assert credentials_present()["gemini"] is False


def test_a_missing_env_file_is_not_an_error(tmp_path):
    from rag.config import load_env_file

    assert load_env_file(tmp_path / "nothing-here") == {}


def test_the_shipped_template_lists_both_providers():
    from pathlib import Path

    template = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in template and "ANTHROPIC_API_KEY" in template


# --------------------------------------------------------------------------
# transient failures
# --------------------------------------------------------------------------

class FlakyModels(FakeModels):
    """Fails `failures` times with `error`, then succeeds."""

    def __init__(self, error, failures: int, response=None):
        super().__init__(response=response)
        self._flaky_error = error
        self.remaining = failures
        self.attempts = 0

    def generate_content(self, **kwargs):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self._flaky_error
        return self._response


def _server_error(code: int = 503):
    from google.genai import errors

    return errors.ServerError(code, {"error": {"message": "overloaded", "code": code}})


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
    return slept


def test_a_503_is_retried_and_then_succeeds(monkeypatch, tmp_path, no_sleep):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    models = FlakyModels(_server_error(503), failures=1, response=_response(text="recovered"))
    monkeypatch.setattr("google.genai.Client", lambda *a, **k: SimpleNamespace(models=models))
    backend = GeminiBackend(Config(index_dir=tmp_path, retry_attempts=2, retry_initial_seconds=1.5))

    assert backend.complete(["sys"], [{"role": "user", "content": "q"}]).text == "recovered"
    assert models.attempts == 2 and no_sleep == [1.5]


def test_retries_give_up_and_report_the_error(monkeypatch, tmp_path, no_sleep):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    models = FlakyModels(_server_error(503), failures=99)
    monkeypatch.setattr("google.genai.Client", lambda *a, **k: SimpleNamespace(models=models))
    backend = GeminiBackend(Config(index_dir=tmp_path, retry_attempts=2, retry_initial_seconds=1))

    with pytest.raises(RuntimeError, match="503"):
        backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert models.attempts == 3  # first try plus two retries
    assert no_sleep == [1, 2]  # doubling backoff


def test_a_bad_request_is_not_retried(monkeypatch, tmp_path, no_sleep):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    models = FlakyModels(_client_error(400), failures=99)
    monkeypatch.setattr("google.genai.Client", lambda *a, **k: SimpleNamespace(models=models))
    backend = GeminiBackend(Config(index_dir=tmp_path, retry_attempts=2))

    with pytest.raises(RuntimeError):
        backend.complete(["sys"], [{"role": "user", "content": "q"}])
    assert models.attempts == 1 and no_sleep == []


def test_a_stream_that_breaks_after_text_says_so(gemini):
    """Mid-stream failures must not be retried: the text is already out."""

    def exploding():
        yield SimpleNamespace(text="the answer begins")
        raise _server_error(503)

    backend, models = gemini()
    models.generate_content_stream = lambda **kw: exploding()
    out = []
    with pytest.raises(RuntimeError, match="cut short"):
        for piece in backend.stream(["sys"], [{"role": "user", "content": "q"}]):
            out.append(piece)
    assert out == ["the answer begins"]
