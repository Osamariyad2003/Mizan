"""Answer-generation backends: Claude, Gemini, or OpenRouter.

Only the *generation* step talks to a hosted model. Loading, chunking,
embedding, keyword search and reranking all run locally, so switching provider
changes nothing about the index and needs no re-ingest.

Both backends expose the same two calls -- :meth:`complete` and
:meth:`stream` -- over a provider-neutral request: a list of system parts and a
list of ``{"role": ..., "content": ...}`` messages. Everything provider-shaped
(Anthropic cache breakpoints, Gemini's ``model`` role and thinking levels) stays
inside the backend, so the prompt, citation parsing and the rest of the
pipeline never branch on which provider is in use.

Provider is chosen by ``RAG_LLM_PROVIDER``, or automatically from whichever
credentials are present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config

CLAUDE_DEFAULT_MODEL = "claude-opus-5"
# Gemini 3.x; the 2.5 models are restricted to accounts that already used them.
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"

GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
ANTHROPIC_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
OPENROUTER_KEY_VARS = ("OPENROUTER_API_KEY",)

CREDENTIAL_HINT = (
    "No credentials for answer generation found. Set GEMINI_API_KEY (free key from "
    "https://aistudio.google.com/apikey), ANTHROPIC_API_KEY, or OPENROUTER_API_KEY, "
    "then retry. "
    "Retrieval (`rag search`) needs no key."
)


class MissingCredentials(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    refused: bool = False
    refusal_detail: str = ""


class Backend(Protocol):
    name: str
    model: str

    def complete(
        self,
        system_parts: Sequence[str],
        messages: Sequence[dict[str, Any]],
        max_tokens: int | None = ...,
        effort: str | None = ...,
    ) -> Completion: ...

    def stream(
        self,
        system_parts: Sequence[str],
        messages: Sequence[dict[str, Any]],
        max_tokens: int | None = ...,
        effort: str | None = ...,
    ) -> Iterator[str]: ...

    def describe_error(self, exc: Exception) -> str: ...


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------

class ClaudeBackend:
    name = "claude"

    def __init__(self, config: Config):
        self.config = config
        self.model = config.model or CLAUDE_DEFAULT_MODEL
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingCredentials("Claude answers need: pip install anthropic") from exc
        self._anthropic = anthropic
        # The SDK resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then an
        # `ant auth login` profile, so an unset API key is not proof of no
        # credentials. Let it decide, and translate its failure.
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:
            raise MissingCredentials(CREDENTIAL_HINT) from exc

    def _params(self, system_parts, messages, max_tokens, effort) -> dict[str, Any]:
        system: list[dict[str, Any]] = [{"type": "text", "text": part} for part in system_parts if part]
        if system and self.config.cache_context:
            # Cache breakpoint after the retrieved context, so follow-up turns
            # over the same passages reuse it.
            system[-1]["cache_control"] = {"type": "ephemeral"}
        return {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort or self.config.effort},
            "system": system,
            "messages": list(messages),
        }

    def complete(self, system_parts, messages, max_tokens=None, effort=None) -> Completion:
        params = self._params(system_parts, messages, max_tokens, effort)
        try:
            if self.config.stream:
                # Streaming keeps a long answer inside the request timeout.
                with self._client.messages.stream(**params) as stream:
                    message = stream.get_final_message()
            else:
                message = self._client.messages.create(**params)
        except Exception as exc:
            raise RuntimeError(self.describe_error(exc)) from exc

        refused = message.stop_reason == "refusal"
        details = getattr(message, "stop_details", None)
        return Completion(
            text="".join(
                b.text for b in message.content if getattr(b, "type", None) == "text"
            ).strip(),
            model=message.model,
            usage=_claude_usage(message.usage),
            stop_reason=message.stop_reason,
            refused=refused,
            refusal_detail=str(getattr(details, "category", "") or "") if refused else "",
        )

    def stream(self, system_parts, messages, max_tokens=None, effort=None) -> Iterator[str]:
        params = self._params(system_parts, messages, max_tokens, effort)
        try:
            with self._client.messages.stream(**params) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise RuntimeError(self.describe_error(exc)) from exc

    def describe_error(self, exc: Exception) -> str:
        anthropic = self._anthropic
        if isinstance(exc, (anthropic.AuthenticationError, MissingCredentials)):
            return CREDENTIAL_HINT
        if "could not resolve authentication" in str(exc).lower():
            return CREDENTIAL_HINT
        if isinstance(exc, anthropic.PermissionDeniedError):
            return "The credential is valid but lacks permission for this model."
        if isinstance(exc, anthropic.NotFoundError):
            return f"Unknown model id {self.model!r}. Check RAG_MODEL."
        if isinstance(exc, anthropic.RateLimitError):
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            return f"Rate limited. Retry after {headers.get('retry-after', '60')}s."
        if isinstance(exc, anthropic.APIConnectionError):
            return "Could not reach the Anthropic API. Check network connectivity."
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                return f"Anthropic server error ({exc.status_code}). Retry shortly."
            return f"API error {exc.status_code}: {getattr(exc, 'message', exc)}"
        return str(exc)


def _claude_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cached_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_written_tokens": getattr(usage, "cache_creation_input_tokens", None),
    }
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

# Claude's effort scale mapped onto Gemini's thinking levels. Gemini 3.x thinks
# by default and cannot be switched off, only turned down.
_THINKING_LEVEL = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# Finish reasons that mean the model declined rather than answered.
_GEMINI_REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class GeminiBackend:
    name = "gemini"

    def __init__(self, config: Config):
        self.config = config
        self.model = config.model or GEMINI_DEFAULT_MODEL
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise MissingCredentials("Gemini answers need: pip install google-genai") from exc
        self._genai, self._types, self._errors = genai, types, errors

        vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
        if not vertex and not any(os.environ.get(v) for v in GEMINI_KEY_VARS):
            raise MissingCredentials(CREDENTIAL_HINT)
        try:
            # Reads GEMINI_API_KEY / GOOGLE_API_KEY, or Vertex AI settings.
            self._client = genai.Client()
        except Exception as exc:
            raise MissingCredentials(CREDENTIAL_HINT) from exc

    def _request(self, system_parts, messages, max_tokens, effort) -> dict[str, Any]:
        types = self._types
        contents = [
            types.Content(
                # Gemini calls the assistant turn "model".
                role="model" if m.get("role") == "assistant" else "user",
                parts=[types.Part(text=str(m.get("content", "")))],
            )
            for m in messages
        ]
        level = _THINKING_LEVEL.get((effort or self.config.effort).lower(), "medium")
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(part for part in system_parts if part),
            max_output_tokens=max_tokens or self.config.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=level),
            # No tools are declared, and leaving automatic function calling on
            # makes the SDK warn on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return {"model": self.model, "contents": contents, "config": config}

    def _retryable(self, exc: Exception) -> bool:
        """Overload (503) and rate limits (429) clear on their own; 4xx do not."""
        errors = self._errors
        if isinstance(exc, errors.ServerError):
            return True
        return isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429

    def _with_retries(self, call):
        """Run ``call``, retrying transient failures with a short backoff.

        Gemini returns 503 when its capacity is busy, often enough to be worth
        handling: observed once in a handful of real requests here.
        """
        import time

        delay = self.config.retry_initial_seconds
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return call()
            except Exception as exc:
                if attempt >= self.config.retry_attempts or not self._retryable(exc):
                    raise RuntimeError(self.describe_error(exc)) from exc
                time.sleep(delay)
                delay *= 2

    def complete(self, system_parts, messages, max_tokens=None, effort=None) -> Completion:
        request = self._request(system_parts, messages, max_tokens, effort)
        response = self._with_retries(lambda: self._client.models.generate_content(**request))

        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        candidate = (response.candidates or [None])[0]
        finish = str(getattr(candidate, "finish_reason", "") or "").rsplit(".", 1)[-1]
        refused = bool(blocked) or finish in _GEMINI_REFUSALS
        return Completion(
            text=(response.text or "").strip() if not refused else "",
            model=self.model,
            usage=_gemini_usage(getattr(response, "usage_metadata", None)),
            stop_reason="refusal" if refused else ("max_tokens" if finish == "MAX_TOKENS" else "end_turn"),
            refused=refused,
            refusal_detail=str(blocked or finish or ""),
        )

    def stream(self, system_parts, messages, max_tokens=None, effort=None) -> Iterator[str]:
        request = self._request(system_parts, messages, max_tokens, effort)
        # Only the *opening* of the stream is retried: once text has reached the
        # caller, restarting would repeat it.
        started = False
        try:
            for chunk in self._with_retries(
                lambda: self._client.models.generate_content_stream(**request)
            ):
                text = getattr(chunk, "text", None)
                if text:
                    started = True
                    yield text
        except RuntimeError:
            raise
        except Exception as exc:
            message = self.describe_error(exc)
            if started:
                message = f"The answer was cut short: {message}"
            raise RuntimeError(message) from exc

    def describe_error(self, exc: Exception) -> str:
        errors = self._errors
        if isinstance(exc, MissingCredentials):
            return CREDENTIAL_HINT
        if isinstance(exc, errors.ClientError):
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))
            # An invalid key comes back as a 400, not a 401.
            if "api key not valid" in str(message).lower() or "api_key_invalid" in str(message).lower():
                return (
                    "Gemini rejected the API key, so a key was found but is wrong or "
                    "expired. Check GEMINI_API_KEY in your .env file (or the "
                    "environment, which overrides it). Free key: "
                    "https://aistudio.google.com/apikey"
                )
            if code == 401 or code == 403:
                return f"Gemini rejected the credentials ({code}). Check GEMINI_API_KEY."
            if code == 404:
                return (
                    f"Gemini does not offer model {self.model!r} to this key. "
                    "Set RAG_MODEL to a current model, e.g. gemini-3.8-flash."
                )
            if code == 429:
                return "Gemini rate limit or quota reached. Retry later, or use a different key."
            return f"Gemini rejected the request ({code}): {message}"
        if isinstance(exc, errors.ServerError):
            return f"Gemini server error ({getattr(exc, 'code', '5xx')}). Retry shortly."
        if isinstance(exc, errors.APIError):
            return f"Gemini API error: {getattr(exc, 'message', exc)}"
        return str(exc)


def _gemini_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out = {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "cached_tokens": getattr(usage, "cached_content_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
    }
    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible Chat Completions API)
# --------------------------------------------------------------------------

class OpenRouterBackend:
    name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: Config):
        self.config = config
        self.model = config.model or OPENROUTER_DEFAULT_MODEL
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise MissingCredentials(CREDENTIAL_HINT)

    def _payload(self, system_parts, messages, max_tokens, stream=False) -> dict[str, Any]:
        conversation: list[dict[str, str]] = []
        system = "\n\n".join(part for part in system_parts if part)
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.extend(
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in messages
        )
        return {
            "model": self.model,
            "messages": conversation,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream,
        }

    def _open(self, payload: dict[str, Any]):
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Mizan Jordanian Legal Research",
            },
            method="POST",
        )
        try:
            return urlopen(request, timeout=120)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(self.describe_error(exc)) from exc

    def complete(self, system_parts, messages, max_tokens=None, effort=None) -> Completion:
        del effort
        try:
            with self._open(self._payload(system_parts, messages, max_tokens)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(self.describe_error(exc)) from exc
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        usage = data.get("usage") or {}
        return Completion(
            text=str(message.get("content") or "").strip(),
            model=str(data.get("model") or self.model),
            usage={
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
            stop_reason="max_tokens" if finish == "length" else "end_turn",
            refused=finish == "content_filter",
            refusal_detail="content_filter" if finish == "content_filter" else "",
        )

    def stream(self, system_parts, messages, max_tokens=None, effort=None) -> Iterator[str]:
        del effort
        started = False
        try:
            with self._open(self._payload(system_parts, messages, max_tokens, stream=True)) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    text = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content")
                    if text:
                        started = True
                        yield str(text)
        except RuntimeError:
            raise
        except Exception as exc:
            message = self.describe_error(exc)
            if started:
                message = f"The answer was cut short: {message}"
            raise RuntimeError(message) from exc

    def describe_error(self, exc: Exception) -> str:
        if isinstance(exc, HTTPError):
            code = exc.code
            if code in (401, 403):
                return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY."
            if code == 404:
                return f"OpenRouter does not offer model {self.model!r}. Check RAG_MODEL."
            if code == 429:
                return "OpenRouter rate limit or credit limit reached."
            if code >= 500:
                return f"OpenRouter server error ({code}). Retry shortly."
            return f"OpenRouter rejected the request ({code})."
        if isinstance(exc, (URLError, TimeoutError)):
            return "Could not reach OpenRouter. Check network connectivity."
        return str(exc)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

BACKENDS = {"claude": ClaudeBackend, "gemini": GeminiBackend, "openrouter": OpenRouterBackend}
_CACHE: dict[tuple[str, str], Backend] = {}


def credentials_present() -> dict[str, bool]:
    """Which providers have credentials in the environment."""
    anthropic_profile = any(
        os.path.isdir(os.path.expanduser(p)) for p in ("~/.config/anthropic", "~/.anthropic")
    )
    return {
        "gemini": any(bool(os.environ.get(v)) for v in GEMINI_KEY_VARS)
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"},
        "claude": any(bool(os.environ.get(v)) for v in ANTHROPIC_KEY_VARS) or anthropic_profile,
        "openrouter": any(bool(os.environ.get(v)) for v in OPENROUTER_KEY_VARS),
    }


def resolve_provider(config: Config) -> str:
    """The provider to use: the configured one, or whichever has credentials."""
    choice = (config.llm_provider or "auto").lower()
    if choice in BACKENDS:
        return choice
    if choice not in ("auto", ""):
        raise ValueError(
            f"Unknown RAG_LLM_PROVIDER {choice!r}. Use one of: {', '.join(BACKENDS)}, auto."
        )
    present = credentials_present()
    # Gemini first: it is the one with a free tier, so a key for it is the more
    # likely deliberate choice when both are present.
    for provider in ("gemini", "claude", "openrouter"):
        if present[provider]:
            return provider
    raise MissingCredentials(CREDENTIAL_HINT)


def get_backend(config: Config) -> Backend:
    provider = resolve_provider(config)
    key = (provider, config.model or "")
    if key not in _CACHE:
        _CACHE[key] = BACKENDS[provider](config)
    return _CACHE[key]


def default_model(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_DEFAULT_MODEL
    if provider == "openrouter":
        return OPENROUTER_DEFAULT_MODEL
    return CLAUDE_DEFAULT_MODEL
