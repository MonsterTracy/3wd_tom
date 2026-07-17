import re

import openai

from werewolf.backends.base import BackendError, LLMBackend


_SAFE_MESSAGE_LIMIT = 500
_DETAIL_FIELDS = (
    "type", "code", "param", "message",
)


def _safe_status_code(exc):
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if type(value) is int else None


def _safe_request_id(exc):
    value = getattr(exc, "request_id", None)
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())[:200]
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        items = headers.items()
    except AttributeError:
        return None
    for key, header_value in items:
        if str(key).lower() in {"x-request-id", "request-id", "x-requestid"}:
            if isinstance(header_value, str) and header_value.strip():
                return " ".join(header_value.split())[:200]
    return None


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _safe_detail_scalar(value, messages):
    if isinstance(value, str):
        return _sanitize_message(value, messages)[:200]
    return value if value is None or type(value) in (int, float, bool) else None


def _safe_provider_error(exc):
    candidates = [getattr(exc, "body", None), getattr(exc, "error", None)]
    response = getattr(exc, "response", None)
    response_json = getattr(response, "json", None)
    if callable(response_json):
        try:
            candidates.append(response_json())
        except Exception:
            pass
    for candidate in candidates:
        if candidate is None:
            continue
        nested = _field(candidate, "error")
        value = nested if nested is not None else candidate
        extracted = {name: _field(value, name) for name in _DETAIL_FIELDS}
        if any(item is not None for item in extracted.values()):
            return extracted
    return {
        name: getattr(exc, name, None)
        for name in _DETAIL_FIELDS
    }


def _sanitize_message(value, messages):
    text = value if isinstance(value, str) and value.strip() else "request failed"
    text = " ".join(text.split())
    for message in messages if isinstance(messages, list) else ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            text = text.replace(" ".join(content.split()), "[REDACTED_PROMPT]")
    text = re.sub(
        r"(?i)(DEEPSEEK_API_KEY\s*=\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(Authorization\s*[:=]\s*)(?:Bearer\s+)?[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", text)
    return text[:_SAFE_MESSAGE_LIMIT]


def _is_retryable(exc, status_code):
    retryable_types = tuple(
        value
        for value in (
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "RateLimitError", None),
        )
        if isinstance(value, type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True
    if type(exc).__name__ in {
        "APITimeoutError", "APIConnectionError", "RateLimitError",
    }:
        return True
    return status_code in {408, 429} or (
        status_code is not None and 500 <= status_code <= 599
    )


def _request_error(exc, *, model, messages):
    status_code = _safe_status_code(exc)
    provider = _safe_provider_error(exc)
    retryable = _is_retryable(exc, status_code)
    details = {
        "backend": "openai_compatible",
        "model": model,
        "cause_type": type(exc).__name__,
        "status_code": status_code,
        "provider_error_type": _safe_detail_scalar(provider.get("type"), messages),
        "provider_error_code": _safe_detail_scalar(provider.get("code"), messages),
        "provider_error_param": _safe_detail_scalar(provider.get("param"), messages),
        "request_id": _safe_detail_scalar(_safe_request_id(exc), messages),
        "safe_message": _sanitize_message(provider.get("message"), messages),
    }
    return BackendError(
        "OpenAI-compatible chat request failed.",
        retryable=retryable,
        details=details,
    )


def _response_shape_error(code, message, *, model):
    return BackendError(
        "OpenAI-compatible chat response is invalid.",
        retryable=False,
        details={
            "backend": "openai_compatible",
            "model": model,
            "cause_type": "ResponseShapeError",
            "provider_error_code": code,
            "safe_message": message,
        },
    )


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        api_key=None,
        base_url=None,
        default_model=None,
        client=None,
    ):
        if client is None:
            if not api_key:
                raise BackendError(
                    "api_key is required when an OpenAI-compatible client is not injected."
                )
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)

        self.client = client
        self.default_model = default_model

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        resolved_model = model or self.default_model
        if not resolved_model:
            raise BackendError("A model is required for an LLM chat request.")

        request = dict(kwargs)
        request["model"] = resolved_model
        request["messages"] = messages
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**request)
            choices = getattr(response, "choices", None)
            if not choices:
                raise _response_shape_error(
                    "empty_choices", "response.choices is missing or empty",
                    model=resolved_model,
                )
            message = getattr(choices[0], "message", None)
            if message is None:
                raise _response_shape_error(
                    "missing_message", "response.choices[0].message is missing",
                    model=resolved_model,
                )
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                raise _response_shape_error(
                    "non_text_content",
                    "response.choices[0].message.content is not text",
                    model=resolved_model,
                )
            if not content.strip():
                raise _response_shape_error(
                    "empty_content",
                    "response.choices[0].message.content is empty",
                    model=resolved_model,
                )
            return content
        except BackendError:
            raise
        except Exception as exc:
            raise _request_error(
                exc, model=resolved_model, messages=messages
            ) from exc
