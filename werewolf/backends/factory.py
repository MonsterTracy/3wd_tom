"""Construction of named OpenAI-compatible backends."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from werewolf.backends.base import BackendError, LLMBackend
from werewolf.backends.openai_compatible import OpenAICompatibleBackend
from werewolf.runtime_config import validate_runtime_config


@dataclass(frozen=True)
class BackendSettings:
    backend_type: str
    api_key: str
    base_url: str | None
    default_model: str


def _non_empty(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_backend_settings(config, env_file: str | Path | None = ".env"):
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    if not isinstance(config, dict):
        raise BackendError("backend config must be a mapping")
    backend_type = config.get("type")
    if backend_type != "openai_compatible":
        raise BackendError(f"unsupported backend type: {backend_type}")
    api_key_env = _non_empty(config.get("api_key_env"))
    if api_key_env is None:
        raise BackendError("backend api_key_env is required")
    api_key = _non_empty(os.environ.get(api_key_env))
    if api_key is None:
        raise BackendError(f"API key environment variable {api_key_env} is required")
    model = _non_empty(config.get("default_model"))
    if model is None:
        raise BackendError("backend default_model is required")
    return BackendSettings(
        backend_type=backend_type,
        api_key=api_key,
        base_url=_non_empty(config.get("base_url")),
        default_model=model,
    )


def create_backend(config, env_file: str | Path | None = ".env") -> LLMBackend:
    settings = config if isinstance(config, BackendSettings) else load_backend_settings(config, env_file)
    return OpenAICompatibleBackend(
        api_key=settings.api_key,
        base_url=settings.base_url,
        default_model=settings.default_model,
    )


def load_named_backends(config, env_file: str | Path | None = ".env"):
    validate_runtime_config(config)
    return {
        name: create_backend(backend_config, env_file=env_file)
        for name, backend_config in config["backends"].items()
    }


def resolve_backend(name, backends):
    try:
        return backends[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend {name!r}; available={sorted(backends)}"
        ) from exc
