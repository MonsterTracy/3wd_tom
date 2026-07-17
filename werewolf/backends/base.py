from abc import ABC, abstractmethod
from types import MappingProxyType


class BackendError(RuntimeError):
    """An LLM backend failure with safe, serialization-ready diagnostics."""

    def __init__(self, message, *, retryable=False, details=None):
        if not isinstance(message, str) or not message.strip():
            raise TypeError("BackendError message must be non-empty text")
        if type(retryable) is not bool:
            raise TypeError("BackendError retryable must be bool")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise TypeError("BackendError details must be a dict")
        safe = {}
        for key, value in details.items():
            if not isinstance(key, str) or not key:
                raise TypeError("BackendError detail keys must be non-empty text")
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError("BackendError detail values must be safe scalars")
            safe[key] = (
                " ".join(value.split())[:1000]
                if isinstance(value, str)
                else value
            )
        self.message = " ".join(message.split())[:1000]
        self.retryable = retryable
        self._safe_details = safe
        parts = [self.message, f"retryable={str(retryable).lower()}"]
        parts.extend(
            f"{key}={value}" for key, value in safe.items() if value is not None
        )
        super().__init__("; ".join(parts))

    @property
    def details(self):
        return dict(self._safe_details)

    @property
    def safe_details(self):
        return MappingProxyType(dict(self._safe_details))


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        raise NotImplementedError
