import openai

from werewolf.backends.base import BackendError, LLMBackend


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
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise BackendError(
                    "OpenAI-compatible chat response content must be text."
                )
            return content
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "OpenAI-compatible chat request failed."
            ) from exc
