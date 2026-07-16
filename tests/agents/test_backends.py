import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from werewolf.backends import (
        BackendError,
        BackendSettings,
        LLMBackend,
        OpenAICompatibleBackend,
        create_backend,
        load_backend_settings,
    )
except ModuleNotFoundError:
    BackendError = None
    BackendSettings = None
    LLMBackend = None
    OpenAICompatibleBackend = None
    create_backend = None
    load_backend_settings = None


class BackendAvailabilityTest(unittest.TestCase):
    def test_backend_package_is_available(self):
        self.assertIsNotNone(LLMBackend)


class FakeCompletions:
    def __init__(self, content="backend response", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, content="backend response", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


@unittest.skipIf(LLMBackend is None, "backend package is not implemented")
class BackendTest(unittest.TestCase):
    def test_fake_backend_can_implement_common_interface(self):
        class FakeBackend(LLMBackend):
            def chat(
                self,
                messages,
                model=None,
                temperature=0.7,
                max_tokens=None,
                response_format=None,
                **kwargs,
            ):
                return "fake response"

        self.assertEqual(FakeBackend().chat([{"role": "user", "content": "hi"}]), "fake response")

    def test_openai_backend_uses_injected_client_without_api_key(self):
        client = FakeClient(content="model output")
        backend = OpenAICompatibleBackend(
            api_key=None,
            default_model="default-model",
            client=client,
        )

        result = backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="override-model",
            temperature=0.2,
            max_tokens=123,
            response_format={"type": "json_object"},
            seed=7,
        )

        self.assertEqual(result, "model output")
        self.assertEqual(
            client.completions.calls,
            [
                {
                    "model": "override-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.2,
                    "max_tokens": 123,
                    "response_format": {"type": "json_object"},
                    "seed": 7,
                }
            ],
        )

    def test_openai_backend_omits_none_optional_parameters(self):
        client = FakeClient()
        backend = OpenAICompatibleBackend(client=client, default_model="model")

        backend.chat(
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        self.assertEqual(
            client.completions.calls[0],
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    def test_openai_backend_requires_model(self):
        backend = OpenAICompatibleBackend(client=FakeClient())

        with self.assertRaises(BackendError):
            backend.chat(messages=[])

    def test_openai_backend_wraps_provider_errors(self):
        backend = OpenAICompatibleBackend(
            client=FakeClient(error=RuntimeError("provider failed")),
            default_model="model",
        )

        with self.assertRaises(BackendError) as raised:
            backend.chat(messages=[])

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_openai_backend_rejects_non_text_content(self):
        backend = OpenAICompatibleBackend(
            client=FakeClient(content=None),
            default_model="model",
        )

        with self.assertRaises(BackendError):
            backend.chat(messages=[])

    def test_settings_precedence_and_independent_models(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = os.path.join(tmp_dir, ".env")
            with open(env_path, "w", encoding="utf-8") as env_file:
                env_file.write(
                    "OPENAI_API_KEY=dotenv-key\n"
                    "OPENAI_API_BASE=https://dotenv.example/v1\n"
                    "DEFAULT_LLM_MODEL=dotenv-default\n"
                    "AGENT_MODEL=dotenv-agent\n"
                    "PARSER_MODEL=dotenv-parser\n"
                )

            process_env = {
                "OPENAI_API_KEY": "process-key",
                "AGENT_MODEL": "process-agent",
            }
            config = {
                "api_key": "config-key",
                "base_url": "https://config.example/v1",
                "parser_model": "config-parser",
            }
            with patch.dict(os.environ, process_env, clear=True):
                settings = load_backend_settings(config=config, env_file=env_path)

        self.assertEqual(settings.api_key, "config-key")
        self.assertEqual(settings.base_url, "https://config.example/v1")
        self.assertEqual(settings.agent_model, "process-agent")
        self.assertEqual(settings.parser_model, "config-parser")
        self.assertEqual(settings.default_model, "dotenv-default")

    def test_specialized_models_fall_back_to_default(self):
        env = {
            "OPENAI_API_KEY": "key",
            "DEFAULT_LLM_MODEL": "default-model",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_backend_settings(config={}, env_file=None)

        self.assertEqual(settings.agent_model, "default-model")
        self.assertEqual(settings.parser_model, "default-model")

    def test_factory_rejects_missing_production_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(BackendError):
                load_backend_settings(
                    config={"default_model": "model"},
                    env_file=None,
                )

    def test_factory_creates_openai_compatible_backend(self):
        settings = BackendSettings(
            backend_type="openai_compatible",
            api_key="dummy-key",
            base_url="https://example.invalid/v1",
            default_model="default",
            agent_model="agent",
            parser_model="parser",
        )

        backend = create_backend(settings)

        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(backend.default_model, "default")


if __name__ == "__main__":
    unittest.main()
