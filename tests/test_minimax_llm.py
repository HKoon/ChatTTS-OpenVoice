"""Unit tests for MiniMax LLM provider in ChatTTS/experimental/llm.py."""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub heavy dependencies so the module can be imported without GPU/torch/vocos
# ---------------------------------------------------------------------------
def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _name in [
    "vocos", "torch", "torchaudio", "torchvision",
    "transformers", "omegaconf", "vector_quantize_pytorch",
    "huggingface_hub", "ChatTTS.core", "ChatTTS.model",
    "ChatTTS.model.dvae", "ChatTTS.model.gpt",
    "ChatTTS.utils", "ChatTTS.utils.gpu_utils",
    "ChatTTS.utils.infer_utils", "ChatTTS.utils.io_utils",
    "ChatTTS.infer", "ChatTTS.infer.api",
]:
    if _name not in sys.modules:
        _stub(_name)

# Provide a minimal OpenAI stub
if "openai" not in sys.modules:
    _stub("openai")


class _FakeChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def create(self, **kwargs):
        return _FakeCompletion("mock response")


class _FakeChat:
    completions = _FakeChatCompletions()


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = _FakeChat()


sys.modules["openai"].OpenAI = _FakeOpenAI

# Patch ChatTTS __init__ to avoid importing core
chattts_pkg = types.ModuleType("ChatTTS")
chattts_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "ChatTTS")]
chattts_pkg.__package__ = "ChatTTS"
sys.modules["ChatTTS"] = chattts_pkg

# Load the experimental sub-package
exp_pkg = types.ModuleType("ChatTTS.experimental")
exp_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "ChatTTS", "experimental")]
exp_pkg.__package__ = "ChatTTS.experimental"
sys.modules["ChatTTS.experimental"] = exp_pkg

# Now import the module under test directly via spec
_llm_path = os.path.join(os.path.dirname(__file__), "..", "ChatTTS", "experimental", "llm.py")
_spec = importlib.util.spec_from_file_location("ChatTTS.experimental.llm", _llm_path)
_llm_mod = importlib.util.module_from_spec(_spec)
sys.modules["ChatTTS.experimental.llm"] = _llm_mod
_spec.loader.exec_module(_llm_mod)

MINIMAX_BASE_URL = _llm_mod.MINIMAX_BASE_URL
MINIMAX_MODELS = _llm_mod.MINIMAX_MODELS
create_minimax_client = _llm_mod.create_minimax_client
llm_api = _llm_mod.llm_api
prompt_dict = _llm_mod.prompt_dict


class TestMinimaxConstants(unittest.TestCase):
    def test_base_url(self):
        self.assertEqual(MINIMAX_BASE_URL, "https://api.minimax.io/v1")

    def test_models_list(self):
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MINIMAX_MODELS)
        self.assertEqual(len(MINIMAX_MODELS), 4)


class TestPromptDict(unittest.TestCase):
    def test_minimax_prompt_exists(self):
        self.assertIn("minimax", prompt_dict)

    def test_minimax_tn_prompt_exists(self):
        self.assertIn("minimax_TN", prompt_dict)

    def test_minimax_prompt_has_system_role(self):
        roles = [m["role"] for m in prompt_dict["minimax"]]
        self.assertIn("system", roles)

    def test_minimax_tn_contains_number_example(self):
        contents = [m["content"] for m in prompt_dict["minimax_TN"]]
        self.assertTrue(any("$123" in c for c in contents))

    def test_existing_prompts_unchanged(self):
        self.assertIn("kimi", prompt_dict)
        self.assertIn("deepseek", prompt_dict)
        self.assertIn("deepseek_TN", prompt_dict)


class TestLlmApiTemperatureClamping(unittest.TestCase):
    """MiniMax requires temperature > 0; zero should be clamped."""

    def _make_client(self):
        return llm_api(
            api_key="test_key",
            base_url=MINIMAX_BASE_URL,
            model="MiniMax-M2.7",
        )

    def test_zero_temperature_clamped(self):
        client = self._make_client()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeCompletion("ok")

        client.client.chat.completions.create = fake_create
        client.call("hello", temperature=0.0, prompt_version="minimax")
        self.assertGreater(captured["temperature"], 0)

    def test_positive_temperature_unchanged(self):
        client = self._make_client()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeCompletion("ok")

        client.client.chat.completions.create = fake_create
        client.call("hello", temperature=0.5, prompt_version="minimax")
        self.assertEqual(captured["temperature"], 0.5)

    def test_call_returns_string(self):
        client = self._make_client()
        result = client.call("hello", prompt_version="minimax")
        self.assertIsInstance(result, str)


class TestCreateMinimaxClient(unittest.TestCase):
    def test_returns_llm_api_instance(self):
        client = create_minimax_client("test_key")
        self.assertIsInstance(client, llm_api)

    def test_default_model(self):
        client = create_minimax_client("test_key")
        self.assertEqual(client.model, "MiniMax-M2.7")

    def test_custom_model(self):
        client = create_minimax_client("test_key", model="MiniMax-M2.5-highspeed")
        self.assertEqual(client.model, "MiniMax-M2.5-highspeed")

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            create_minimax_client("test_key", model="gpt-4o")

    def test_base_url_set(self):
        client = create_minimax_client("test_key")
        self.assertEqual(client.client.base_url, MINIMAX_BASE_URL)

    def test_api_key_passed(self):
        client = create_minimax_client("my_secret_key")
        self.assertEqual(client.client.api_key, "my_secret_key")


class TestIntegrationMinimax(unittest.TestCase):
    """Integration-style test: end-to-end call flow without a real API."""

    def test_minimax_call_uses_correct_prompt(self):
        client = create_minimax_client("test_key")
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeCompletion("integrated response")

        client.client.chat.completions.create = fake_create
        result = client.call("Tell me a joke.", prompt_version="minimax")

        self.assertEqual(result, "integrated response")
        self.assertEqual(captured["model"], "MiniMax-M2.7")
        messages = captured["messages"]
        self.assertEqual(messages[-1]["content"], "Tell me a joke.")
        self.assertEqual(messages[0]["role"], "system")

    def test_minimax_tn_call(self):
        client = create_minimax_client("test_key", model="MiniMax-M2.7-highspeed")
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeCompletion("normalized text")

        client.client.chat.completions.create = fake_create
        result = client.call("We paid $456 for this chair.", prompt_version="minimax_TN")
        self.assertEqual(result, "normalized text")
        self.assertEqual(captured["model"], "MiniMax-M2.7-highspeed")

    def test_minimax_env_key(self):
        """create_minimax_client should accept key from env variable pattern."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env_key"}):
            key = os.environ["MINIMAX_API_KEY"]
            client = create_minimax_client(key)
            self.assertEqual(client.client.api_key, "env_key")


if __name__ == "__main__":
    unittest.main()
