import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

import sys


# Ensure project root is on sys.path so that `owg_mod` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from owg_mod import model_utils_litellm as mul  # noqa: E402


class _DummyResponse:
    def __init__(self, status_code: int, json_data: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self) -> Dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise ValueError(f"HTTP {self.status_code}")


# ------------------------
# encode_image_to_base64
# ------------------------


def test_encode_image_to_base64_from_path_and_pil_and_array(tmp_path):
    # Create a simple red RGB image
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))

    # As file path
    img_path = tmp_path / "test.jpg"
    img.save(img_path, format="JPEG")

    b64_from_path = mul.encode_image_to_base64(str(img_path))
    raw_from_path = base64.b64decode(b64_from_path.encode("utf-8"))
    assert len(raw_from_path) > 0

    # As PIL image
    b64_from_pil = mul.encode_image_to_base64(img)
    raw_from_pil = base64.b64decode(b64_from_pil.encode("utf-8"))
    assert len(raw_from_pil) > 0

    # As numpy array
    arr = np.array(img)
    b64_from_array = mul.encode_image_to_base64(arr)
    raw_from_array = base64.b64decode(b64_from_array.encode("utf-8"))
    assert len(raw_from_array) > 0


# ------------------------
# check_litellm
# ------------------------


def test_check_litellm_running(monkeypatch):
    calls: list[Dict[str, Any]] = []

    def fake_get(url, timeout=3):
        calls.append({"url": url, "timeout": timeout})
        if "health" in url:
            return _DummyResponse(200)
        # models endpoint
        if url.endswith("/v1/models"):
            return _DummyResponse(
                200,
                {
                    "data": [
                        {"id": "gpt-4o"},
                        {"id": "gpt-4o-mini"},
                    ]
                },
            )
        return _DummyResponse(404)

    monkeypatch.setattr(mul, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    result = mul.check_litellm(endpoints=["http://localhost:4000/health"])

    assert result["running"] is True
    assert result["endpoint"] == "http://localhost:4000"
    assert "gpt-4o" in result["models"]
    assert result["error"] is None


def test_check_litellm_not_running(monkeypatch):
    def fake_get(url, timeout=3):
        raise mul.RequestException("connection error")

    monkeypatch.setattr(mul, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    result = mul.check_litellm(endpoints=["http://localhost:4000/health"])
    assert result["running"] is False
    assert result["endpoint"] is None
    assert result["models"] is None
    assert "not reachable" in result["error"].lower()


# ------------------------
# LiteLLMRequestHandler.prepare_messages
# ------------------------


def test_prepare_messages_builds_expected_structure(tmp_path):
    handler = mul.LiteLLMRequestHandler(api_url="http://localhost:4000")

    # Simple dummy image
    img = Image.new("RGB", (2, 2), color=(0, 255, 0))
    images = [img]
    prompt = "Describe the object."
    system_prompt = "You are a helpful assistant."

    in_ctx_img = Image.new("RGB", (2, 2), color=(0, 0, 255))
    in_context_examples = [
        {
            "prompt": "Example prompt",
            "images": [in_ctx_img],
            "response": "Example response",
        }
    ]

    messages = handler.prepare_messages(
        images=images,
        prompt=prompt,
        system_prompt=system_prompt,
        in_context_examples=in_context_examples,
    )

    # System message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt

    # User message
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]

    # Should contain example text, example image, expected response, main prompt, and main image
    types = [c["type"] for c in content]
    assert "text" in types
    assert "image_url" in types
    assert any("The answer should be" in c.get("text", "") for c in content if c["type"] == "text")
    assert any(c.get("text") == prompt for c in content if c["type"] == "text")


# ------------------------
# LiteLLMRequestHandler.request & wrappers
# ------------------------


def test_litellm_request_builds_payload_and_parses_response(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        # Simple chat completion-style response
        return _DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from model",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(mul, "requests", type("R", (), {"post": staticmethod(fake_post)}))

    handler = mul.LiteLLMRequestHandler(api_url="http://localhost:4000")
    img = Image.new("RGB", (2, 2), color=(255, 255, 0))

    result = handler.request(
        images=img,
        prompt="Say hi",
        system_prompt="You are friendly.",
        model_name="gpt-4o",
        max_tokens=128,
        temperature=0.3,
        n=2,
        seed=123,
        return_logprobs=True,
        timeout=10,
    )

    # Returned text
    assert result == "Hello from model"

    # URL normalized to /v1/chat/completions
    assert captured["url"].endswith("/v1/chat/completions")

    # Headers contain Authorization and JSON content-type
    assert "Authorization" in captured["headers"]
    assert captured["headers"]["Content-Type"] == "application/json"

    payload = captured["json"]
    assert payload["model"] == "gpt-4o"
    assert payload["n"] == 2
    assert payload.get("max_completion_tokens") == 128
    assert payload["temperature"] == 0.3
    assert payload["seed"] == 123
    assert payload["logprobs"] is True
    assert isinstance(payload["messages"], list) and len(payload["messages"]) >= 2


def test_litellm_request_gpt5_nano_temperature_override(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _DummyResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(mul, "requests", type("R", (), {"post": staticmethod(fake_post)}))

    handler = mul.LiteLLMRequestHandler(api_url="http://localhost:4000")
    img = Image.new("RGB", (2, 2), color=(0, 0, 0))
    handler.request(
        images=img,
        prompt="test",
        system_prompt="sys",
        model_name="gpt-5-nano",
        temperature=0.0,
        max_tokens=64,
    )

    payload = captured["json"]
    # For gpt-5-nano, temperature should be forced to 1
    assert payload["temperature"] == 1
    assert payload.get("max_completion_tokens") == 64


def test_request_model_and_request_gpt_wrappers(monkeypatch):
    # Spy on LiteLLMRequestHandler.request
    calls: list[Dict[str, Any]] = []

    class DummyHandler:
        def __init__(self, api_url=None, api_key=None):
            self.api_url = api_url
            self.api_key = api_key

        def request(self, images, prompt, system_prompt, model_name, **kwargs):
            calls.append(
                {
                    "images": images,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "model_name": model_name,
                    "kwargs": kwargs,
                }
            )
            return "dummy-response"

    monkeypatch.setattr(mul, "LiteLLMRequestHandler", DummyHandler)

    img = np.zeros((2, 2, 3), dtype=np.uint8)

    # request_model
    out1 = mul.request_model(
        images=img,
        prompt="p1",
        system_prompt="s1",
        model_name="gpt-4o-mini",
        litellm_api_url="http://localhost:4001",
        litellm_api_key="key123",
        temperature=0.2,
    )
    assert out1 == "dummy-response"

    # request_gpt (backward compatibility)
    out2 = mul.request_gpt(
        images=img,
        prompt="p2",
        system_prompt="s2",
        temp=0.4,
        n_tokens=32,
        n=3,
        model_name="gpt-4o",
        seed=7,
    )
    assert out2 == "dummy-response"

    # We should have at least two calls captured with expected values
    assert len(calls) >= 2
    c1, c2 = calls[0], calls[1]
    assert c1["prompt"] == "p1"
    assert c1["system_prompt"] == "s1"
    assert c1["model_name"] == "gpt-4o-mini"
    assert c1["kwargs"]["temperature"] == 0.2

    assert c2["prompt"] == "p2"
    assert c2["system_prompt"] == "s2"
    assert c2["model_name"] == "gpt-4o"
    assert c2["kwargs"]["temperature"] == 0.4
    assert c2["kwargs"]["max_tokens"] == 32
    assert c2["kwargs"]["n"] == 3
    assert c2["kwargs"]["seed"] == 7



