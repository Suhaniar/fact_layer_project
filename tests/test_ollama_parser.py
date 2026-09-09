from unittest.mock import MagicMock, patch

import pytest
import requests

from app.extraction.ollama_client import OllamaError, call_ollama, extract_json


def test_extract_json_direct():
    parsed, error = extract_json('{"facts": []}')
    assert error is None
    assert parsed == {"facts": []}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"facts": [{"subject": "x"}]}\n```'
    parsed, error = extract_json(raw)
    assert error is None
    assert parsed["facts"][0]["subject"] == "x"


def test_extract_json_with_surrounding_text():
    raw = 'Sure, here is the JSON: {"facts": []} Hope that helps!'
    parsed, error = extract_json(raw)
    assert error is None
    assert parsed == {"facts": []}


def test_extract_json_empty_string():
    parsed, error = extract_json("")
    assert parsed is None
    assert error is not None


def test_extract_json_invalid_json():
    parsed, error = extract_json("{not valid json")
    assert parsed is None
    assert error is not None


@patch("app.extraction.ollama_client.requests.post")
def test_call_ollama_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()
    with pytest.raises(OllamaError):
        call_ollama("some prompt")


@patch("app.extraction.ollama_client.requests.post")
def test_call_ollama_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()
    with pytest.raises(OllamaError):
        call_ollama("some prompt")


@patch("app.extraction.ollama_client.requests.post")
def test_call_ollama_http_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "internal error"
    mock_post.return_value = mock_response
    with pytest.raises(OllamaError):
        call_ollama("some prompt")


@patch("app.extraction.ollama_client.requests.post")
def test_call_ollama_empty_response(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": ""}
    mock_post.return_value = mock_response
    with pytest.raises(OllamaError):
        call_ollama("some prompt")


@patch("app.extraction.ollama_client.requests.post")
def test_call_ollama_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": '{"facts": []}'}
    mock_post.return_value = mock_response
    assert call_ollama("some prompt") == '{"facts": []}'