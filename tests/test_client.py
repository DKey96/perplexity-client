from unittest.mock import MagicMock

import pytest
import requests

from client import PerplexityClient, PerplexityModels
from exceptions import PerplexityClientError


@pytest.fixture
def mock_perplexity_client(monkeypatch):
    mock_post = MagicMock()
    monkeypatch.setattr(requests, 'post', mock_post)
    api_key = 'test_api_key'
    client = PerplexityClient(api_key)
    return client, mock_post


def test_chat_completion_successful(mock_perplexity_client):
    client, mock_post = mock_perplexity_client

    messages = [{"text": "Hello", "role": "user"}, {"text": "Hi", "role": "assistant"}]
    expected_response = {"completion": "Some completion text"}
    mock_post.return_value.json.return_value = expected_response

    response = client.chat_completion(messages)

    assert response == expected_response
    mock_post.assert_called_once()


def test_chat_completion_with_custom_model(mock_perplexity_client):
    client, mock_post = mock_perplexity_client

    messages = [{"text": "Hello", "role": "user"}, {"text": "Hi", "role": "assistant"}]
    custom_model = PerplexityModels.PPLX_70B_CHAT
    expected_response = {"completion": "Some completion text"}
    mock_post.return_value.json.return_value = expected_response

    response = client.chat_completion(messages, model=custom_model)

    assert response == expected_response
    mock_post.assert_called_once()


@pytest.mark.parametrize("kwargs", [
    {"presence_penalty": 0.5, "frequency_penalty": 0.5},
    {"temperature": 2.5},
    {"top_p": 1.5},
    {"top_k": 3000}
])
def test_chat_completion_invalid_arguments(mock_perplexity_client, kwargs):
    client, mock_post = mock_perplexity_client
    messages = [{"text": "Hello", "role": "user"}, {"text": "Hi", "role": "assistant"}]

    with pytest.raises(ValueError):
        client.chat_completion(messages, **kwargs)


@pytest.mark.parametrize("exception", [
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    requests.exceptions.RequestException
])
def test_chat_completion_network_errors(mock_perplexity_client, exception):
    client, mock_post = mock_perplexity_client
    messages = [{"text": "Hello", "role": "user"}, {"text": "Hi", "role": "assistant"}]

    mock_post.side_effect = exception

    with pytest.raises(PerplexityClientError):
        client.chat_completion(messages)


def test_chat_completion_error_response(mock_perplexity_client):
    client, mock_post = mock_perplexity_client
    messages = [{"text": "Hello", "role": "user"}, {"text": "Hi", "role": "assistant"}]

    mock_post.side_effect = requests.exceptions.HTTPError

    with pytest.raises(PerplexityClientError):
        client.chat_completion(messages)
