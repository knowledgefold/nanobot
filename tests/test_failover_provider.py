"""Tests for FailoverProvider functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import Config, FailoverConfig, ProvidersConfig
from nanobot.providers.base import LLMResponse
from nanobot.providers.failover_provider import FailoverProvider


@pytest.fixture
def mock_config():
    """Create a mock config with multiple providers."""
    config = MagicMock()  # Don't use spec to allow dynamic attribute access
    config.agents.defaults.model = "anthropic/claude-3-5-sonnet"

    # Mock provider configs - no spec to allow dynamic attributes
    providers = MagicMock()
    providers.anthropic.api_key = "sk-ant-test"
    providers.openai.api_key = "sk-openai-test"
    providers.deepseek.api_key = "sk-deepseek-test"
    providers.groq.api_key = "sk-groq-test"
    providers.failover = FailoverConfig(
        enabled=True,
        max_retries_per_provider=2,
        provider_priority=["anthropic", "openai", "deepseek", "groq"],
        model_mapping={
            "openai": {"anthropic/claude-3-5-sonnet": "gpt-4o"},
            "deepseek": {"anthropic/claude-3-5-sonnet": "deepseek-chat"},
        },
    )

    # Mock get_provider and get_provider_name
    def mock_get_provider(model):
        model_lower = model.lower()
        if "anthropic" in model_lower or "claude" in model_lower:
            return providers.anthropic
        if "openai" in model_lower or "gpt" in model_lower:
            return providers.openai
        if "deepseek" in model_lower:
            return providers.deepseek
        if "groq" in model_lower:
            return providers.groq
        return providers.anthropic  # default

    def mock_get_provider_name(model):
        model_lower = model.lower()
        if "anthropic" in model_lower or "claude" in model_lower:
            return "anthropic"
        if "openai" in model_lower or "gpt" in model_lower:
            return "openai"
        if "deepseek" in model_lower:
            return "deepseek"
        if "groq" in model_lower:
            return "groq"
        return "anthropic"

    config.providers = providers
    config.get_provider = mock_get_provider
    config.get_provider_name = mock_get_provider_name
    config.get_api_base = MagicMock(return_value=None)

    return config


class TestFailoverProvider:
    """Test suite for FailoverProvider."""

    def test_provider_list_building(self, mock_config):
        """Test that provider list is built correctly based on availability."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Should include all configured providers
        assert "anthropic" in provider._providers
        assert "openai" in provider._providers
        assert "deepseek" in provider._providers
        assert "groq" in provider._providers

    def test_model_mapping(self, mock_config):
        """Test that model names are mapped correctly for different providers."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Test mapping for OpenAI
        mapped = provider._map_model_for_provider("openai", "anthropic/claude-3-5-sonnet")
        assert mapped == "gpt-4o"

        # Test mapping for DeepSeek
        mapped = provider._map_model_for_provider("deepseek", "anthropic/claude-3-5-sonnet")
        assert mapped == "deepseek-chat"

        # Test no mapping (original model returned)
        mapped = provider._map_model_for_provider("groq", "anthropic/claude-3-5-sonnet")
        assert mapped == "anthropic/claude-3-5-sonnet"

    def test_error_classification(self, mock_config):
        """Test that errors are classified correctly."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        from httpx import ConnectError
        from litellm import AuthenticationError, APIError

        # TimeoutError is a built-in Python exception
        assert provider._classify_error(TimeoutError()) == "retryable"
        assert provider._classify_error(ConnectError("Connection failed")) == "retryable"

        # Auth errors are fatal (requires llm_provider and model args)
        assert provider._classify_error(AuthenticationError(
            llm_provider="anthropic", model="claude-3-5-sonnet", message="Auth failed"
        )) == "fatal"

        # API errors with 429 are retryable
        api_error_429 = APIError(
            status_code=429, message="Rate limit",
            llm_provider="anthropic", model="claude-3-5-sonnet"
        )
        assert provider._classify_error(api_error_429) == "retryable"

        # API errors with 401 are fatal
        api_error_401 = APIError(
            status_code=401, message="Unauthorized",
            llm_provider="anthropic", model="claude-3-5-sonnet"
        )
        assert provider._classify_error(api_error_401) == "fatal"

        # API errors with 500 are retryable
        api_error_500 = APIError(
            status_code=500, message="Server error",
            llm_provider="anthropic", model="claude-3-5-sonnet"
        )
        assert provider._classify_error(api_error_500) == "retryable"

    def test_is_provider_available(self, mock_config):
        """Test provider availability detection."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Providers with API keys are available
        assert provider._is_provider_available("anthropic")
        assert provider._is_provider_available("openai")
        assert provider._is_provider_available("deepseek")

        # Unknown providers are not available
        assert not provider._is_provider_available("unknown_provider")

    @pytest.mark.asyncio
    async def test_successful_chat_primary_provider(self, mock_config):
        """Test successful chat with primary provider."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Mock primary provider to succeed
        mock_primary = AsyncMock()
        mock_primary.chat.return_value = LLMResponse(
            content="Hello from Claude!",
            finish_reason="stop",
        )

        provider._provider_instances["anthropic"] = mock_primary

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        assert response.content == "Hello from Claude!"
        assert response.finish_reason == "stop"
        mock_primary.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_failover_on_fatal_error(self, mock_config):
        """Test failover to secondary provider on fatal error."""
        from litellm import AuthenticationError

        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Mock primary provider to fail with fatal auth error
        mock_primary = AsyncMock()
        mock_primary.chat.side_effect = AuthenticationError(
            llm_provider="anthropic",
            model="claude-3-5-sonnet",
            message="Invalid API key"
        )

        # Mock secondary provider to succeed
        mock_secondary = AsyncMock()
        mock_secondary.chat.return_value = LLMResponse(
            content="Hello from GPT-4o!",
            finish_reason="stop",
        )

        provider._provider_instances["anthropic"] = mock_primary
        provider._provider_instances["openai"] = mock_secondary

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        # Should have failed over to OpenAI
        assert response.content == "Hello from GPT-4o!"
        assert mock_primary.chat.call_count == 1  # Tried once (fatal error, no retry)
        mock_secondary.chat.assert_called_once()

        # Verify the model was mapped for OpenAI
        call_kwargs = mock_secondary.chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, mock_config):
        """Test retry behavior on timeout errors."""
        # Create config with shorter retry settings for testing
        mock_config.providers.failover.max_retries_per_provider = 2
        mock_config.providers.failover.initial_backoff_seconds = 0.01

        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Mock primary provider to fail twice then succeed
        mock_primary = AsyncMock()

        # Use built-in TimeoutError
        mock_primary.chat.side_effect = [
            TimeoutError("Request timeout"),
            TimeoutError("Request timeout"),
            LLMResponse(content="Success after retries!", finish_reason="stop"),
        ]

        provider._provider_instances["anthropic"] = mock_primary

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        assert response.content == "Success after retries!"
        assert mock_primary.chat.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_failover_disabled(self, mock_config):
        """Test that failover is disabled when configured."""
        mock_config.providers.failover.enabled = False

        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Mock only primary provider
        mock_primary = AsyncMock()
        mock_primary.chat.return_value = LLMResponse(
            content="Hello from Claude!",
            finish_reason="stop",
        )

        provider._provider_instances["anthropic"] = mock_primary

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        assert response.content == "Hello from Claude!"
        mock_primary.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, mock_config):
        """Test behavior when all providers fail."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        # Mock all providers to fail
        for name in ["anthropic", "openai", "deepseek", "groq"]:
            mock_provider = AsyncMock()
            mock_provider.chat.side_effect = Exception("Provider error")
            provider._provider_instances[name] = mock_provider

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        assert "All LLM providers failed" in response.content
        assert response.finish_reason == "error"

    def test_default_model(self, mock_config):
        """Test get_default_model returns correct model."""
        provider = FailoverProvider(
            config=mock_config,
            model="anthropic/claude-3-5-sonnet",
        )

        assert provider.get_default_model() == "anthropic/claude-3-5-sonnet"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
