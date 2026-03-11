"""Test provider switching with environment variable cleanup."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import FailoverConfig
from nanobot.providers.base import LLMResponse
from nanobot.providers.failover_provider import FailoverProvider
from nanobot.providers.litellm_provider import LiteLLMProvider


@pytest.fixture
def mock_config_with_providers():
    """Create a mock config with multiple providers for testing switching."""
    config = MagicMock()
    config.agents.defaults.model = "anthropic/claude-3-5-sonnet"

    providers = MagicMock()
    # Configure zhipu provider
    providers.zhipu.api_key = "zhipu-test-key-123"
    providers.zhipu.api_base = None
    providers.zhipu.model = "zhipu/glm-4"
    providers.zhipu.extra_headers = None

    # Configure minimax provider
    providers.minimax.api_key = "minimax-test-key-456"
    providers.minimax.api_base = None
    providers.minimax.model = "minimax/minimax-6b"
    providers.minimax.extra_headers = None

    # Configure openai provider
    providers.openai.api_key = "sk-openai-test-789"
    providers.openai.api_base = None
    providers.openai.model = "gpt-4o"
    providers.openai.extra_headers = None

    providers.failover = FailoverConfig(
        enabled=True,
        max_retries_per_provider=2,
        provider_priority=["zhipu", "minimax", "openai"],
    )

    config.providers = providers
    return config


class TestProviderSwitchCleanup:
    """Test provider switching with environment variable cleanup."""

    def test_litellm_provider_tracks_env_vars(self, mock_config_with_providers):
        """Test that LiteLLMProvider tracks environment variables it sets."""
        # Save current env state
        old_env = dict(os.environ)

        try:
            # Create a zhipu provider
            provider = LiteLLMProvider(
                api_key="zhipu-test-key-123",
                provider_name="zhipu",
                default_model="zhipu/glm-4",
            )

            # Check that env vars were tracked
            assert len(provider._env_vars_set) > 0

            # Verify the tracked env vars are actually set
            for env_var in provider._env_vars_set:
                assert env_var in os.environ, f"Env var {env_var} should be set"

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)

    def test_litellm_provider_cleanup_removes_env_vars(self, mock_config_with_providers):
        """Test that cleanup() removes all tracked environment variables."""
        # Save current env state
        old_env = dict(os.environ)

        try:
            # Create a zhipu provider
            provider = LiteLLMProvider(
                api_key="zhipu-test-key-123",
                provider_name="zhipu",
                default_model="zhipu/glm-4",
            )

            # Get the tracked env vars before cleanup
            tracked_vars = list(provider._env_vars_set)

            # Call cleanup
            provider.cleanup()

            # Verify all tracked env vars are removed
            for env_var in tracked_vars:
                assert env_var not in os.environ, f"Env var {env_var} should be removed after cleanup"
                assert env_var not in provider._env_vars_set, f"Env var {env_var} should be removed from tracking"

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)

    def test_failover_switch_calls_cleanup(self, mock_config_with_providers):
        """Test that switch_to_next_provider() calls cleanup on old provider."""
        # Save current env state
        old_env = dict(os.environ)

        try:
            provider = FailoverProvider(
                config=mock_config_with_providers,
                model="anthropic/claude-3-5-sonnet",
            )

            # Get the first provider instance
            first_provider_name = provider.get_current_provider()
            first_provider = provider._get_provider_instance(first_provider_name)

            # Mock the cleanup method to verify it's called
            if hasattr(first_provider, 'cleanup'):
                first_provider.cleanup = MagicMock()

            # Switch to next provider
            new_provider_name = provider.switch_to_next_provider()

            # Verify we switched
            assert new_provider_name is not None
            assert new_provider_name != first_provider_name

            # Verify cleanup was called if the provider has that method
            if hasattr(first_provider, 'cleanup'):
                first_provider.cleanup.assert_called_once()

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)

    def test_no_env_var_contamination_after_switch(self, mock_config_with_providers):
        """Test that environment variables don't contaminate after switching providers."""
        # Save current env state
        old_env = dict(os.environ)

        try:
            provider = FailoverProvider(
                config=mock_config_with_providers,
                model="anthropic/claude-3-5-sonnet",
            )

            # Create first provider instance and track its env vars
            first_provider_name = provider.get_current_provider()
            first_provider = provider._get_provider_instance(first_provider_name)
            first_provider_env_vars = set()

            if hasattr(first_provider, '_env_vars_set'):
                first_provider_env_vars = set(first_provider._env_vars_set)

            # Switch to second provider
            second_provider_name = provider.switch_to_next_provider()
            assert second_provider_name is not None

            # Create second provider instance
            second_provider = provider._get_provider_instance(second_provider_name)

            # The first provider's tracked env vars should be cleaned up
            if hasattr(first_provider, '_env_vars_set'):
                # After cleanup, the first provider's env vars should be gone
                for env_var in first_provider_env_vars:
                    # This env var should not be in the first provider's tracking anymore
                    assert env_var not in first_provider._env_vars_set

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)

    @pytest.mark.asyncio
    async def test_switch_then_call_succeeds(self, mock_config_with_providers):
        """Test that after switching, the new provider can be called successfully."""
        # Save current env state
        old_env = dict(os.environ)

        try:
            provider = FailoverProvider(
                config=mock_config_with_providers,
                model="anthropic/claude-3-5-sonnet",
            )

            # Create mock providers for testing
            first_provider = AsyncMock()
            first_provider.get_default_model = MagicMock(return_value="zhipu/glm-4")
            first_provider.chat.side_effect = Exception("Zhipu failed")
            first_provider.cleanup = MagicMock()

            second_provider = AsyncMock()
            second_provider.get_default_model = MagicMock(return_value="minimax/minimax-6b")
            second_provider.chat.return_value = LLMResponse(
                content="Hello from MiniMax!",
                finish_reason="stop",
            )
            second_provider.cleanup = MagicMock()

            # Inject mock providers
            first_name = provider._providers[0]
            second_name = provider._providers[1]
            provider._provider_instances[first_name] = first_provider
            provider._provider_instances[second_name] = second_provider

            # Manually switch (simulating /switch command)
            new_name = provider.switch_to_next_provider()
            assert new_name == second_name

            # Verify cleanup was called on the first provider
            first_provider.cleanup.assert_called_once()

            # Now try to chat with the new provider
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.chat(messages)

            # Should get response from second provider
            assert response.content == "Hello from MiniMax!"
            assert response.finish_reason == "stop"

            # Second provider's chat should have been called
            second_provider.chat.assert_called_once()

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
