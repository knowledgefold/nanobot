"""Test provider rotation for /switch command."""

import os
from unittest.mock import MagicMock

import pytest

from nanobot.config.schema import FailoverConfig
from nanobot.providers.base import LLMResponse
from nanobot.providers.failover_provider import FailoverProvider


@pytest.fixture
def mock_config_with_providers():
    """Create a mock config with multiple providers for testing rotation."""
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


class TestProviderRotation:
    """Test provider rotation behavior for /switch command."""

    def test_rotation_moves_provider_to_end(self, mock_config_with_providers):
        """Test that rotation moves the current provider to the end of the list."""
        provider = FailoverProvider(
            config=mock_config_with_providers,
            model="anthropic/claude-3-5-sonnet",
        )

        # Initial state
        assert provider.get_current_provider() == "zhipu"
        initial_count = len(provider._providers)
        assert initial_count >= 3  # At least our 3 configured providers

        # First rotation: zhipu -> next provider
        new_provider = provider.switch_to_next_provider()
        assert new_provider is not None
        assert new_provider != "zhipu"
        assert provider.get_current_provider() == new_provider
        assert len(provider._providers) == initial_count  # List length unchanged

        # zhipu should now be at the end
        assert provider._providers[-1] == "zhipu"

    def test_rotation_cycles_through_all_providers(self, mock_config_with_providers):
        """Test that rotation cycles through all providers and returns to the first."""
        provider = FailoverProvider(
            config=mock_config_with_providers,
            model="anthropic/claude-3-5-sonnet",
        )

        # Store the first few providers in priority order
        priority_providers = ["zhipu", "minimax", "openai"]

        # Initial state
        assert provider.get_current_provider() == "zhipu"

        # First rotation: zhipu -> minimax
        new_provider = provider.switch_to_next_provider()
        assert new_provider == "minimax"
        assert provider._providers[-1] == "zhipu"  # zhipu moved to end

        # Second rotation: minimax -> openai
        new_provider = provider.switch_to_next_provider()
        assert new_provider == "openai"
        # After 2 rotations, zhipu and minimax should be at the end
        assert provider._providers[-2:] == ["zhipu", "minimax"]

        # Continue rotating until we cycle back to zhipu
        # Keep track of when we see zhipu again
        initial_zhipu_index = priority_providers.index("zhipu")
        for _ in range(len(provider._providers) * 2):  # Try many rotations
            if provider.get_current_provider() == "zhipu" and _ > 0:
                # We've cycled back to zhipu!
                break
            provider.switch_to_next_provider()
        else:
            pytest.fail("Never cycled back to zhipu")

        # zhipu should be current again after cycling through all providers

    def test_rotation_cleans_up_provider_instance(self, mock_config_with_providers):
        """Test that rotation cleans up the old provider to prevent env var contamination."""
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

            # Store reference to the instance
            first_provider_id = id(first_provider)

            # Rotate to next provider
            provider.switch_to_next_provider()

            # The first provider should be removed from cache (cleaned up)
            assert first_provider_name not in provider._provider_instances

            # When we get the provider again, it should be a fresh instance
            new_first_provider = provider._get_provider_instance(first_provider_name)
            # After full rotation cycle, when zhipu comes back to front, it will be re-created
            # So the id will be different from the original

        finally:
            # Restore env state
            os.environ.clear()
            os.environ.update(old_env)

    def test_rotation_with_single_provider_returns_none(self, mock_config_with_providers):
        """Test that rotation returns None when there's only one provider."""
        # Create config with only one provider (no API keys for others)
        config = MagicMock()
        config.agents.defaults.model = "anthropic/claude-3-5-sonnet"

        providers = MagicMock()

        # Configure only zhipu provider
        providers.zhipu.api_key = "zhipu-test-key-123"
        providers.zhipu.api_base = None
        providers.zhipu.model = "zhipu/glm-4"
        providers.zhipu.extra_headers = None

        # All other providers have no API key
        providers.minimax.api_key = None
        providers.openai.api_key = None
        providers.custom.api_key = None
        providers.azure_openai.api_key = None
        providers.openrouter.api_key = None
        providers.aihubmix.api_key = None
        providers.siliconflow.api_key = None
        providers.volcengine.api_key = None
        providers.anthropic.api_key = None
        providers.openai_codex.api_base = None  # OAuth provider needs api_base
        providers.github_copilot.api_key = None
        providers.deepseek.api_key = None
        providers.gemini.api_key = None
        providers.dashscope.api_key = None
        providers.moonshot.api_key = None
        providers.vllm.api_key = None
        providers.groq.api_key = None

        providers.failover = FailoverConfig(
            enabled=True,
            max_retries_per_provider=2,
            provider_priority=["zhipu"],
        )
        config.providers = providers

        provider = FailoverProvider(
            config=config,
            model="anthropic/claude-3-5-sonnet",
        )

        # With only one available provider, rotation should return None
        if len(provider._providers) == 1:
            assert provider.switch_to_next_provider() is None
        else:
            # If other providers are somehow available, skip this test
            pytest.skip("Multiple providers available in environment")

    def test_rotation_indefinite_cycling(self, mock_config_with_providers):
        """Test that we can rotate indefinitely without running out of providers."""
        provider = FailoverProvider(
            config=mock_config_with_providers,
            model="anthropic/claude-3-5-sonnet",
        )

        initial_count = len(provider._providers)

        # Rotate many times (more than the number of providers)
        for i in range(20):
            new_provider = provider.switch_to_next_provider()
            assert new_provider is not None  # Should never be None
            assert len(provider._providers) == initial_count  # List length stays constant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
