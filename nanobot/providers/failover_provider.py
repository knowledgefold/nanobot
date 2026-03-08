"""Provider wrapper with automatic failover and retry logic."""

import asyncio
from typing import Any

from httpx import ConnectError
from litellm import AuthenticationError, BudgetExceededError, APIError
from loguru import logger

from nanobot.config.schema import Config, FailoverConfig
from nanobot.providers.base import LLMProvider, LLMResponse


class FailoverProvider(LLMProvider):
    """Provider wrapper with automatic failover and retry logic.

    When a provider fails, this class automatically:
    1. Retries temporary errors (timeouts, rate limits, server errors) with exponential backoff
    2. Switches to the next available provider for fatal errors (auth failures, invalid model)
    3. Optionally maps model names for different providers

    Example:
        provider = FailoverProvider(
            config=config,
            model="anthropic/claude-3-5-sonnet",
            failover_config=failover_config,
        )
        response = await provider.chat(messages)
    """

    # Error types that should trigger immediate failover (not retry)
    FATAL_ERROR_TYPES = (
        AuthenticationError,
        # BudgetExceededError,
        # PermissionError,
    )

    # HTTP status codes that should trigger immediate failover
    FATAL_STATUS_CODES = {401, 403, 400, 404}

    def __init__(
        self,
        config: Config,
        model: str | None = None,
        failover_config: FailoverConfig | None = None,
    ):
        """Initialize the failover provider.

        Args:
            config: Nanobot configuration object
            model: Default model to use
            failover_config: Failover configuration (uses config.providers.failover if None)
        """
        super().__init__(None, None)
        self.config = config
        self.default_model = model or config.agents.defaults.model
        self.failover_config = failover_config or config.providers.failover

        # Build provider list based on priority and availability
        self._providers = self._build_provider_list()
        self._provider_instances: dict[str, LLMProvider] = {}

    def _build_provider_list(self) -> list[str]:
        """Build ordered list of available provider names."""
        providers = []

        # Start with configured priority (filter to available providers)
        for name in self.failover_config.provider_priority:
            if self._is_provider_available(name):
                providers.append(name)

        # Add any other available providers not in priority list
        from nanobot.providers.registry import PROVIDERS
        for spec in PROVIDERS:
            if spec.name not in providers and self._is_provider_available(spec.name):
                providers.append(spec.name)

        logger.debug(f"Failover provider list: {providers}")
        return providers

    def _is_provider_available(self, provider_name: str) -> bool:
        """Check if provider has API key configured (OAuth providers exempt)."""
        from nanobot.providers.registry import find_by_name

        spec = find_by_name(provider_name)
        if not spec:
            return False

        # OAuth providers are always available (they use interactive login)
        if spec.is_oauth:
            return True

        # Direct providers (custom, azure_openai) require explicit config
        if spec.is_direct:
            provider_config = getattr(self.config.providers, provider_name, None)
            return provider_config is not None and bool(provider_config.api_key)

        # Standard providers need API key
        provider_config = getattr(self.config.providers, provider_name, None)
        return provider_config is not None and bool(provider_config.api_key)

    def _classify_error(self, error: Exception) -> str:
        """Classify error as 'retryable', 'fatal', or 'unknown'.

        Args:
            error: The exception to classify

        Returns:
            'retryable', 'fatal', or 'unknown'
        """
        # Network/timeout errors are retryable
        if isinstance(error, (TimeoutError, ConnectError)):
            return "retryable"

        # LiteLLM API errors - check status code
        if isinstance(error, APIError):
            if hasattr(error, "status_code"):
                status = error.status_code
                if status in self.FATAL_STATUS_CODES:
                    return "fatal"
                if status == 429 or status >= 500:
                    return "retryable"

        # Authentication errors are fatal
        if isinstance(error, AuthenticationError):
            return "fatal"

        # Default to retryable for unknown errors (transient issues)
        return "retryable"

    async def _try_provider_with_retry(
        self,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> LLMResponse | None:
        """Try a provider with exponential backoff retry.

        Returns None if all retries are exhausted with fatal errors.
        """
        max_retries = self.failover_config.max_retries_per_provider
        backoff = self.failover_config.initial_backoff_seconds
        max_backoff = self.failover_config.max_backoff_seconds
        multiplier = self.failover_config.backoff_multiplier

        for attempt in range(max_retries + 1):
            try:
                return await provider.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                error_type = self._classify_error(e)

                if error_type == "fatal":
                    logger.warning(f"Fatal error with provider: {e}")
                    return None  # Signal to switch provider

                if attempt < max_retries:
                    # Retry with exponential backoff
                    sleep_time = min(backoff, max_backoff)
                    logger.info(
                        f"Retryable error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {sleep_time:.1f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    backoff *= multiplier
                else:
                    logger.error(f"All retries exhausted: {e}")
                    return None

        return None

    def _map_model_for_provider(self, provider_name: str, model: str) -> str:
        """Map model name for a specific provider.

        Args:
            provider_name: Name of the target provider
            model: Original model name

        Returns:
            Mapped model name, or original if no mapping exists
        """
        mapping = self.failover_config.model_mapping.get(provider_name, {})
        return mapping.get(model, model)

    def _get_provider_instance(self, provider_name: str) -> LLMProvider | None:
        """Get or create provider instance.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance or None if provider cannot be created
        """
        # Return cached instance if available
        if provider_name in self._provider_instances:
            return self._provider_instances[provider_name]

        # Create new instance
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
        from nanobot.providers.custom_provider import CustomProvider
        from nanobot.providers.litellm_provider import LiteLLMProvider
        from nanobot.providers.registry import find_by_name

        spec = find_by_name(provider_name)
        if not spec:
            logger.error(f"Provider spec not found: {provider_name}")
            return None

        provider_config = getattr(self.config.providers, provider_name, None)
        if not provider_config:
            logger.error(f"Provider config not found: {provider_name}")
            return None

        try:
            # OpenAI Codex (OAuth)
            if provider_name == "openai_codex":
                instance = OpenAICodexProvider(default_model=self.default_model)

            # Custom: direct OpenAI-compatible endpoint
            elif provider_name == "custom":
                # For custom provider, use provider's own api_base or default
                api_base = provider_config.api_base or "http://localhost:8000/v1"
                instance = CustomProvider(
                    api_key=provider_config.api_key or "no-key",
                    api_base=api_base,
                    default_model=self.default_model,
                )

            # Azure OpenAI
            elif provider_name == "azure_openai":
                instance = AzureOpenAIProvider(
                    api_key=provider_config.api_key,
                    api_base=provider_config.api_base,
                    default_model=self.default_model,
                )

            # Standard LiteLLM providers
            else:
                # Get api_base from provider config or spec default
                api_base = provider_config.api_base or (spec.default_api_base if spec else None)
                instance = LiteLLMProvider(
                    api_key=provider_config.api_key,
                    api_base=api_base,
                    default_model=self.default_model,
                    extra_headers=provider_config.extra_headers,
                    provider_name=provider_name,
                )

            self._provider_instances[provider_name] = instance
            return instance

        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {e}")
            return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Execute chat with automatic failover.

        Tries each provider in priority order with retry logic.
        Returns first successful response or error message.
        """
        original_model = model or self.default_model

        # If failover is disabled, use primary provider only
        if not self.failover_config.enabled:
            primary_name = self._providers[0] if self._providers else None
            if not primary_name:
                return LLMResponse(
                    content="Error: No available LLM provider configured",
                    finish_reason="error",
                )

            provider = self._get_provider_instance(primary_name)
            if not provider:
                return LLMResponse(
                    content=f"Error: Failed to initialize provider '{primary_name}'",
                    finish_reason="error",
                )

            try:
                return await provider.chat(
                    messages=messages,
                    tools=tools,
                    model=original_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                return LLMResponse(
                    content=f"Error calling LLM: {str(e)}",
                    finish_reason="error",
                )

        # Try each provider in order
        last_error = None
        for provider_name in self._providers:
            provider = self._get_provider_instance(provider_name)
            if not provider:
                continue

            # Map model name for this provider
            mapped_model = self._map_model_for_provider(provider_name, original_model)

            logger.info(f"Attempting provider: {provider_name} with model: {mapped_model}")

            result = await self._try_provider_with_retry(
                provider=provider,
                messages=messages,
                tools=tools,
                model=mapped_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

            if result is not None:
                # Log successful failover if not primary provider
                if provider_name != self._providers[0]:
                    logger.info(f"Failover successful: using {provider_name}")
                return result

            # Store last error for final message
            last_error = f"{provider_name}: failed after retries"

        # All providers failed
        error_msg = f"Error: All LLM providers failed. "
        if last_error:
            error_msg += f"Last: {last_error}"
        return LLMResponse(content=error_msg, finish_reason="error")

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
