"""Morty, an agent independent of Harbor."""

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings


class Morty:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        # Start with settings = self.provider_settings(context), then open
        # settings.create_async_client() and use model=settings.model.
        # Send Responses requests with store=False and background=False.
        # Run shell tools with context.cwd as their explicit working directory.
        raise NotImplementedError("Implement the UNC Azure model/tool loop in MortyAgent.run().")
