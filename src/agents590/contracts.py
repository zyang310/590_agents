"""The small interface shared by standalone agent implementations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from agents590.events import AgentEvent, EventSink


@dataclass(frozen=True)
class RunContext:
    """Explicit working directory, optional per-run model override and progress listener."""

    cwd: Path
    model: str | None = None
    on_event: EventSink | None = field(default=None, compare=False, repr=False)

    def report(self, event: AgentEvent) -> None:
        """Pass a progress event to the caller; agents may report none at all."""
        if self.on_event is not None:
            self.on_event(event)


@runtime_checkable
class Agent(Protocol):
    """Implementations return their final response and may act within `context.cwd`."""

    async def run(self, instruction: str, context: RunContext) -> str: ...
