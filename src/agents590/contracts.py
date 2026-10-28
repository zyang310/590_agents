"""The small interface shared by standalone agent implementations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RunContext:
    """Explicit working directory and optional per-run model override."""

    cwd: Path
    model: str | None = None


@runtime_checkable
class Agent(Protocol):
    """Implementations return their final response and may act within `context.cwd`."""

    async def run(self, instruction: str, context: RunContext) -> str: ...
