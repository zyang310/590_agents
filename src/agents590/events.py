"""Progress events an agent may report while it works; the caller decides how to show them."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StepStarted:
    """The agent is asking the model for its next move."""

    step: int
    max_steps: int
    model: str


@dataclass(frozen=True)
class ModelResponded:
    """The model replied; with no tool calls, `text` is the agent's final answer."""

    step: int
    text: str
    tool_calls: int
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class CommandStarted:
    """A shell command is about to run in the agent's working directory."""

    command: str


@dataclass(frozen=True)
class CommandFinished:
    """A shell command ended; `exit_code` is None when it was killed for running too long."""

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float


type AgentEvent = StepStarted | ModelResponded | CommandStarted | CommandFinished
type EventSink = Callable[[AgentEvent], None]
