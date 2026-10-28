"""Draw an agent's progress as it happens, on the CLI's stderr or in the terminal UI's log.

morty · ~/project
› count the python files

● step 1 · 2.3s · 1.2k tokens
  $ find . -name '*.py' | wc -l
  │       12
  ╰ exit 0 · 0.0s

✓ done · 2 steps · 1 command · 2.6k tokens · gpt-5-mini · 3.4s
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.status import Status
from rich.text import Text

from agents590.events import (
    AgentEvent,
    CommandFinished,
    CommandStarted,
    ModelResponded,
    StepStarted,
)

INSTRUCTION_LINES = 3
COMMAND_LINES = 4
OUTPUT_HEAD_LINES = 4
OUTPUT_TAIL_LINES = 2


def _ignore(label: RenderableType | None) -> None:
    pass


class Trace:
    """Event sink that writes one block per model step and per shell command.

    `write` receives each finished block. `wait` receives a live label while the agent waits
    on the model or a command, then None once the wait is over.
    """

    def __init__(
        self,
        write: Callable[[RenderableType], object],
        wait: Callable[[RenderableType | None], object] = _ignore,
    ) -> None:
        self._write = write
        self._wait = wait
        self._started: float | None = None
        self._model = ""
        self._steps = 0
        self._commands = 0
        self._tokens = 0
        self._answered = False

    @classmethod
    def to_console(cls, console: Console) -> Self:
        """Print to `console`; the spinner runs only on an interactive terminal."""
        return cls(console.print, _ConsoleSpinner(console))

    def start(self, agent: str, instruction: str, cwd: Path) -> None:
        """Show which agent runs where, and the start of its instruction."""
        self._started = time.monotonic()
        self._write(Text.assemble((agent, "bold magenta"), (f" · {_home(cwd)}", "dim")))
        shown, hidden, _ = _clip(instruction.strip().splitlines() or [""], INSTRUCTION_LINES, 0)
        text = Text("› ", style="bold")
        text.append("\n  ".join(shown))
        if hidden:
            text.append(f"\n  … +{hidden} lines", style="dim italic")
        self._write(text)

    def __call__(self, event: AgentEvent) -> None:
        self._wait(None)
        match event:
            case StepStarted():
                self._model = event.model
                self._wait(_Ticker(f"thinking · step {event.step}/{event.max_steps}"))
            case ModelResponded():
                self._steps = event.step
                self._tokens += event.input_tokens + event.output_tokens
                self._answered = event.tool_calls == 0
                if not self._answered:  # the caller prints the final answer itself
                    self._print_step(event)
            case CommandStarted():
                self._commands += 1
                self._write(_command(event.command))
                self._wait(_Ticker("running"))
            case CommandFinished():
                self._write(_result(event))

    def finish(self, problem: str | None = None) -> None:
        """Show the outcome and totals; does nothing if the run never started."""
        self._wait(None)
        if self._started is None:
            return
        stats: list[str] = []
        if self._steps:
            stats.append(_count(self._steps, "step"))
        if self._commands:
            stats.append(_count(self._commands, "command"))
        if self._tokens:
            stats.append(_tokens(self._tokens))
        if self._model:
            stats.append(self._model)
        stats.append(_duration(time.monotonic() - self._started))
        if problem is not None:
            mark, color = f"✗ {problem}", "red"
        elif self._steps and not self._answered:
            mark, color = "! stopped without an answer", "yellow"
        else:
            mark, color = "✓ done", "green"
        self._write(Text())
        self._write(Text.assemble((mark, f"bold {color}"), (f" · {' · '.join(stats)}", "dim")))
        self._write(Text())
        self._started = None

    def _print_step(self, event: ModelResponded) -> None:
        stats = [_duration(event.seconds)]
        if event.input_tokens or event.output_tokens:
            stats.append(_tokens(event.input_tokens + event.output_tokens))
        self._write(Text())
        self._write(
            Text.assemble(
                ("● ", "cyan"), (f"step {event.step}", "bold"), (f" · {' · '.join(stats)}", "dim")
            )
        )
        if event.text.strip():
            self._write(Padding(Text(event.text.strip()), (0, 0, 0, 2), expand=False))


class _ConsoleSpinner:
    """Shows the wait label as a transient spinner line below the printed blocks."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._status: Status | None = None

    def __call__(self, label: RenderableType | None) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if label is not None and self._console.is_interactive and not self._console.quiet:
            self._status = self._console.status(label, spinner_style="cyan")
            self._status.start()


class _Ticker:
    """Spinner label that counts the seconds spent waiting."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._since = time.monotonic()

    def __rich__(self) -> Text:
        return Text(f"{self._label} · {time.monotonic() - self._since:.0f}s", style="dim")


def _command(command: str) -> Text:
    shown, hidden, _ = _clip(command.strip().splitlines() or [""], COMMAND_LINES, 0)
    text = Text("  $ ", style="cyan", no_wrap=True, overflow="ellipsis")
    text.append("\n    ".join(shown), style="bold")
    if hidden:
        text.append(f"\n    … +{hidden} lines", style="dim italic")
    return text


def _result(event: CommandFinished) -> Text:
    if event.exit_code is None:
        color, status = "yellow", "timed out, killed"
    else:
        color, status = ("green" if event.exit_code == 0 else "red"), f"exit {event.exit_code}"
    lines = [(line, "dim") for line in event.stdout.splitlines()]
    lines += [(line, "dim red") for line in event.stderr.splitlines()]
    head, hidden, tail = _clip(lines, OUTPUT_HEAD_LINES, OUTPUT_TAIL_LINES)
    text = Text(no_wrap=True, overflow="ellipsis")
    for line, style in head:
        text.append("  │ ", style=color)
        text.append_text(Text.from_ansi(line, style=style))
        text.append("\n")
    if hidden:
        text.append("  │ ", style=color)
        text.append(f"… +{hidden} lines\n", style="dim italic")
    for line, style in tail:
        text.append("  │ ", style=color)
        text.append_text(Text.from_ansi(line, style=style))
        text.append("\n")
    text.append("  ╰ ", style=color)
    text.append(status, style=f"bold {color}")
    text.append(f" · {_duration(event.seconds)}", style="dim")
    return text


def _clip[T](items: list[T], head: int, tail: int) -> tuple[list[T], int, list[T]]:
    """Split items into those shown first, the number hidden, and those shown last."""
    if len(items) <= head + tail + 1:  # hiding a single line saves nothing
        return items, 0, []
    return items[:head], len(items) - head - tail, items[len(items) - tail :]


def _home(path: Path) -> str:
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return "~" if relative == Path() else f"~/{relative}"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m {seconds:02d}s"


def _tokens(count: int) -> str:
    return f"{count / 1000:.1f}k tokens" if count >= 1000 else f"{count} tokens"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
