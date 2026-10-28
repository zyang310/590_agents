"""Check the progress trace as plain text, the way logs and Harbor's stderr.txt receive it."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from agents590.events import (
    AgentEvent,
    CommandFinished,
    CommandStarted,
    ModelResponded,
    StepStarted,
)
from agents590.trace import Trace


def _render(events: list[AgentEvent], problem: str | None = None) -> str:
    output = io.StringIO()
    trace = Trace.to_console(Console(file=output, width=80, highlight=False))
    trace.start("morty", "count the python files", Path("/work"))
    for event in events:
        trace(event)
    trace.finish(problem)
    return output.getvalue()


def _command(stdout: str = "", stderr: str = "", exit_code: int | None = 0) -> list[AgentEvent]:
    return [
        StepStarted(1, 50, "test-deployment"),
        ModelResponded(1, "Looking around.", 1, 2.34, 1_000, 200),
        CommandStarted("ls"),
        CommandFinished("ls", exit_code, stdout, stderr, 0.25),
    ]


def test_trace_shows_steps_commands_and_totals() -> None:
    events = [
        *_command(stdout="a.py\nb.py\n"),
        StepStarted(2, 50, "test-deployment"),
        ModelResponded(2, "There are two files.", 0, 1.0, 1_500, 100),
    ]
    assert _render(events).splitlines() == [
        "morty · /work",
        "› count the python files",
        "",
        "● step 1 · 2.3s · 1.2k tokens",
        "  Looking around.",
        "  $ ls",
        "  │ a.py",
        "  │ b.py",
        "  ╰ exit 0 · 0.2s",
        "",
        "✓ done · 2 steps · 1 command · 2.8k tokens · test-deployment · 0.0s",
        "",
    ]


def test_final_answer_is_left_to_the_caller() -> None:
    events = [StepStarted(1, 50, "m"), ModelResponded(1, "the final answer", 0, 1.0)]
    output = _render(events)
    assert "the final answer" not in output
    assert "step 1 ·" not in output
    assert "✓ done · 1 step · m" in output


def test_long_output_keeps_its_head_and_tail() -> None:
    stdout = "\n".join(f"line {number}" for number in range(1, 21))
    lines = _render(_command(stdout=stdout)).splitlines()
    shown = [line.removeprefix("  │ ") for line in lines if line.startswith("  │ ")]
    assert shown == ["line 1", "line 2", "line 3", "line 4", "… +14 lines", "line 19", "line 20"]


def test_multiline_commands_are_clipped() -> None:
    command = "cat > notes.txt <<'EOF'\n" + "\n".join(["text"] * 10) + "\nEOF"
    output = _render([StepStarted(1, 50, "m"), CommandStarted(command)])
    assert "  $ cat > notes.txt <<'EOF'\n    text\n    text\n    text\n    … +8 lines\n" in output


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [(0, "╰ exit 0 · 0.2s"), (2, "╰ exit 2 · 0.2s"), (None, "╰ timed out, killed · 0.2s")],
)
def test_command_status_line(exit_code: int | None, expected: str) -> None:
    assert expected in _render(_command(exit_code=exit_code))


def test_stderr_and_ansi_output_render_as_text() -> None:
    output = _render(_command(stdout="\x1b[31mred\x1b[0m", stderr="warning: careful"))
    assert "  │ red\n  │ warning: careful\n" in output
    assert "\x1b" not in output


@pytest.mark.parametrize(
    ("problem", "expected"), [(None, "! stopped without an answer"), ("failed", "✗ failed")]
)
def test_unfinished_runs_say_so(problem: str | None, expected: str) -> None:
    assert f"{expected} · 1 step · 1 command" in _render(_command(), problem)


def test_long_instructions_are_clipped_in_the_header() -> None:
    output = io.StringIO()
    Trace.to_console(Console(file=output, width=60)).start("morty", "a\nb\nc\nd\ne", Path("/work"))
    assert output.getvalue().splitlines()[1:] == ["› a", "  b", "  c", "  … +2 lines"]


def test_finish_without_start_prints_nothing() -> None:
    output = io.StringIO()
    Trace.to_console(Console(file=output)).finish("failed")
    assert output.getvalue() == ""
