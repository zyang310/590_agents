"""Exercise the terminal UI against the echo agent; no model requests are made."""

import asyncio
from pathlib import Path

import pytest
from textual.geometry import Region
from textual.widgets import Input, RichLog

from agents590.contracts import RunContext
from agents590.events import CommandFinished, CommandStarted, ModelResponded, StepStarted
from agents590.ui import session
from agents590.ui.app import MortyApp, Waiting
from agents590.ui.session import execute


@pytest.mark.parametrize(
    ("instruction", "cwd", "expected"),
    [("  ", ".", "Enter an instruction"), ("hi", "/definitely/not/a/dir", "does not exist")],
)
def test_execute_rejects_bad_input(instruction: str, cwd: str, expected: str) -> None:
    ok, text = asyncio.run(execute("krisj_echo", instruction, cwd, ""))
    assert not ok
    assert expected in text


def test_execute_reports_unknown_agent_instead_of_raising(tmp_path: Path) -> None:
    ok, text = asyncio.run(execute("nope", "hi", str(tmp_path), ""))
    assert not ok
    assert "Unknown agent" in text


def test_execute_returns_agent_output(tmp_path: Path) -> None:
    assert asyncio.run(execute("krisj_echo", "hello", str(tmp_path), "")) == (True, "hello")


def test_app_runs_instruction_and_shows_output(tmp_path: Path) -> None:
    async def scenario() -> list[str]:
        app = MortyApp(default_agent="krisj_echo", cwd=tmp_path)
        async with app.run_test() as pilot:
            app.query_one("#instruction", Input).value = "ping"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            return [line.text for line in app.query_one("#log", RichLog).lines]

    lines = asyncio.run(scenario())
    assert any("ping" in line for line in lines)
    assert any(line.strip() == "ping" for line in lines)


class SteppingAgent:
    """Runs one command, then waits for the test before giving its final answer."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def run(self, instruction: str, context: RunContext) -> str:
        context.report(StepStarted(1, 50, "test-deployment"))
        context.report(ModelResponded(1, "", 1, 1.0, 900, 100))
        context.report(CommandStarted("ls"))
        context.report(CommandFinished("ls", 0, "a.py\n", "", 0.1))
        context.report(StepStarted(2, 50, "test-deployment"))
        await self.release.wait()
        context.report(ModelResponded(2, "one file", 0, 1.0))
        return "one file"


def test_app_shows_the_live_trace_and_a_waiting_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = SteppingAgent()
    monkeypatch.setattr(session, "create_agent", lambda name: agent)

    async def scenario() -> tuple[bool, bool, list[str]]:
        app = MortyApp(default_agent="krisj_echo", cwd=tmp_path)
        async with app.run_test() as pilot:
            app.query_one("#instruction", Input).value = "count files"
            await pilot.press("enter")
            await pilot.pause()
            waiting_during = app.query_one(Waiting).display
            agent.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            lines = [line.text.rstrip() for line in app.query_one("#log", RichLog).lines]
            return waiting_during, app.query_one(Waiting).display, lines

    waiting_during, waiting_after, lines = asyncio.run(scenario())
    assert waiting_during
    assert not waiting_after
    assert "› count files" in lines
    step = lines.index("● step 1 · 1.0s · 1.0k tokens")
    assert lines[step + 1 : step + 4] == ["  $ ls", "  │ a.py", "  ╰ exit 0 · 0.1s"]
    assert any(line.startswith("✓ done · 2 steps · 1 command · 1.0k tokens") for line in lines)
    assert lines.count("one file") == 1


def test_settings_share_one_line_and_the_prompt_sits_below_the_log(tmp_path: Path) -> None:
    names = ["agent", "model", "cwd", "log", "instruction", "run"]

    async def scenario() -> dict[str, Region]:
        app = MortyApp(cwd=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            return {name: app.query_one(f"#{name}").region for name in names}

    regions = asyncio.run(scenario())
    settings = [regions["agent"], regions["model"], regions["cwd"]]
    assert {(region.y, region.height) for region in settings} == {(regions["agent"].y, 1)}
    assert regions["agent"].width >= 20
    assert regions["instruction"].y >= regions["log"].bottom
    assert all(region.right <= 100 for region in regions.values())
