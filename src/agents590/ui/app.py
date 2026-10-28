"""Textual front end: pick an agent, model and directory, then run an instruction."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from rich.console import RenderableType
from rich.rule import Rule
from rich.spinner import Spinner
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select

from agents590.registry import agent_names
from agents590.trace import Trace
from agents590.ui.session import execute


class Waiting(Widget):
    """One spinner line under the log while the agent waits on the model or a command."""

    DEFAULT_CSS = "Waiting { height: 1; padding: 0 1; display: none; }"

    def __init__(self) -> None:
        super().__init__(id="waiting")
        self._spinner: Spinner | None = None

    def show(self, label: RenderableType | None) -> None:
        """Animate `label` beside a spinner, or hide the line when given None."""
        self._spinner = None if label is None else Spinner("dots", text=label, style="cyan")
        self.display = label is not None
        self.auto_refresh = None if label is None else 1 / 12
        self.refresh()

    def render(self) -> RenderableType:
        return self._spinner or ""


class MortyApp(App[None]):
    """Collects the same inputs as the CLI and shows the agent's progress and answer."""

    TITLE = "agents590"
    BINDINGS = [("ctrl+l", "clear", "Clear output")]
    CSS = """
    #settings { height: 1; padding: 0 1; background: $panel; }
    #settings Label { padding: 0 1; color: $text-muted; }
    #agent { width: 22; }
    #model { width: 1fr; }
    #cwd { width: 2fr; }
    #log { border: round $primary; height: 1fr; }
    #prompt { height: auto; }
    #instruction { width: 1fr; }
    """

    def __init__(self, default_agent: str | None = None, cwd: Path | None = None) -> None:
        super().__init__()
        self._default_agent = default_agent
        self._cwd = cwd or Path.cwd()

    def compose(self) -> ComposeResult:
        names = agent_names()
        default = self._default_agent if self._default_agent in names else None
        if default is None and "morty" in names:
            default = "morty"
        yield Header()
        # one borderless line of run settings above the log; the prompt sits at the bottom
        with Horizontal(id="settings"):
            yield Label("Agent")
            yield Select[str](
                [(name, name) for name in names],
                value=Select.NULL if default is None else default,
                id="agent",
                compact=True,
            )
            yield Label("Model")
            yield Input(placeholder="env default", id="model", compact=True)
            yield Label("Dir")
            yield Input(value=str(self._cwd), id="cwd", compact=True)
        yield RichLog(id="log", wrap=True, markup=False)
        yield Waiting()
        with Horizontal(id="prompt"):
            yield Input(placeholder="Tell the agent what to do (Enter to run)", id="instruction")
            yield Button("Run", id="run", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#instruction", Input).focus()

    def action_clear(self) -> None:
        self.query_one("#log", RichLog).clear()

    @on(Input.Submitted, "#instruction")
    @on(Button.Pressed, "#run")
    def start_run(self) -> None:
        log = self.query_one("#log", RichLog)
        agent: object = self.query_one("#agent", Select).value  # pyright: ignore
        instruction = self.query_one("#instruction", Input).value
        if not isinstance(agent, str):
            log.write("Select an agent first.")
            return
        self.query_one("#run", Button).disabled = True
        self.run_worker(self._run(agent, instruction), exclusive=True)

    async def _run(self, agent: str, instruction: str) -> None:
        log = self.query_one("#log", RichLog)
        # the same blocks the CLI prints to stderr, written into the log as they happen
        trace = Trace(log.write, self.query_one(Waiting).show)
        ok, text = await execute(
            agent,
            instruction,
            self.query_one("#cwd", Input).value,
            self.query_one("#model", Input).value,
            trace,
        )
        log.write(text if ok else f"ERROR: {text}")
        log.write(Rule(style="dim"))
        self.query_one("#run", Button).disabled = False


def main(argv: Sequence[str] | None = None) -> int:
    """Load the environment file, then open the UI."""
    parser = argparse.ArgumentParser(prog="agents590-ui")
    parser.add_argument("agent", nargs="?", help="Agent preselected in the UI")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Initial working directory")
    parser.add_argument("--env-file", type=Path, help="Explicit environment file to load")
    args = parser.parse_args(argv)
    if args.env_file is not None:
        env_file = args.env_file.expanduser().resolve()
        if not env_file.is_file():
            parser.error("--env-file must name an existing file")
        load_dotenv(env_file, override=False)
    MortyApp(default_agent=args.agent, cwd=args.cwd.expanduser().resolve()).run()
    return 0
