"""Run one agent instruction for the UI, reporting failures as text instead of raising."""

import asyncio
from pathlib import Path

from rich.console import RenderableType

from agents590.contracts import RunContext
from agents590.registry import create_agent
from agents590.trace import Trace


def _discard(block: RenderableType) -> None:
    pass


async def execute(
    agent_name: str, instruction: str, cwd: str, model: str, trace: Trace | None = None
) -> tuple[bool, str]:
    """Return (succeeded, text) for one run; never raises for ordinary agent failures.

    Like the CLI, `trace` follows the run from the moment the agent loads until it ends.
    """
    if not instruction.strip():
        return False, "Enter an instruction first."
    directory = Path(cwd).expanduser().resolve()
    if not directory.is_dir():
        return False, f"Working directory does not exist: {directory}"
    trace = trace or Trace(_discard)
    try:
        agent = create_agent(agent_name)
        trace.start(agent_name, instruction, directory)
        context = RunContext(cwd=directory, model=model.strip() or None, on_event=trace)
        output = await agent.run(instruction, context)
    except asyncio.CancelledError:  # a newer run replaced this one
        trace.finish("cancelled")
        raise
    except Exception as error:  # the UI must survive any agent or API failure
        trace.finish("failed")
        return False, f"{type(error).__name__}: {error}"
    trace.finish()
    return True, output
