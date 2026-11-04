"""What Morty's tools share: their shape, expected failures, and reporting for file tools."""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai.types.responses import FunctionToolParam

from agents590.contracts import RunContext
from agents590.events import ToolFinished, ToolStarted


class ToolError(Exception):
    """An expected failure the model can act on: it reads the message and the run carries on."""


@dataclass(frozen=True)
class Tool:
    """A tool's schema for the model, and the code that runs a call to it.

    `keep_result` exempts the tool's results from masking, for text the model needs all run.
    """

    definition: FunctionToolParam
    run: Callable[[dict[str, Any], RunContext], Awaitable[str]]
    keep_result: bool = False


@dataclass(frozen=True)
class FileResult:
    """What a file tool produced: `text` goes to the model and `summary` to the trace."""

    text: str
    summary: str


def file_tool(
    definition: FunctionToolParam, work: Callable[[Path, dict[str, Any]], FileResult]
) -> Tool:
    """Build a tool from synchronous work on the `path` argument, resolved against the cwd.

    Absolute paths stay as they are: bash isn't confined to the cwd either.
    """
    name = definition["name"]

    async def run(arguments: dict[str, Any], context: RunContext) -> str:
        detail: str = arguments["path"]
        context.report(ToolStarted(name, detail))
        started = time.monotonic()
        try:
            result = work(context.cwd / detail, arguments)
        except (ToolError, OSError) as error:
            seconds = time.monotonic() - started
            context.report(ToolFinished(name, detail, str(error), True, seconds))
            raise
        seconds = time.monotonic() - started
        context.report(ToolFinished(name, detail, result.summary, False, seconds))
        return result.text

    return Tool(definition, run)


def count_lines(count: int) -> str:
    return "1 line" if count == 1 else f"{count} lines"
