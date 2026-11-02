"""Morty's tools. To add one, write a module that builds a `Tool` and list it in TOOLS."""

import json

from agents590.contracts import RunContext
from agents590.morty.tools.base import Tool, ToolError
from agents590.morty.tools.bash import BASH
from agents590.morty.tools.edit import EDIT
from agents590.morty.tools.read import READ
from agents590.morty.tools.write import WRITE

# Every request sends these, so keep their order fixed or the prompt cache stops matching.
TOOLS: list[Tool] = [BASH, READ, WRITE, EDIT]
TOOL_DEFINITIONS = [tool.definition for tool in TOOLS]
_TOOLS_BY_NAME = {tool.definition["name"]: tool for tool in TOOLS}


async def run_tool(name: str, raw_arguments: str, context: RunContext) -> str:
    """Run one call from the model; failures it can recover from come back as text."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}."
    try:
        return await tool.run(json.loads(raw_arguments), context)
    except (ToolError, OSError) as error:
        return f"Error: {error}"
