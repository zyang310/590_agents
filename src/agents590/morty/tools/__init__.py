"""Morty's tools. To add one, write a module that builds a `Tool` and list it in TOOLS."""

import json
from collections.abc import Sequence

from agents590.contracts import RunContext
from agents590.morty.skills import Skill
from agents590.morty.tools.base import Tool, ToolError
from agents590.morty.tools.bash import BASH
from agents590.morty.tools.edit import EDIT
from agents590.morty.tools.read import READ
from agents590.morty.tools.skill import skill_tool
from agents590.morty.tools.write import WRITE

# Every request sends these, so keep their order fixed or the prompt cache stops matching.
TOOLS: list[Tool] = [BASH, READ, WRITE, EDIT]


class Toolset:
    """The tools one run offers the model: their definitions, and how to run a call to them."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.definitions = [tool.definition for tool in tools]
        self._by_name = {tool.definition["name"]: tool for tool in tools}

    async def run(self, name: str, raw_arguments: str, context: RunContext) -> str:
        """Run one call from the model; failures it can recover from come back as text."""
        tool = self._by_name.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}."
        try:
            return await tool.run(json.loads(raw_arguments), context)
        except (ToolError, OSError) as error:
            return f"Error: {error}"

    def keeps_result(self, name: str) -> bool:
        """Whether results of this tool must stay in full for the rest of the run."""
        tool = self._by_name.get(name)
        return tool is not None and tool.keep_result


def tools_for(skills: Sequence[Skill]) -> Toolset:
    """The tools for a run that has `skills`; with none, it offers exactly the usual ones."""
    return Toolset([*TOOLS, skill_tool(skills)] if skills else TOOLS)
