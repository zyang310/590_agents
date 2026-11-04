"""The skill tool: load one skill's full instructions into the conversation."""

from collections.abc import Sequence
from typing import Any

from agents590.contracts import RunContext
from agents590.events import ToolFinished, ToolStarted
from agents590.morty.skills import Skill, format_activation
from agents590.morty.tools.base import Tool, ToolError, count_lines


def skill_tool(skills: Sequence[Skill]) -> Tool:
    """Build a tool for one run that can load `skills`, each of them once."""
    by_name = {skill.name: skill for skill in skills}
    loaded: set[str] = set()

    async def run(arguments: dict[str, Any], context: RunContext) -> str:
        name: str = arguments["name"]
        context.report(ToolStarted("skill", name))
        skill = by_name.get(name)
        if skill is None:  # strict mode's enum should prevent this, but the model isn't trusted
            problem = f"unknown skill {name!r}; choose one of {', '.join(by_name)}."
            context.report(ToolFinished("skill", name, problem, True, 0.0))
            raise ToolError(problem)
        if name in loaded:  # these results are never masked, so repeats would pile up
            context.report(ToolFinished("skill", name, "already loaded", False, 0.0))
            return f"Skill {name!r} is already loaded earlier in this conversation; follow it."
        loaded.add(name)
        summary = count_lines(len(skill.body.splitlines()))
        context.report(ToolFinished("skill", name, summary, False, 0.0))
        return format_activation(skill)

    return Tool(
        {
            "type": "function",
            "name": "skill",
            "description": (
                "Load a skill's full instructions into the conversation. Call this when the task "
                "matches a skill listed in the system prompt, before starting that work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": sorted(by_name),
                        "description": "The skill to load.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        run,
        keep_result=True,
    )
