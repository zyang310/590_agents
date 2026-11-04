"""What the model reads about skills: a catalog up front, and a skill's instructions on demand."""

import re
from collections.abc import Sequence
from html import escape

from agents590.morty.skills.parser import Skill

# About the 5,000 tokens the spec recommends for a skill's instructions, which are never masked.
MAX_SKILL_CHARS = 20_000

_CATALOG_INTRO = (
    "The following skills provide specialized instructions for specific tasks. When a task "
    "matches a skill's description, call the skill tool with its name to load its full "
    "instructions before starting that work."
)
_COMMAND = re.compile(r"/skill:(\S+)\s*(.*)", re.DOTALL)


def format_catalog(skills: Sequence[Skill]) -> str:
    """List each skill's name and description; the tool result supplies its location later."""
    entries = [
        "  <skill>\n"
        f"    <name>{escape(skill.name, quote=False)}</name>\n"
        f"    <description>{escape(skill.description, quote=False)}</description>\n"
        "  </skill>"
        for skill in skills
    ]
    return "\n".join([_CATALOG_INTRO, "", "<available_skills>", *entries, "</available_skills>"])


def format_activation(skill: Skill) -> str:
    """Wrap a skill's instructions with where they live, cutting any that are far too long."""
    body = skill.body
    if len(body) > MAX_SKILL_CHARS:
        cut = (
            f"[instructions cut at {MAX_SKILL_CHARS:,} characters; read {skill.path} for the rest]"
        )
        body = f"{body[:MAX_SKILL_CHARS]}\n{cut}"
    return (
        f'<skill_content name="{escape(skill.name)}">\n'
        f"{body}\n\n"
        f"Skill directory: {skill.directory}\n"
        "Relative paths in this skill start from that directory. Commands still start in the "
        "working directory, so use absolute paths or cd there first.\n"
        "</skill_content>"
    )


def expand_command(instruction: str, skills: Sequence[Skill]) -> str:
    """Put a skill's instructions in place of a leading `/skill:name`, keeping the rest."""
    match = _COMMAND.fullmatch(instruction.strip())
    if match is None:
        return instruction
    skill = next((skill for skill in skills if skill.name == match[1]), None)
    if skill is None:
        available = ", ".join(skill.name for skill in skills) or "none"
        raise ValueError(f"unknown skill {match[1]!r}; available: {available}")
    activation = format_activation(skill)
    return f"{activation}\n\n{match[2]}" if match[2] else activation
