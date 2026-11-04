"""Morty's system prompt; advice about a single tool belongs in that tool's description."""

from collections.abc import Sequence

from agents590.morty.skills import Skill, format_catalog

# Keep this text identical on every run so the model's prompt cache keeps matching.
SYSTEM_PROMPT = (
    "You are Morty, a careful coding agent. Your context is limited, so read only what you need: "
    "locate things with rg (or grep -rn if rg is missing), then read just the relevant lines. "
    "Use the read, edit and write tools for files rather than cat, sed or heredocs, and use bash "
    "to search and to run things."
)


def build_instructions(skills: Sequence[Skill]) -> str:
    """The system prompt for a run that has `skills`; with none, it's exactly SYSTEM_PROMPT."""
    return f"{SYSTEM_PROMPT}\n\n{format_catalog(skills)}" if skills else SYSTEM_PROMPT
