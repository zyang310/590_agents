"""Agent Skills (https://agentskills.io/specification) for Morty.

This package imports nothing from the rest of Morty: the prompt and tools depend on it, and the
loop only ever sees the instructions and tools they build.
"""

from agents590.morty.skills.disclosure import expand_command, format_activation, format_catalog
from agents590.morty.skills.discovery import Discovery, discover_skills
from agents590.morty.skills.parser import Skill, SkillError, parse_skill

__all__ = [
    "Discovery",
    "Skill",
    "SkillError",
    "discover_skills",
    "expand_command",
    "format_activation",
    "format_catalog",
    "parse_skill",
]
