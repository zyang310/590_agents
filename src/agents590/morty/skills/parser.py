"""Read one SKILL.md leniently: cosmetic problems warn, and only an unusable file is rejected."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

MAX_SKILL_BYTES = 1_048_576
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1_024

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)^---[ \t]*$\n?(.*)\Z", re.DOTALL | re.MULTILINE)
_VALID_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# `description: Use when: ...` isn't valid YAML, but skills written for other clients may say it.
_PLAIN_VALUE = re.compile(r"^(name|description):[ \t]+(?!['\"|>])(.*\S)[ \t]*$", re.MULTILINE)


class SkillError(Exception):
    """A SKILL.md that can't be used; the message says why."""


@dataclass(frozen=True)
class Skill:
    """One skill: its catalog entry, where it lives, and its instructions."""

    name: str
    description: str
    path: Path
    body: str
    model_invocable: bool = True

    @property
    def directory(self) -> Path:
        """Where relative paths in the instructions start from."""
        return self.path.parent


def _load_yaml(block: str) -> object:
    # BaseLoader keeps every scalar a string: `name: on` isn't a boolean and a bad date can't raise.
    return yaml.load(block, Loader=yaml.BaseLoader)


def _quote_plain_value(match: re.Match[str]) -> str:
    # A JSON string is also a YAML one; escaping non-ASCII would split emoji into bad surrogates.
    return f"{match[1]}: {json.dumps(match[2], ensure_ascii=False)}"


def _frontmatter_fields(block: str) -> dict[str, object]:
    try:
        try:
            fields = _load_yaml(block)
        except yaml.YAMLError:
            fields = _load_yaml(_PLAIN_VALUE.sub(_quote_plain_value, block))
    except (yaml.YAMLError, RecursionError) as error:
        raise SkillError(f"frontmatter isn't valid YAML: {' '.join(str(error).split())}") from error
    if not isinstance(fields, dict):
        raise SkillError("frontmatter isn't a YAML mapping")
    return cast(dict[str, object], fields)


def _clean_name(value: object, directory_name: str, warnings: list[str]) -> str:
    name = " ".join(value.split()) if isinstance(value, str) else ""
    if not name:
        warnings.append(f"name is missing, so using the directory name {directory_name!r}")
        name = directory_name
    if len(name) > MAX_NAME_CHARS:
        warnings.append(f"name is over {MAX_NAME_CHARS} characters, so it was cut")
        name = name[:MAX_NAME_CHARS]
    if not _VALID_NAME.fullmatch(name):
        warnings.append(f"name {name!r} should be lowercase letters and digits joined by hyphens")
    if name != directory_name:
        warnings.append(f"name {name!r} doesn't match its directory {directory_name!r}")
    return name


def _clean_description(value: object, warnings: list[str]) -> str:
    description = " ".join(value.split()) if isinstance(value, str) else ""
    if not description:
        raise SkillError("description is required")
    if len(description) > MAX_DESCRIPTION_CHARS:
        warnings.append(f"description is over {MAX_DESCRIPTION_CHARS:,} characters, so it was cut")
        description = description[:MAX_DESCRIPTION_CHARS]
    return description


def parse_skill(path: Path) -> tuple[Skill, list[str]]:
    """Return the skill in `path` and any warnings, or raise SkillError if it can't be used."""
    try:
        if path.stat().st_size > MAX_SKILL_BYTES:
            raise SkillError(f"file is over {MAX_SKILL_BYTES:,} bytes")
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        raise SkillError(str(error)) from error
    match = _FRONTMATTER.match(text)
    if match is None:
        raise SkillError("file doesn't start with YAML frontmatter between --- lines")

    fields = _frontmatter_fields(match[1])
    warnings: list[str] = []
    # Not in the spec, but pi and Claude Code honor it: the author kept this skill for users.
    user_only = fields.get("disable-model-invocation")
    skill = Skill(
        name=_clean_name(fields.get("name"), path.parent.name, warnings),
        description=_clean_description(fields.get("description"), warnings),
        path=path,
        body=match[2].strip(),
        model_invocable=not (isinstance(user_only, str) and user_only.lower() == "true"),
    )
    return skill, warnings
