"""Find the skills a run can use: where to look, in what order, and which one wins a name."""

import os
from dataclasses import dataclass
from pathlib import Path

from agents590.morty.skills.parser import Skill, SkillError, parse_skill

SKILLS_DIR = Path(".agents", "skills")
SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class Discovery:
    """Every usable skill, sorted by name so the prompt stays stable, and the problems found."""

    skills: tuple[Skill, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def for_model(self) -> tuple[Skill, ...]:
        """The skills the model may load itself; the rest only answer to `/skill:name`."""
        return tuple(skill for skill in self.skills if skill.model_invocable)


def _project_dirs(cwd: Path) -> list[Path]:
    """The cwd and, inside a git repository, each parent up to its root, nearest first.

    Outside a repository only the cwd counts, so a stray /tmp/.agents/skills can't add skills.
    """
    chain = [cwd, *cwd.parents]
    for depth, directory in enumerate(chain):
        if os.path.exists(directory / ".git"):  # a file in a worktree or submodule
            return chain[: depth + 1]
    return [cwd]


def _skill_files(root: Path, warnings: list[str]) -> list[Path]:
    """The SKILL.md of each direct subdirectory of `root`, which usually doesn't exist at all."""
    try:
        entries = sorted(root.iterdir())
    except FileNotFoundError:
        return []
    except OSError as error:
        warnings.append(f"couldn't scan {root}: {error}")
        return []
    # os.path.isfile follows the symlinks installers often use, and never raises on a locked entry.
    return [entry / SKILL_FILE for entry in entries if os.path.isfile(entry / SKILL_FILE)]


def discover_skills(
    cwd: Path, extra_dir: Path | None = None, home: Path | None = None
) -> Discovery:
    """Scan `extra_dir`, then the project's directories, then the user's; the first name wins."""
    roots = [directory / SKILLS_DIR for directory in _project_dirs(cwd.resolve())]
    # Unlike Path.home(), this doesn't raise for a container user who has no home directory.
    roots.append((home or Path(os.path.expanduser("~"))) / SKILLS_DIR)
    if extra_dir is not None:
        roots.insert(0, extra_dir)

    found: dict[str, Skill] = {}
    seen: set[Path] = set()
    warnings: list[str] = []
    for root in roots:
        for path in _skill_files(root, warnings):
            real_path = path.resolve()
            if real_path in seen:  # a repeated root, or a skill linked into a second one
                continue
            seen.add(real_path)
            try:
                skill, problems = parse_skill(path)
            except SkillError as error:
                warnings.append(f"skipped {path}: {error}")
                continue
            warnings += [f"{path}: {problem}" for problem in problems]
            winner = found.setdefault(skill.name, skill)
            if winner is not skill:
                warnings.append(f"skipped {path}: {winner.path} already uses {skill.name!r}")
    skills = sorted(found.values(), key=lambda skill: skill.name)
    return Discovery(tuple(skills), tuple(warnings))
