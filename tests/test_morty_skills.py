"""Check how Morty finds, reads and presents skills, and the tool that loads them."""

import asyncio
import json
import pwd
from pathlib import Path
from typing import Any

import pytest

from agents590.contracts import RunContext
from agents590.events import AgentEvent, ToolFinished, ToolStarted
from agents590.morty.skills import (
    Discovery,
    Skill,
    SkillError,
    discover_skills,
    expand_command,
    format_activation,
    format_catalog,
    parse_skill,
)
from agents590.morty.skills.disclosure import MAX_SKILL_CHARS
from agents590.morty.skills.discovery import SKILLS_DIR
from agents590.morty.skills.parser import MAX_DESCRIPTION_CHARS, MAX_NAME_CHARS, MAX_SKILL_BYTES
from agents590.morty.tools import TOOLS, Toolset, tools_for
from agents590.morty.tools.skill import skill_tool


def _skill_file(root: Path, directory: str, text: str) -> Path:
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def _add_skill(root: Path, name: str, description: str = "Use for demos.") -> Path:
    return _skill_file(root, name, f"---\nname: {name}\ndescription: {description}\n---\nDo it.\n")


def _skill(
    name: str = "demo",
    description: str = "Use for demos.",
    body: str = "Do it.",
    model_invocable: bool = True,
) -> Skill:
    return Skill(name, description, Path("/skills") / name / "SKILL.md", body, model_invocable)


def _names(found: Discovery) -> list[str]:
    return [skill.name for skill in found.skills]


def _paths(found: Discovery) -> list[Path]:
    return [skill.path for skill in found.skills]


def _load(tools: Toolset, name: str, context: RunContext) -> str:
    return asyncio.run(tools.run("skill", json.dumps({"name": name}), context))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    return root


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("# Just markdown\n", id="no frontmatter"),
        pytest.param("---\nname: demo\ndescription: Use it.\n", id="unclosed frontmatter"),
        pytest.param("---\n---\nBody\n", id="empty frontmatter"),
        pytest.param("---\nname: demo\n---\nBody\n", id="missing description"),
        pytest.param('---\nname: demo\ndescription: "  "\n---\nBody\n', id="blank description"),
        pytest.param("---\nname: demo\ndescription: [a, b]\n---\nBody\n", id="list description"),
        pytest.param("---\ndescription: Use it.\nother: [unclosed\n---\nBody\n", id="bad yaml"),
        pytest.param("---\n- a\n---\nBody\n", id="not a mapping"),
        pytest.param(f"---\n{'[' * 100_000}\n---\nBody\n", id="deep nesting"),
    ],
)
def test_parser_rejects_a_file_it_cannot_use(tmp_path: Path, text: str) -> None:
    with pytest.raises(SkillError):
        parse_skill(_skill_file(tmp_path, "demo", text))


def test_parser_rejects_a_file_over_the_size_limit(tmp_path: Path) -> None:
    text = f"---\nname: demo\ndescription: Use it.\n---\n{'x' * MAX_SKILL_BYTES}"
    with pytest.raises(SkillError, match="bytes"):
        parse_skill(_skill_file(tmp_path, "demo", text))


def test_parser_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SkillError):
        parse_skill(tmp_path / "demo" / "SKILL.md")


def test_a_yaml_error_stays_on_one_line_for_the_trace(tmp_path: Path) -> None:
    path = _skill_file(tmp_path, "demo", "---\ndescription: Use it.\nother: [unclosed\n---\n")
    with pytest.raises(SkillError) as error:
        parse_skill(path)
    assert "\n" not in str(error.value)


@pytest.mark.parametrize(
    ("directory", "name_field", "expected", "warning"),
    [
        ("other", "name: demo\n", "demo", "doesn't match its directory"),
        ("Demo_Skill", "name: Demo_Skill\n", "Demo_Skill", "lowercase"),
        ("a--b", "name: a--b\n", "a--b", "lowercase"),
        ("-demo", "name: -demo\n", "-demo", "lowercase"),
        ("demo", "", "demo", "name is missing"),
        ("demo", "name: [a]\n", "demo", "name is missing"),
    ],
)
def test_parser_warns_about_a_bad_name_and_still_loads_the_skill(
    tmp_path: Path, directory: str, name_field: str, expected: str, warning: str
) -> None:
    path = _skill_file(tmp_path, directory, f"---\n{name_field}description: Use it.\n---\nBody\n")

    skill, warnings = parse_skill(path)

    assert (skill.name, skill.body) == (expected, "Body")
    assert len(warnings) == 1
    assert warning in warnings[0]


def test_parser_cuts_an_overlong_name_and_description(tmp_path: Path) -> None:
    name, description = "n" * (MAX_NAME_CHARS + 1), "d" * (MAX_DESCRIPTION_CHARS + 1)
    path = _skill_file(tmp_path, name, f"---\nname: {name}\ndescription: {description}\n---\n")

    skill, warnings = parse_skill(path)

    assert skill.name == name[:MAX_NAME_CHARS]
    assert skill.description == description[:MAX_DESCRIPTION_CHARS]
    assert "name is over" in warnings[0]
    assert "description is over" in warnings[-1]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("description: >\n  Use when\n  things happen.\n", "Use when things happen."),
        ("description: Use when\n  things happen.\n", "Use when things happen."),
        ("description: Use when: café 😀\n", "Use when: café 😀"),
        ('description: Use when "quoted": it\'s \\ odd\n', 'Use when "quoted": it\'s \\ odd'),
        ("description: 2001-13-45\n", "2001-13-45"),
        ("description: 12\n", "12"),
    ],
)
def test_parser_reads_descriptions_that_other_clients_accept(
    tmp_path: Path, field: str, expected: str
) -> None:
    skill, warnings = parse_skill(_skill_file(tmp_path, "demo", f"---\nname: demo\n{field}---\n"))

    assert skill.description == expected
    assert warnings == []


def test_parser_keeps_a_yaml_boolean_word_as_a_name(tmp_path: Path) -> None:
    path = _skill_file(tmp_path, "on", "---\nname: on\ndescription: Use it.\n---\n")

    skill, warnings = parse_skill(path)

    assert skill.name == "on"
    assert warnings == []


def test_parser_handles_a_byte_order_mark_and_windows_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "demo" / "SKILL.md"
    path.parent.mkdir()
    text = "﻿---\r\nname: demo\r\ndescription: Use it.\r\n---\r\nOne\r\nTwo\r\n"
    path.write_bytes(text.encode())

    skill, warnings = parse_skill(path)

    assert (skill.name, skill.description, skill.body) == ("demo", "Use it.", "One\nTwo")
    assert warnings == []


def test_parser_survives_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "demo" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes(b"---\nname: demo\ndescription: Caf\xe9\n---\nBody\n")

    skill, _ = parse_skill(path)

    assert skill.description == "Caf�"


def test_parser_ends_the_frontmatter_at_the_first_closing_line(tmp_path: Path) -> None:
    text = "---\nname: demo\ndescription: Use it.\n---\nOne\n---\nTwo\n"

    skill, _ = parse_skill(_skill_file(tmp_path, "demo", text))

    assert skill.body == "One\n---\nTwo"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("", True),
        ("disable-model-invocation: true\n", False),
        ("disable-model-invocation: True\n", False),
        ("disable-model-invocation: false\n", True),
        ("disable-model-invocation: [true]\n", True),
    ],
)
def test_parser_reads_whether_the_model_may_load_the_skill(
    tmp_path: Path, field: str, expected: bool
) -> None:
    text = f"---\nname: demo\ndescription: Use it.\n{field}---\n"

    skill, _ = parse_skill(_skill_file(tmp_path, "demo", text))

    assert skill.model_invocable is expected


def test_discovery_finds_nothing_when_no_location_exists(tmp_path: Path, home: Path) -> None:
    assert discover_skills(tmp_path, home=home) == Discovery()


def test_discovery_combines_every_location_sorted_by_name(
    tmp_path: Path, project: Path, home: Path
) -> None:
    _add_skill(tmp_path / "extra", "zeta")
    _add_skill(project / SKILLS_DIR, "mid")
    _add_skill(home / SKILLS_DIR, "alpha")

    found = discover_skills(project, tmp_path / "extra", home)

    assert _names(found) == ["alpha", "mid", "zeta"]
    assert found.warnings == ()


def test_the_first_location_wins_a_name_and_the_warning_names_both_files(
    tmp_path: Path, project: Path, home: Path
) -> None:
    winner = _add_skill(tmp_path / "extra", "demo", "From the extra directory.")
    second = _add_skill(project / SKILLS_DIR, "demo", "From the project.")
    third = _add_skill(home / SKILLS_DIR, "demo", "From the user.")

    found = discover_skills(project, tmp_path / "extra", home)

    assert _paths(found) == [winner]
    assert len(found.warnings) == 2
    assert str(second) in found.warnings[0]
    assert str(third) in found.warnings[1]
    assert all(str(winner) in warning for warning in found.warnings)


def test_a_project_skill_beats_a_user_skill(project: Path, home: Path) -> None:
    winner = _add_skill(project / SKILLS_DIR, "demo", "From the project.")
    _add_skill(home / SKILLS_DIR, "demo", "From the user.")

    assert _paths(discover_skills(project, home=home)) == [winner]


def test_the_nearest_project_directory_wins(project: Path, home: Path) -> None:
    cwd = project / "packages" / "app"
    nearest = _add_skill(cwd / SKILLS_DIR, "demo", "Nearest.")
    _add_skill(project / SKILLS_DIR, "demo", "At the root.")

    assert _paths(discover_skills(cwd, home=home)) == [nearest]


def test_discovery_stops_at_the_git_root(tmp_path: Path, project: Path, home: Path) -> None:
    _add_skill(tmp_path / SKILLS_DIR, "outside")
    _add_skill(project / SKILLS_DIR, "inside")
    cwd = project / "packages"
    cwd.mkdir()

    assert _names(discover_skills(cwd, home=home)) == ["inside"]


def test_a_git_file_marks_the_root_of_a_worktree(tmp_path: Path, home: Path) -> None:
    root = tmp_path / "worktree"
    _add_skill(root / SKILLS_DIR, "inside")
    (root / ".git").write_text("gitdir: /somewhere/else\n")
    _add_skill(tmp_path / SKILLS_DIR, "outside")
    cwd = root / "packages"
    cwd.mkdir()

    assert _names(discover_skills(cwd, home=home)) == ["inside"]


def test_discovery_checks_only_the_cwd_outside_a_repository(tmp_path: Path, home: Path) -> None:
    cwd = tmp_path / "work"
    _add_skill(cwd / SKILLS_DIR, "here")
    _add_skill(tmp_path / SKILLS_DIR, "parent")

    assert _names(discover_skills(cwd, home=home)) == ["here"]


def test_discovery_looks_only_one_level_down(project: Path, home: Path) -> None:
    root = project / SKILLS_DIR
    _add_skill(root / "group", "nested")
    (root / "notes.md").write_text("not a skill")
    (root / "empty").mkdir()

    assert discover_skills(project, home=home) == Discovery()


def test_discovery_defaults_to_the_users_home(
    project: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_skill(home / SKILLS_DIR, "mine")
    monkeypatch.setenv("HOME", str(home))

    assert _names(discover_skills(project)) == ["mine"]


def test_discovery_survives_a_user_with_no_home(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_entry(uid: int) -> pwd.struct_passwd:
        raise KeyError(uid)

    _add_skill(project / SKILLS_DIR, "demo")
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", no_entry)

    assert _names(discover_skills(project)) == ["demo"]


def test_a_symlinked_skill_is_checked_against_the_name_of_its_link(
    tmp_path: Path, project: Path, home: Path
) -> None:
    text = "---\nname: linked\ndescription: Use it.\n---\n"
    target = _skill_file(tmp_path / "store", "v1", text).parent
    root = project / SKILLS_DIR
    root.mkdir(parents=True)
    (root / "linked").symlink_to(target, target_is_directory=True)

    found = discover_skills(project, home=home)

    assert [skill.directory for skill in found.skills] == [root / "linked"]
    assert found.warnings == ()


def test_a_skill_linked_into_a_second_location_counts_once(project: Path, home: Path) -> None:
    original = _add_skill(project / SKILLS_DIR, "demo")
    (home / SKILLS_DIR).mkdir(parents=True)
    (home / SKILLS_DIR / "demo").symlink_to(original.parent, target_is_directory=True)

    found = discover_skills(project, home=home)

    assert _paths(found) == [original]
    assert found.warnings == ()


def test_a_home_that_is_also_the_project_counts_once(project: Path) -> None:
    original = _add_skill(project / SKILLS_DIR, "demo")

    found = discover_skills(project, home=project)

    assert _paths(found) == [original]
    assert found.warnings == ()


@pytest.mark.parametrize("locked", ["directory", "file"])
def test_an_unreadable_skill_does_not_stop_discovery(
    project: Path, home: Path, locked: str
) -> None:
    root = project / SKILLS_DIR
    _add_skill(root, "good")
    path = _add_skill(root, "locked")
    target = path.parent if locked == "directory" else path
    target.chmod(0)
    try:
        found = discover_skills(project, home=home)
    finally:
        target.chmod(0o700)

    assert "good" in _names(found)  # root can still read the locked one, so check only this


def test_a_skills_path_that_is_a_file_is_reported(project: Path, home: Path) -> None:
    root = project / SKILLS_DIR
    root.parent.mkdir()
    root.write_text("not a directory")

    found = discover_skills(project, home=home)

    assert found.skills == ()
    assert "couldn't scan" in found.warnings[0]


def test_a_broken_skill_warns_and_the_others_still_load(project: Path, home: Path) -> None:
    root = project / SKILLS_DIR
    _add_skill(root, "good")
    broken = _skill_file(root, "broken", "no frontmatter")

    found = discover_skills(project, home=home)

    assert _names(found) == ["good"]
    assert len(found.warnings) == 1
    assert found.warnings[0].startswith(f"skipped {broken}:")


def test_a_cosmetic_problem_warns_with_the_file_it_came_from(project: Path, home: Path) -> None:
    text = "---\nname: demo\ndescription: Use it.\n---\n"
    path = _skill_file(project / SKILLS_DIR, "other", text)

    found = discover_skills(project, home=home)

    assert _names(found) == ["demo"]
    assert found.warnings[0].startswith(f"{path}: name 'demo'")


def test_for_model_leaves_out_skills_kept_for_users() -> None:
    hidden, shown = _skill("hidden", model_invocable=False), _skill("shown")

    assert Discovery((hidden, shown)).for_model == (shown,)


def test_catalog_lists_names_and_descriptions_without_paths() -> None:
    assert format_catalog([_skill("alpha", "Use for alpha."), _skill("beta", "Use for beta.")]) == (
        "The following skills provide specialized instructions for specific tasks. When a task "
        "matches a skill's description, call the skill tool with its name to load its full "
        "instructions before starting that work.\n"
        "\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>alpha</name>\n"
        "    <description>Use for alpha.</description>\n"
        "  </skill>\n"
        "  <skill>\n"
        "    <name>beta</name>\n"
        "    <description>Use for beta.</description>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def test_catalog_escapes_markup_and_leaves_quotes_readable() -> None:
    catalog = format_catalog([_skill("a<b", 'Use for <x> & "y" when it\'s late.')])

    assert "<name>a&lt;b</name>" in catalog
    assert '<description>Use for &lt;x&gt; &amp; "y" when it\'s late.</description>' in catalog


def test_activation_wraps_the_instructions_with_their_directory() -> None:
    assert format_activation(_skill()) == (
        '<skill_content name="demo">\n'
        "Do it.\n\n"
        "Skill directory: /skills/demo\n"
        "Relative paths in this skill start from that directory. Commands still start in the "
        "working directory, so use absolute paths or cd there first.\n"
        "</skill_content>"
    )


def test_activation_escapes_the_name_in_its_attribute() -> None:
    assert format_activation(_skill('a"b<c')).startswith('<skill_content name="a&quot;b&lt;c">\n')


def test_activation_keeps_instructions_up_to_the_limit() -> None:
    body = "x" * MAX_SKILL_CHARS

    text = format_activation(_skill(body=body))

    assert body in text
    assert "instructions cut" not in text


def test_activation_cuts_longer_instructions_and_says_where_the_rest_is() -> None:
    skill = _skill(body="x" * MAX_SKILL_CHARS + "tail")

    text = format_activation(skill)

    assert "tail" not in text
    assert "x" * MAX_SKILL_CHARS + "\n[instructions cut at 20,000 characters; " in text
    assert f"read {skill.path} for the rest]" in text


@pytest.mark.parametrize(
    "instruction", ["do it", "  do it  ", "see /skill:demo", "/skillful", "/skill:", "/skill: demo"]
)
def test_other_instructions_pass_through_unchanged(instruction: str) -> None:
    assert expand_command(instruction, [_skill()]) == instruction


@pytest.mark.parametrize(
    ("instruction", "request_text"),
    [
        ("/skill:demo", None),
        ("  /skill:demo\n", None),
        ("/skill:demo fix it", "fix it"),
        ("/skill:demo\nfix\n\nit", "fix\n\nit"),
    ],
)
def test_a_skill_command_expands_and_keeps_the_rest_of_the_request(
    instruction: str, request_text: str | None
) -> None:
    skill = _skill()
    activation = format_activation(skill)

    expanded = expand_command(instruction, [skill])

    assert expanded == (f"{activation}\n\n{request_text}" if request_text else activation)


@pytest.mark.parametrize(
    ("skills", "available"), [([], "available: none"), ([_skill()], "available: demo")]
)
def test_an_unknown_skill_command_fails_and_lists_what_exists(
    skills: list[Skill], available: str
) -> None:
    with pytest.raises(ValueError, match="unknown skill 'missing'") as error:
        expand_command("/skill:missing do it", skills)
    assert available in str(error.value)


def test_a_skill_command_can_reach_a_skill_kept_for_users() -> None:
    skill = _skill(model_invocable=False)

    assert expand_command("/skill:demo", [skill]) == format_activation(skill)


def test_tools_for_adds_the_skill_tool_only_when_there_are_skills() -> None:
    usual = [tool.definition for tool in TOOLS]

    assert tools_for([]).definitions == usual
    assert tools_for([_skill()]).definitions[:4] == usual
    assert [tool["name"] for tool in tools_for([_skill()]).definitions[4:]] == ["skill"]


def test_the_skill_tool_offers_exactly_the_skills_it_was_given_in_order() -> None:
    parameters: dict[str, Any] = (
        skill_tool([_skill("zeta"), _skill("alpha")]).definition["parameters"] or {}
    )

    assert parameters["properties"]["name"]["enum"] == ["alpha", "zeta"]


def test_only_the_skill_tool_keeps_its_results() -> None:
    tools = tools_for([_skill()])

    assert tools.keeps_result("skill")
    assert not tools.keeps_result("bash")
    assert not tools.keeps_result("missing")


def test_loading_a_skill_returns_its_activation_and_reports_it(tmp_path: Path) -> None:
    skill = _skill(body="One\nTwo")
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)

    assert _load(tools_for([skill]), "demo", context) == format_activation(skill)
    assert events == [
        ToolStarted("skill", "demo"),
        ToolFinished("skill", "demo", "2 lines", False, 0.0),
    ]


def test_loading_a_skill_twice_returns_a_short_note(tmp_path: Path) -> None:
    tools = tools_for([_skill()])
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)

    _load(tools, "demo", context)
    repeat = _load(tools, "demo", context)

    assert "already loaded" in repeat
    assert "Do it." not in repeat
    assert events[-1] == ToolFinished("skill", "demo", "already loaded", False, 0.0)


def test_each_run_remembers_its_own_loaded_skills(tmp_path: Path) -> None:
    skill = _skill()
    context = RunContext(cwd=tmp_path)

    _load(tools_for([skill]), "demo", context)

    assert _load(tools_for([skill]), "demo", context) == format_activation(skill)


def test_an_unknown_skill_comes_back_as_text_and_reports_a_failure(tmp_path: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)

    result = _load(tools_for([_skill()]), "missing", context)

    assert result == "Error: unknown skill 'missing'; choose one of demo."
    finished = events[-1]
    assert isinstance(finished, ToolFinished)
    assert finished.failed is True
    assert finished.summary in result
