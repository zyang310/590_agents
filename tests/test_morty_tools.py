"""Exercise Morty's tools through `Toolset.run`, the same entry point the agent loop uses."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from agents590.contracts import RunContext
from agents590.events import AgentEvent, ToolFinished, ToolStarted
from agents590.morty.skills import Skill
from agents590.morty.tools import TOOLS, Toolset
from agents590.morty.tools.base import Tool
from agents590.morty.tools.skill import skill_tool
from agents590.morty.trimming import MAX_OUTPUT_CHARS


def _call(cwd: Path, name: str, **arguments: object) -> str:
    return asyncio.run(Toolset(TOOLS).run(name, json.dumps(arguments), RunContext(cwd=cwd)))


def _read(cwd: Path, offset: int | None = None, limit: int | None = None) -> str:
    return _call(cwd, "read", path="f.txt", offset=offset, limit=limit)


def _edit(cwd: Path, old: str, new: str, replace_all: bool = False) -> str:
    return _call(cwd, "edit", path="f.txt", old_string=old, new_string=new, replace_all=replace_all)


@pytest.fixture
def five_lines(tmp_path: Path) -> Path:
    (tmp_path / "f.txt").write_text("".join(f"line {n}\n" for n in range(1, 6)))
    return tmp_path


DEMO_SKILL = Skill("demo", "Use for demos.", Path("/skills/demo/SKILL.md"), "Do it.")


@pytest.mark.parametrize(
    "tool", [*TOOLS, skill_tool([DEMO_SKILL])], ids=lambda tool: tool.definition["name"]
)
def test_definitions_satisfy_strict_mode(tool: Tool) -> None:
    parameters: dict[str, Any] = tool.definition["parameters"] or {}
    assert tool.definition["strict"] is True
    assert parameters["additionalProperties"] is False
    assert sorted(parameters["required"]) == sorted(parameters["properties"])


def test_read_numbers_the_requested_range_and_says_where_to_continue(five_lines: Path) -> None:
    assert _read(five_lines, offset=2, limit=2) == (
        "     2\tline 2\n     3\tline 3\n[showing lines 2-3; continue with offset=4]"
    )


def test_read_that_reaches_the_last_line_has_no_hint(five_lines: Path) -> None:
    assert _read(five_lines, offset=4, limit=2) == "     4\tline 4\n     5\tline 5"
    assert _read(five_lines, offset=5) == "     5\tline 5"


def test_paging_through_a_large_file_stays_in_budget_and_reaches_every_line(
    tmp_path: Path,
) -> None:
    (tmp_path / "f.txt").write_text("".join(f"line {n:>30}\n" for n in range(1, 2001)))

    seen: list[str] = []
    pages = 0
    offset: int | None = None
    while True:
        page = _read(tmp_path, offset=offset)
        pages += 1
        assert len(page) < MAX_OUTPUT_CHARS + 100
        seen += re.findall(r"^\s*(\d+)\t", page, re.MULTILINE)
        following = re.search(r"continue with offset=(\d+)\]$", page)
        if following is None:
            break
        offset = int(following[1])

    assert pages > 1
    assert seen == [str(n) for n in range(1, 2001)]


def test_an_oversized_line_is_cut_and_the_next_page_moves_past_it(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("short\n" + "x" * 20_000 + "\nafter\n")

    first = _read(tmp_path)
    assert first == "     1\tshort\n[showing lines 1-1; continue with offset=2]"

    second = _read(tmp_path, offset=2)
    assert "[line cut; 10,007 characters omitted]" in second
    assert second.endswith("[showing lines 2-2; continue with offset=3]")
    assert len(second) < MAX_OUTPUT_CHARS + 100

    assert _read(tmp_path, offset=3) == "     3\tafter"


def test_an_oversized_last_line_offers_no_further_page(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x" * 20_000)
    page = _read(tmp_path)
    assert "line cut" in page
    assert "continue with offset" not in page


def test_read_refuses_binary_files(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_bytes(b"abc\0def")
    assert "binary" in _read(tmp_path)


def test_read_replaces_invalid_utf8_rather_than_failing(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_bytes(b"caf\xe9\n")
    assert _read(tmp_path) == "     1\tcaf�"


def test_read_leaves_carriage_returns_visible(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_bytes(b"one\r\ntwo\r\n")
    assert _read(tmp_path) == "     1\tone\r\n     2\ttwo\r"


def test_read_past_the_end_reports_the_line_count(five_lines: Path) -> None:
    result = _read(five_lines, offset=9)
    assert result.startswith("Error: offset 9 is past the end")
    assert result.endswith("which has 5 lines.")


@pytest.mark.parametrize(("offset", "limit"), [(0, None), (-1, None), (None, 0)])
def test_read_rejects_offset_and_limit_below_one(
    five_lines: Path, offset: int | None, limit: int | None
) -> None:
    assert _read(five_lines, offset, limit) == "Error: offset and limit must be at least 1."


def test_read_of_an_empty_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("")
    assert _read(tmp_path) == "[empty file]"


def test_write_creates_parent_directories_relative_to_the_cwd(tmp_path: Path) -> None:
    result = _call(tmp_path, "write", path="a/b/c.txt", content="one\ntwo\n")
    assert (tmp_path / "a" / "b" / "c.txt").read_bytes() == b"one\ntwo\n"
    assert "Wrote 2 lines" in result


def test_write_replaces_existing_contents(five_lines: Path) -> None:
    _call(five_lines, "write", path="f.txt", content="only\n")
    assert (five_lines / "f.txt").read_text() == "only\n"


def test_write_accepts_an_absolute_path_outside_the_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    target = tmp_path / "outside.txt"
    _call(cwd, "write", path=str(target), content="hi")
    assert target.read_text() == "hi"


def test_edit_replaces_a_unique_match(five_lines: Path) -> None:
    result = _edit(five_lines, "line 3", "LINE THREE")
    assert "Replaced 1 occurrence" in result
    assert (five_lines / "f.txt").read_text().splitlines()[2] == "LINE THREE"


def test_edit_replace_all_changes_every_match(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a a a\n")
    assert "Replaced 3 occurrences" in _edit(tmp_path, "a", "b", replace_all=True)
    assert (tmp_path / "f.txt").read_text() == "b b b\n"


def test_edit_keeps_crlf_line_endings(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_bytes(b"one\r\ntwo\r\n")
    _edit(tmp_path, "one", "uno")
    assert (tmp_path / "f.txt").read_bytes() == b"uno\r\ntwo\r\n"


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("missing", "x", "was not found"),
        ("dup", "x", "matches 2 places"),
        ("", "x", "must not be empty"),
        ("same", "same", "nothing to change"),
    ],
)
def test_a_failed_edit_explains_itself_and_leaves_the_file_alone(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    original = b"dup dup same\n"
    (tmp_path / "f.txt").write_bytes(original)

    result = _edit(tmp_path, old, new)

    assert result.startswith("Error: ")
    assert expected in result
    assert (tmp_path / "f.txt").read_bytes() == original


def test_edit_refuses_invalid_utf8_rather_than_corrupting_the_file(tmp_path: Path) -> None:
    original = b"caf\xe9\n"
    (tmp_path / "f.txt").write_bytes(original)

    result = _edit(tmp_path, "caf", "tea")

    assert result.startswith("Error: ")
    assert "bash" in result
    assert (tmp_path / "f.txt").read_bytes() == original


def test_unknown_tool_comes_back_as_text(tmp_path: Path) -> None:
    assert _call(tmp_path, "nope") == "Error: unknown tool 'nope'."


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read", {"path": "missing.txt", "offset": None, "limit": None}),
        ("read", {"path": ".", "offset": None, "limit": None}),
        (
            "edit",
            {"path": "missing.txt", "old_string": "a", "new_string": "b", "replace_all": False},
        ),
        ("write", {"path": ".", "content": "x"}),
    ],
)
def test_file_system_failures_come_back_as_text(
    tmp_path: Path, name: str, arguments: dict[str, object]
) -> None:
    assert _call(tmp_path, name, **arguments).startswith("Error: ")


def test_file_tools_report_their_start_and_outcome(five_lines: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=five_lines, on_event=events.append)
    arguments = json.dumps({"path": "f.txt", "offset": 1, "limit": 2})

    asyncio.run(Toolset(TOOLS).run("read", arguments, context))

    started, finished = events
    assert started == ToolStarted("read", "f.txt")
    assert isinstance(finished, ToolFinished)
    assert (finished.name, finished.detail) == ("read", "f.txt")
    assert (finished.summary, finished.failed) == ("2 lines", False)


def test_file_tools_report_a_failure_with_its_error(tmp_path: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)
    arguments = json.dumps({"path": "missing.txt", "offset": None, "limit": None})

    result = asyncio.run(Toolset(TOOLS).run("read", arguments, context))

    finished = events[-1]
    assert isinstance(finished, ToolFinished)
    assert finished.failed is True
    assert finished.summary in result
