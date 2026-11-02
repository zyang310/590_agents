"""The read tool: numbered lines from a text file, one page at a time."""

from pathlib import Path
from typing import Any

from agents590.morty.tools.base import FileResult, ToolError, count_lines, file_tool
from agents590.morty.trimming import MAX_OUTPUT_CHARS

BINARY_PROBE_BYTES = 8_192


def _read(path: Path, arguments: dict[str, Any]) -> FileResult:
    offset: int = 1 if arguments["offset"] is None else arguments["offset"]
    limit: int | None = arguments["limit"]
    if offset < 1 or (limit is not None and limit < 1):
        raise ToolError("offset and limit must be at least 1.")
    with path.open("rb") as file:
        if b"\0" in file.read(BINARY_PROBE_BYTES):
            raise ToolError(f"{path} looks like a binary file; inspect it with bash instead.")

    shown: list[str] = []
    room = MAX_OUTPUT_CHARS
    total = 0
    next_offset: int | None = None
    # newline="\n" leaves any \r visible, so the model can copy CRLF text exactly into an edit
    with path.open(encoding="utf-8", errors="replace", newline="\n") as file:
        for total, line in enumerate(file, 1):
            if total < offset:
                continue
            entry = f"{total:>6}\t" + line.removesuffix("\n")
            if len(shown) == limit or (shown and len(entry) >= room):
                next_offset = total
                break
            if len(entry) > room:
                # a line too long for any page is cut here, or the next page would stop at it too
                omitted = len(entry) - room
                entry = entry[:room] + f" [line cut; {omitted:,} characters omitted]"
                room = 0
            else:
                room -= len(entry) + 1
            shown.append(entry)

    if not shown:
        if total == 0:
            return FileResult("[empty file]", "empty file")
        raise ToolError(f"offset {offset} is past the end of {path}, which has {total} lines.")
    text = "\n".join(shown)
    if next_offset is not None:
        last = offset + len(shown) - 1
        text += f"\n[showing lines {offset}-{last}; continue with offset={next_offset}]"
    return FileResult(text, count_lines(len(shown)))


READ = file_tool(
    {
        "type": "function",
        "name": "read",
        "description": (
            "Read a text file as numbered lines. Output is limited, so use offset (first line, "
            "starting at 1) and limit (number of lines) for large files. The line numbers are "
            "not part of the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to read."},
                "offset": {
                    "type": ["integer", "null"],
                    "description": "The first line to read, or null to start at line 1.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "The most lines to return, or null for as many as fit.",
                },
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    _read,
)
