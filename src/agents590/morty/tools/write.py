"""The write tool: create a file or replace its entire contents."""

from pathlib import Path
from typing import Any

from agents590.morty.tools.base import FileResult, count_lines, file_tool


def _write(path: Path, arguments: dict[str, Any]) -> FileResult:
    content: str = arguments["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)
    summary = count_lines(len(content.splitlines()))
    return FileResult(f"Wrote {summary} to {path}.", summary)


WRITE = file_tool(
    {
        "type": "function",
        "name": "write",
        "description": (
            "Create a file or replace its entire contents, creating parent directories as "
            "needed. Use edit to change part of an existing file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to write."},
                "content": {"type": "string", "description": "The file's full new contents."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    _write,
)
