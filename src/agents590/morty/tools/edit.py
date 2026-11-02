"""The edit tool: replace an exact string in a file, or change nothing and say why."""

from pathlib import Path
from typing import Any

from agents590.morty.tools.base import FileResult, ToolError, file_tool


def _edit(path: Path, arguments: dict[str, Any]) -> FileResult:
    old: str = arguments["old_string"]
    new: str = arguments["new_string"]
    if not old:
        raise ToolError("old_string must not be empty; use write to create a file.")
    if old == new:
        raise ToolError("old_string and new_string are the same, so there is nothing to change.")

    # newline="" keeps the file's own line endings through the read and the write
    try:
        with path.open(encoding="utf-8", newline="") as file:
            text = file.read()
    except UnicodeDecodeError as error:
        raise ToolError(f"{path} is not valid UTF-8; change it with bash instead.") from error

    count = text.count(old)
    if count == 0:
        raise ToolError(
            f"old_string was not found in {path}; read the file and copy the text exactly, "
            "without line numbers."
        )
    if count > 1 and not arguments["replace_all"]:
        raise ToolError(
            f"old_string matches {count} places in {path}; include more surrounding text to "
            "make it unique, or set replace_all to true."
        )
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(text.replace(old, new))

    summary = "replaced 1 occurrence" if count == 1 else f"replaced {count} occurrences"
    return FileResult(f"{summary.capitalize()} in {path}.", summary)


EDIT = file_tool(
    {
        "type": "function",
        "name": "edit",
        "description": (
            "Replace an exact string in a file. old_string must match exactly, without line "
            "numbers, and be unique unless replace_all is true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to change."},
                "old_string": {"type": "string", "description": "The exact text to replace."},
                "new_string": {"type": "string", "description": "The text to put in its place."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every match instead of requiring a single one.",
                },
            },
            "required": ["path", "old_string", "new_string", "replace_all"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    _edit,
)
