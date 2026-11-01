"""Record a Morty run as an ATIF trajectory without depending on Harbor."""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast
from uuid import uuid4

from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseUsage,
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _tool_call(call: ResponseFunctionToolCall) -> dict[str, object]:
    arguments = json.loads(call.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    return {
        "tool_call_id": call.call_id,
        "function_name": call.name,
        "arguments": cast(dict[str, object], arguments),
    }


class Trajectory:
    """Rewrites `path` after every change, so a run killed mid-way still leaves a valid file."""

    def __init__(
        self, path: Path | None, instruction: str, model: str, tool: FunctionToolParam
    ) -> None:
        self._path = path
        self._model = model
        self._session_id = str(uuid4())
        self._agent = {
            "name": "morty",
            "version": version("590-agents"),
            "model_name": model,
            # ATIF nests tool definitions the way Chat Completions does, unlike Responses.
            "tool_definitions": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description"),
                        "parameters": tool["parameters"],
                    },
                }
            ],
        }
        self._steps: list[dict[str, object]] = []
        self._results: list[dict[str, str]] = []
        self._usages: list[ResponseUsage] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._add_step({"source": "user", "message": instruction})

    def add_response(self, response: Response) -> None:
        """Record one model inference; its tool results arrive through `add_result`."""
        step: dict[str, object] = {
            "source": "agent",
            "message": response.output_text,
            "model_name": self._model,
            "llm_call_count": 1,
        }
        calls = [item for item in response.output if item.type == "function_call"]
        if calls:
            self._results = []
            step["tool_calls"] = [_tool_call(call) for call in calls]
            step["observation"] = {"results": self._results}
        if response.usage is not None:
            self._usages.append(response.usage)
            step["metrics"] = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
        self._add_step(step)

    def add_result(self, call_id: str, content: str) -> None:
        """Attach what the model will observe for a call to the step that made it."""
        self._results.append({"source_call_id": call_id, "content": content})
        self._write()

    def _add_step(self, step: dict[str, object]) -> None:
        self._steps.append({"step_id": len(self._steps) + 1, "timestamp": _now(), **step})
        self._write()

    def _write(self) -> None:
        if self._path is None:
            return
        final_metrics = {"total_steps": len(self._steps)}
        if self._usages:
            final_metrics["total_prompt_tokens"] = sum(u.input_tokens for u in self._usages)
            final_metrics["total_completion_tokens"] = sum(u.output_tokens for u in self._usages)
        document = {
            "schema_version": "ATIF-v1.8",
            "session_id": self._session_id,
            "agent": self._agent,
            "steps": self._steps,
            "final_metrics": final_metrics,
        }
        # Replacing atomically keeps readers from ever seeing a half-written file.
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        temporary.write_text(json.dumps(document, indent=2))
        temporary.replace(self._path)
