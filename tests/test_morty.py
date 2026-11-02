"""Run Morty's loop against a mocked Responses API and record the events it reports."""

import asyncio
import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError

from agents590 import provider
from agents590.contracts import RunContext
from agents590.events import (
    AgentEvent,
    CommandFinished,
    CommandStarted,
    ModelResponded,
    StepStarted,
)
from agents590.morty import Morty
from agents590.morty.prompt import SYSTEM_PROMPT
from agents590.morty.tools.bash import run_bash
from agents590.morty.trimming import MASKED_RESULT, MAX_OUTPUT_CHARS


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("OPENAI_MODEL", "course-deployment")


USAGE = {
    "input_tokens": 30,
    "output_tokens": 7,
    "total_tokens": 37,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _reply(*items: dict[str, object], usage: dict[str, object] | None = USAGE) -> httpx.Response:
    body = {"id": "resp_mock", "object": "response", "created_at": 0, "status": "completed"}
    return httpx.Response(200, json={**body, "output": list(items), "usage": usage})


BASH_CALL = {
    "type": "function_call",
    "id": "fc_mock",
    "call_id": "call_mock",
    "name": "bash",
    "arguments": json.dumps({"command": "pwd"}),
    "status": "completed",
}
ANSWER = {
    "type": "message",
    "id": "msg_mock",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "all done", "annotations": []}],
}


def test_morty_reports_each_step_and_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter([_reply(BASH_CALL), _reply(ANSWER)])
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(replies)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)

    assert asyncio.run(Morty().run("where are you?", context)) == "all done"

    assert [type(event) for event in events] == [
        StepStarted,
        ModelResponded,
        CommandStarted,
        CommandFinished,
        StepStarted,
        ModelResponded,
    ]
    first, finished, last = events[1], events[3], events[5]
    assert events[0] == StepStarted(1, "course-deployment")
    assert isinstance(first, ModelResponded)
    assert (first.tool_calls, first.input_tokens, first.output_tokens) == (1, 30, 7)
    assert events[2] == CommandStarted("pwd")
    assert isinstance(finished, CommandFinished)
    assert finished.exit_code == 0
    assert Path(finished.stdout.strip()).resolve() == tmp_path.resolve()
    assert isinstance(last, ModelResponded)
    assert (last.step, last.text, last.tool_calls) == (2, "all done", 0)
    assert json.loads(requests[1].content)["input"][-1]["type"] == "function_call_output"
    assert json.loads(requests[0].content)["instructions"] == SYSTEM_PROMPT


def test_timed_out_command_reports_no_exit_code(tmp_path: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)
    result = asyncio.run(run_bash("sleep 5", context, timeout=0.1))
    assert "timed out" in result
    finished = events[-1]
    assert isinstance(finished, CommandFinished)
    assert finished.exit_code is None


def _run_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: list[httpx.Response],
) -> tuple[str, dict[str, Any]]:
    """Run Morty against canned replies and return its answer and parsed trajectory."""
    remaining = iter(replies)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: next(remaining)))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
    path = tmp_path / "logs" / "agent" / "trajectory.json"
    context = RunContext(cwd=tmp_path, trajectory_path=path)
    answer = asyncio.run(Morty().run("do it", context))
    return answer, json.loads(path.read_text())


def _bash_call(call_id: str, command: str) -> dict[str, object]:
    return {**BASH_CALL, "call_id": call_id, "arguments": json.dumps({"command": command})}


def _tool_call(call_id: str, name: str, **arguments: object) -> dict[str, object]:
    return {**BASH_CALL, "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


def test_trajectory_records_a_final_answer_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer, trajectory = _run_recorded(tmp_path, monkeypatch, [_reply(ANSWER)])

    assert answer == "all done"
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert UUID(trajectory["session_id"])
    assert trajectory["agent"]["name"] == "morty"
    assert trajectory["agent"]["version"] == version("590-agents")
    assert trajectory["agent"]["model_name"] == "course-deployment"
    definitions = trajectory["agent"]["tool_definitions"]
    assert [tool["function"]["name"] for tool in definitions] == [
        "bash",
        "read",
        "write",
        "edit",
    ]
    user, agent = trajectory["steps"]
    assert (user["step_id"], user["source"], user["message"]) == (1, "user", "do it")
    assert (agent["step_id"], agent["source"], agent["message"]) == (2, "agent", "all done")
    assert agent["model_name"] == "course-deployment"
    assert agent["llm_call_count"] == 1
    assert agent["metrics"] == {"prompt_tokens": 30, "completion_tokens": 7}
    assert "tool_calls" not in agent
    assert "observation" not in agent
    assert trajectory["final_metrics"] == {
        "total_steps": 2,
        "total_prompt_tokens": 30,
        "total_completion_tokens": 7,
    }


def test_trajectory_pairs_each_inference_with_its_ordered_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = [
        _reply(_bash_call("call_a", "echo first"), _bash_call("call_b", "echo second; exit 3")),
        _reply(_bash_call("call_c", "echo third")),
        _reply(ANSWER),
    ]
    _, trajectory = _run_recorded(tmp_path, monkeypatch, replies)

    steps = trajectory["steps"]
    assert [step["step_id"] for step in steps] == [1, 2, 3, 4]
    assert steps[1]["tool_calls"] == [
        {
            "tool_call_id": "call_a",
            "function_name": "bash",
            "arguments": {"command": "echo first"},
        },
        {
            "tool_call_id": "call_b",
            "function_name": "bash",
            "arguments": {"command": "echo second; exit 3"},
        },
    ]
    first, second = steps[1]["observation"]["results"]
    assert first["source_call_id"] == "call_a"
    assert "exit code: 0" in first["content"]
    assert "first" in first["content"]
    assert second["source_call_id"] == "call_b"
    assert "exit code: 3" in second["content"]
    assert steps[2]["observation"]["results"][0]["source_call_id"] == "call_c"
    assert trajectory["final_metrics"] == {
        "total_steps": 4,
        "total_prompt_tokens": 90,
        "total_completion_tokens": 21,
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [("sleep 5", "timed out"), ("yes | head -c 20000", "characters omitted")],
)
def test_trajectory_records_the_result_the_model_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, expected: str
) -> None:
    monkeypatch.setattr("agents590.morty.tools.bash.BASH_TIMEOUT_SECONDS", 0.5)
    replies = [_reply(_bash_call("call_a", command)), _reply(ANSWER)]
    _, trajectory = _run_recorded(tmp_path, monkeypatch, replies)

    assert expected in trajectory["steps"][1]["observation"]["results"][0]["content"]


def test_trajectory_omits_unavailable_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, trajectory = _run_recorded(tmp_path, monkeypatch, [_reply(ANSWER, usage=None)])

    assert "metrics" not in trajectory["steps"][1]
    assert trajectory["final_metrics"] == {"total_steps": 2}


def test_trajectory_survives_a_failed_model_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = [_reply(_bash_call("call_a", "echo kept")), httpx.Response(400, json={})]
    with pytest.raises(BadRequestError):
        _run_recorded(tmp_path, monkeypatch, replies)

    saved = tmp_path / "logs" / "agent" / "trajectory.json"
    steps = json.loads(saved.read_text())["steps"]
    assert "kept" in steps[1]["observation"]["results"][0]["content"]
    assert [path.name for path in saved.parent.iterdir()] == ["trajectory.json"]


def test_no_trajectory_is_written_without_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _reply(ANSWER)))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))

    assert asyncio.run(Morty().run("do it", RunContext(cwd=tmp_path))) == "all done"
    assert list(tmp_path.iterdir()) == []


def test_morty_stops_before_running_tools_once_the_context_hits_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agents590.morty.MAX_CONTEXT_TOKENS", 37)  # one USAGE reply
    answer, trajectory = _run_recorded(
        tmp_path, monkeypatch, [_reply(_bash_call("call_a", "touch ran"))]
    )

    assert answer == "Stopped at 37 context tokens (cap 37) without a final answer."
    assert not (tmp_path / "ran").exists()
    assert trajectory["steps"][1]["observation"] == {"results": []}


def test_long_output_keeps_its_start_and_end_so_a_late_error_survives(tmp_path: Path) -> None:
    command = "seq 1 5000; echo FATAL boom >&2; exit 3"
    result = asyncio.run(run_bash(command, RunContext(cwd=tmp_path)))

    assert result.startswith("exit code: 3")
    assert "1\n2\n3\n" in result
    assert "4999\n5000" in result
    assert "FATAL boom" in result
    assert "characters omitted" in result
    assert len(result) < MAX_OUTPUT_CHARS + 100


def test_a_flood_of_output_is_bounded_before_it_reaches_the_event_or_the_model(
    tmp_path: Path,
) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)
    result = asyncio.run(run_bash("head -c 2000000 /dev/zero | tr '\\0' x", context))

    finished = events[-1]
    assert isinstance(finished, CommandFinished)
    assert "1,990,000 characters omitted" in finished.stdout
    assert len(finished.stdout) < MAX_OUTPUT_CHARS + 100
    assert len(result) < MAX_OUTPUT_CHARS + 200


def test_an_endless_stream_is_cut_off_at_the_timeout(tmp_path: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)
    result = asyncio.run(run_bash("yes", context, timeout=0.5))

    assert "timed out" in result
    finished = events[-1]
    assert isinstance(finished, CommandFinished)
    assert finished.exit_code is None


def test_only_the_last_ten_tool_results_stay_in_the_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = [_bash_call(f"call_{n}", f"echo out{n}") for n in range(12)]
    replies = iter([_reply(*calls), _reply(ANSWER)])
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(replies)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))

    assert asyncio.run(Morty().run("do it", RunContext(cwd=tmp_path))) == "all done"

    items = json.loads(requests[1].content)["input"]
    outputs = [item["output"] for item in items if item.get("type") == "function_call_output"]
    assert [item.get("type") for item in items].count("function_call") == 12
    assert outputs[:2] == [MASKED_RESULT] * 2
    assert all(f"out{n}" in outputs[n] for n in range(2, 12))


def test_file_tool_calls_change_real_files_and_a_failed_edit_does_not_end_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edit = {"path": "notes.txt", "replace_all": False}
    replies = [
        _reply(_tool_call("call_w", "write", path="notes.txt", content="one\ntwo\n")),
        _reply(_tool_call("call_m", "edit", old_string="missing", new_string="x", **edit)),
        _reply(_tool_call("call_e", "edit", old_string="two", new_string="2", **edit)),
        _reply(_tool_call("call_r", "read", path="notes.txt", offset=None, limit=None)),
        _reply(ANSWER),
    ]

    answer, trajectory = _run_recorded(tmp_path, monkeypatch, replies)

    assert answer == "all done"
    assert (tmp_path / "notes.txt").read_text() == "one\n2\n"
    steps = trajectory["steps"][1:5]
    names = [step["tool_calls"][0]["function_name"] for step in steps]
    assert names == ["write", "edit", "edit", "read"]
    contents = [step["observation"]["results"][0]["content"] for step in steps]
    assert "Wrote 2 lines" in contents[0]
    assert contents[1].startswith("Error: old_string was not found")
    assert "Replaced 1 occurrence" in contents[2]
    assert contents[3] == "     1\tone\n     2\t2"
