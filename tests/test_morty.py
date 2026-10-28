"""Run Morty's loop against a mocked Responses API and record the events it reports."""

import asyncio
import json
import os
from functools import partial
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

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


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("OPENAI_MODEL", "course-deployment")


def _reply(item: dict[str, object]) -> httpx.Response:
    usage = {
        "input_tokens": 30,
        "output_tokens": 7,
        "total_tokens": 37,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    body = {"id": "resp_mock", "object": "response", "created_at": 0, "status": "completed"}
    return httpx.Response(200, json={**body, "output": [item], "usage": usage})


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
    assert events[0] == StepStarted(1, 50, "course-deployment")
    assert isinstance(first, ModelResponded)
    assert (first.tool_calls, first.input_tokens, first.output_tokens) == (1, 30, 7)
    assert events[2] == CommandStarted("pwd")
    assert isinstance(finished, CommandFinished)
    assert finished.exit_code == 0
    assert Path(finished.stdout.strip()).resolve() == tmp_path.resolve()
    assert isinstance(last, ModelResponded)
    assert (last.step, last.text, last.tool_calls) == (2, "all done", 0)
    assert json.loads(requests[1].content)["input"][-1]["type"] == "function_call_output"


def test_timed_out_command_reports_no_exit_code(tmp_path: Path) -> None:
    events: list[AgentEvent] = []
    context = RunContext(cwd=tmp_path, on_event=events.append)
    result = asyncio.run(Morty().run_bash("sleep 5", context, timeout=0.1))
    assert "timed out" in result
    finished = events[-1]
    assert isinstance(finished, CommandFinished)
    assert finished.exit_code is None
