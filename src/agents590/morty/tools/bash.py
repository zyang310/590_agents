"""The bash tool: run a shell command with bounded time and bounded output."""

import asyncio
import time
from asyncio.subprocess import PIPE
from contextlib import suppress
from typing import Any

from agents590.contracts import RunContext
from agents590.events import CommandFinished, CommandStarted
from agents590.morty.tools.base import Tool
from agents590.morty.trimming import MAX_OUTPUT_CHARS, Capture, head_and_tail

BASH_TIMEOUT_SECONDS = 30.0
KILL_GRACE_SECONDS = 2.0


async def run_bash(
    command: str,
    context: RunContext,
    timeout: float = BASH_TIMEOUT_SECONDS,
) -> str:

    context.report(CommandStarted(command))
    started = time.monotonic()

    # create a process and and define proc as that handle
    proc = await asyncio.create_subprocess_shell(command, cwd=context.cwd, stdout=PIPE, stderr=PIPE)
    out_pipe, err_pipe = proc.stdout, proc.stderr
    assert out_pipe is not None and err_pipe is not None  # both are PIPEs
    stdout, stderr = Capture(), Capture()

    async def drain() -> None:
        await asyncio.gather(stdout.read(out_pipe), stderr.read(err_pipe), proc.wait())

    # reads both streams as they arrive, keeping only their head and tail, until the
    # process exits or the whole command has run for a timeout amount of time
    try:
        async with asyncio.timeout(timeout):
            await drain()
    except TimeoutError:
        proc.kill()

        # waits to reap this killed process, could leave a zombie process if not. The wait
        # only ends once the pipes are read to their end, so keep draining them, but not
        # forever: a child the shell left behind can hold them open.
        with suppress(TimeoutError):
            async with asyncio.timeout(KILL_GRACE_SECONDS):
                await drain()
        context.report(CommandFinished(command, None, "", "", time.monotonic() - started))
        return f"Command timed out after {timeout:g} seconds and was killed."

    out = stdout.text
    err = stderr.text
    elapsed = time.monotonic() - started
    context.report(CommandFinished(command, proc.returncode, out, err, elapsed))
    result = f"exit code: {proc.returncode} \n stdout: \n {out} \n stderr: \n {err}"

    # both streams together may still be over budget, so keep the start and end of the result
    return head_and_tail(result)


async def _run(arguments: dict[str, Any], context: RunContext) -> str:
    return await run_bash(arguments["command"], context, BASH_TIMEOUT_SECONDS)


BASH = Tool(
    {
        "type": "function",
        "name": "bash",
        "description": (
            "Run a shell command in the working directory and return its exit code, stdout and "
            f"stderr. Commands are killed after {BASH_TIMEOUT_SECONDS:g} seconds. Output over "
            f"{MAX_OUTPUT_CHARS:,} characters loses its middle, so pipe noisy commands through "
            "tail."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    _run,
)
