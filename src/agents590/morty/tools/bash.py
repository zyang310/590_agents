"""The bash tool: run a shell command with bounded time and bounded output."""

import asyncio
import os
import signal
import sys
import time
from asyncio.subprocess import DEVNULL, PIPE
from contextlib import suppress
from typing import Any

from agents590.contracts import RunContext
from agents590.events import CommandFinished, CommandStarted
from agents590.morty.tools.base import Tool
from agents590.morty.trimming import MAX_OUTPUT_CHARS, Capture, head_and_tail

BASH_TIMEOUT_SECONDS = 30.0
KILL_GRACE_SECONDS = 2.0

# Commands don't need these, and without the model settings a stray `make smoke` or nested
# agent fails instead of spending money. A nested agent must not overwrite this run's log either,
# and the model should reach skills through its catalog, not by finding them in the environment.
HIDDEN_ENV_PREFIXES = ("OPENAI_", "AZURE_OPENAI_")
HIDDEN_ENV_NAMES = {"AGENTS590_ATIF_PATH", "AGENTS590_SKILLS_DIR"}
SECRET_ENV_WORDS = {
    "APIKEY",
    "AUTH",
    "CREDENTIAL",
    "CREDENTIALS",
    "KEY",
    "PASSWD",
    "PASSWORD",
    "SECRET",
    "TOKEN",
}


def _is_hidden(name: str) -> bool:
    upper = name.upper()
    return (
        upper in HIDDEN_ENV_NAMES
        or upper.startswith(HIDDEN_ENV_PREFIXES)
        or not SECRET_ENV_WORDS.isdisjoint(upper.split("_"))
    )


def command_environment() -> dict[str, str]:
    """This process's environment without secrets or the harness's own virtual environment.

    This prevents accidents, not attacks: a command running as the same user can still read its
    parent's environment. The boundary against that is the container described in selfhost/.
    """
    env = {name: value for name, value in os.environ.items() if not _is_hidden(name)}
    if sys.prefix == sys.base_prefix:
        return env

    # `uv run` activates the harness's venv, where a bare `pytest` in another checkout would
    # import this copy of agents590 and pass without testing that checkout's code. A venv the
    # user activated for their own project stays, since sys.prefix is not that venv.
    own_venv = os.path.realpath(sys.prefix)
    own_bin = os.path.join(own_venv, "bin")
    if os.path.realpath(env.get("VIRTUAL_ENV", "")) == own_venv:
        del env["VIRTUAL_ENV"]
    if "PATH" in env:
        entries = env["PATH"].split(os.pathsep)
        env["PATH"] = os.pathsep.join(e for e in entries if os.path.realpath(e) != own_bin)
    return env


def _kill_group(pid: int) -> None:
    """Kill the command and everything it started, which may already be gone."""
    # macOS answers EPERM rather than ESRCH for a group holding only zombies
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


async def run_bash(
    command: str,
    context: RunContext,
    timeout: float = BASH_TIMEOUT_SECONDS,
) -> str:

    context.report(CommandStarted(command))
    started = time.monotonic()

    # create a process and and define proc as that handle. Its own session makes it the leader
    # of a process group, so a timeout can stop everything it started and not just the shell.
    # Nobody can type at a command, so reading stdin ends at once rather than at the timeout.
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=context.cwd,
        env=command_environment(),
        stdin=DEVNULL,
        stdout=PIPE,
        stderr=PIPE,
        start_new_session=True,
    )
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
    except (TimeoutError, asyncio.CancelledError) as stopped:
        # Ctrl+C and a replaced UI run arrive as cancellation; neither reaches this session
        _kill_group(proc.pid)

        # waits to reap this killed process, could leave a zombie process if not. The wait
        # only ends once the pipes are read to their end, so keep draining them, but not
        # forever: a child that left the process group can hold them open.
        with suppress(TimeoutError):
            async with asyncio.timeout(KILL_GRACE_SECONDS):
                await drain()
        if isinstance(stopped, asyncio.CancelledError):
            raise
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
