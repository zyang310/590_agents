"""Morty, an agent independent of Harbor."""

import asyncio
import json
from asyncio.subprocess import PIPE
from typing import cast

from openai.types.responses import FunctionToolParam, ResponseInputParam

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

MAX_BASH_OUTPUT_CHARS = 10_000
BASH_TIMEOUT_SECONDS = 30.0
MAX_STEPS = 25

BASH_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "bash",
    "description": "Run a shell command and return its exit code, stdout and stderr.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command to run."}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}



'''json
{
    "type": "function_call",
    "id": "fc_68a1...",             
    "call_id": "call_9xK2...",      
    "name": "bash",                 
    "arguments": '{"command":"ls -la"}', 
    "status": "completed",
}
'''

class Morty:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        # Start with settings = self.provider_settings(context), then open
        # settings.create_async_client() and use model=settings.model.
        # Send Responses requests with store=False and background=False.
        # Run shell tools with context.cwd as their explicit working directory.

        settings = self.provider_settings(context)
        client = settings.create_async_client()
        model = settings.model

        history: ResponseInputParam = [{"role": "user", "content": instruction}]

        for _ in range(MAX_STEPS):
            response = await client.responses.create(
                model=model,
                input=history,
                store=False,
                background=False,
                tools=[BASH_TOOL],
            )

            # append output to history to pass into next iteration
            history += cast(
                ResponseInputParam,
                [item.model_dump(exclude_none=True) for item in response.output],
            )

            # grabs all calls from the output, keep iterating when there are more
            # calls to be made, else just returns since we are done.
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text
            
            # calls our run bash function to execute the tool
            for call in calls:
                command = json.loads(call.arguments)["command"]
                result = await self.run_bash(command, context)
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

        return f"Stopped after {MAX_STEPS} steps without a final answer."

    async def run_bash(
        self,
        command: str,
        context: RunContext,
        timeout: float = BASH_TIMEOUT_SECONDS,
    ) -> str:

        # create a process and and define proc as that handle
        proc = await asyncio.create_subprocess_shell(
            command, cwd=context.cwd, stdout=PIPE, stderr=PIPE
        )

        # awais for the result of proc.communicate for a timeout amount of time
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()

            # waits to reap this killed process, could leave a zombie process if not
            await proc.wait()
            return f"Command timed out after {timeout:g} seconds and was killed."

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        result = f"exit code: {proc.returncode} \n stdout: \n {out} \n stderr: \n {err}"

        # limits the result to MAX_BASH_OUTPUT_CHARS char
        if len(result) > MAX_BASH_OUTPUT_CHARS:
            result = result[:MAX_BASH_OUTPUT_CHARS] + "\n[output truncated]"
        return result
