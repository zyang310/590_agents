# Aw, geez, I guess this is the Morty agent package—please be careful with the tools, okay?
"""Morty, an agent independent of Harbor."""

import time
from typing import cast

from openai.types.responses import ResponseInputParam
from openai.types.responses.response_input_item_param import FunctionCallOutput

from agents590.contracts import RunContext
from agents590.events import ModelResponded, StepStarted
from agents590.morty.prompt import SYSTEM_PROMPT
from agents590.morty.tools import TOOL_DEFINITIONS, run_tool
from agents590.morty.trajectory import Trajectory
from agents590.morty.trimming import mask_old_results
from agents590.provider import UNCOpenAISettings

# The deployment's real context window isn't known here, so keep this well below it:
# the next request also carries this step's tool results (each at most MAX_OUTPUT_CHARS).
MAX_CONTEXT_TOKENS = 200_000


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
        results: list[FunctionCallOutput] = []
        trajectory = Trajectory(context.trajectory_path, instruction, model, TOOL_DEFINITIONS)

        # Every step adds to the history, so the token cap below always ends this loop.
        step = 0
        while True:
            step += 1
            # tell whoever is watching (the CLI trace) what we're doing
            context.report(StepStarted(step, model))
            started = time.monotonic()
            response = await client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=history,
                store=False,
                background=False,
                tools=TOOL_DEFINITIONS,
            )

            # append output to history to pass into next iteration
            history += cast(
                ResponseInputParam,
                [item.model_dump(exclude_none=True) for item in response.output],
            )

            # grabs all calls from the output, keep iterating when there are more
            # calls to be made, else just returns since we are done.
            calls = [item for item in response.output if item.type == "function_call"]
            usage = response.usage
            trajectory.add_response(response)
            context.report(
                ModelResponded(
                    step=step,
                    text=response.output_text,
                    tool_calls=len(calls),
                    seconds=time.monotonic() - started,
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                )
            )
            if not calls:
                return response.output_text

            # input + output is the history the next request starts from
            used = usage.input_tokens + usage.output_tokens if usage else 0
            if used >= MAX_CONTEXT_TOKENS:
                return (
                    f"Stopped at {used:,} context tokens (cap {MAX_CONTEXT_TOKENS:,}) "
                    "without a final answer."
                )

            # runs the calls one at a time and in order, since later edits can depend on earlier
            for call in calls:
                result = await run_tool(call.name, call.arguments, context)
                trajectory.add_result(call.call_id, result)
                output: FunctionCallOutput = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
                history.append(output)
                results.append(output)

            # only the newest results stay in full; the calls themselves remain in history
            mask_old_results(results)
