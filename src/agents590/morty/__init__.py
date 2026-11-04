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
# Limits on a whole run. Masking old tool results makes the history grow slowly, so without
# these a stuck model could spend thousands of billable steps before reaching the context cap.
MAX_STEPS = 200
MAX_TOTAL_TOKENS = 10_000_000
MAX_RUN_SECONDS = 30 * 60


def _stop_reason(step: int, context_tokens: int, total_tokens: int, seconds: float) -> str | None:
    """Say which limit the run has reached, or None while it may continue."""
    reached = None
    if context_tokens >= MAX_CONTEXT_TOKENS:
        reached = f"at {context_tokens:,} context tokens (cap {MAX_CONTEXT_TOKENS:,})"
    elif total_tokens >= MAX_TOTAL_TOKENS:
        reached = f"after {total_tokens:,} total tokens (cap {MAX_TOTAL_TOKENS:,})"
    elif step >= MAX_STEPS:
        reached = f"after {step:,} steps (cap {MAX_STEPS:,})"
    elif seconds >= MAX_RUN_SECONDS:
        reached = f"after {seconds:,.0f} seconds (cap {MAX_RUN_SECONDS:,})"
    return f"Stopped {reached} without a final answer." if reached else None


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

        # The step limit in _stop_reason always ends this loop.
        step = 0
        total_tokens = 0
        run_started = time.monotonic()
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
            total_tokens += used
            stop = _stop_reason(step, used, total_tokens, time.monotonic() - run_started)
            if stop:
                return stop

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
