# 590 agents

Standalone Python 3.12 CLI agents for COMP 590. Each implementation lives in its
own package under `src/agents590/`; all share one CLI and typed interface. This
project owns model configuration and agent logic. The sibling `evals/` project
owns Harbor tasks and adapters. Harbor is not a dependency of this package.

## Setup and usage

### 1. Install dependencies (once)

Run this once after cloning, and again after changing entry points in `pyproject.toml`:

```sh
uv sync --locked
```

### 2. See which agents exist

```sh
uv run agents590 --list
```

### 3. Run an agent

Pick one of the two forms below. They run the same code.

```sh
# Installed command
uv run agents590 krisj_echo -i "Hello from the CLI"

# Equivalent, via the Python module
uv run python -m agents590 krisj_echo --instruction "The same shared entry point"
```

### CLI options

`-i` / `--instruction` passes one instruction. `--cwd` selects an existing working
directory, `--model` overrides the agent's model, and `--env-file` explicitly loads
configuration without overriding variables already exported in the shell.

While an agent runs, the CLI draws its progress on stderr: one block per model step
and per shell command (exit code, timing, and the head and tail of its output), then
totals for steps, commands and tokens. stdout carries only the final answer, so pipes
and Harbor's `stdout.txt` are unchanged. `-q` / `--quiet` hides the trace. Agents feed
it by calling `context.report(...)` with the events in `agents590.events`; agents that
report nothing still get the header and summary.

The `krisj_echo` agent returns the instruction without tools or API requests. The
`krisj_starter` agent is the migrated course scaffold: its model/tool loop intentionally
raises `NotImplementedError` until implemented. Its settings already support
per-run model overrides through `RunContext.model`.

For use outside this checkout:

```sh
uv tool install --editable /absolute/path/to/590_agents
agents590 krisj_echo -i "An instruction from another project"
```

Changes to implementation source are picked up by new invocations. Reinstall or
run `uv sync` after changing entry-point registrations in `pyproject.toml`.

## Add an agent

1. Create a package such as `src/agents590/my_agent/`.
2. Implement a no-argument class with `async run(instruction: str, context:
   RunContext) -> str`. Honor `context.cwd` when executing tools and use
   `context.model` for the explicit model override.
3. Register `my-agent = "agents590.my_agent:MyAgent"` under
   `[project.entry-points."agents590.agents"]`, then run `uv sync`.
4. Add pytest tests and run `make check` and `make test`.

`agents590 my-agent -i "..."` and `python -m agents590 my-agent -i "..."`
then use the same implementation. Entry points also allow separately installed
packages to register additional agents. Duplicate names produce an error.

## UNC Azure configuration

```sh
cp .env.example .env
# Fill in your UNC subscription key and an enabled deployment alias.
uv run agents590 krisj_starter --env-file .env --model openai/YOUR_DEPLOYMENT -i "..."
```

The coding loop remains a scaffold. To explicitly make one billable connectivity
request through the migrated smoke command, run `make smoke`.

| Setting | Azure alias (takes precedence) | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | `AZURE_OPENAI_API_KEY` | UNC subscription key |
| `OPENAI_MODEL` | `AZURE_OPENAI_DEPLOYMENT` | Default deployment alias |
| `OPENAI_ENDPOINT` | `AZURE_OPENAI_ENDPOINT` | HTTPS origin; defaults to `https://azureaiapi.cloud.unc.edu` |

The explicit `--model` wins over environment defaults. The provider removes an
optional `openai/` prefix, preserves the deployment alias, and sends standard
OpenAI SDK requests to the UNC `/openai/v1/` gateway with bearer authentication.
Requests use `store=False` and `background=False`. Blank explicit settings fail;
public OpenAI endpoints are rejected. `OPENAI_BASE_URL` is ignored because the
client sets its base URL explicitly. Keep `.env` private; it is excluded from
Git and built wheels.

This configuration was migrated from `evals/src/coding_agent/provider.py`, whose
UNC routing was checked against LearnWithAI revision
`5c239db1b27abdb28c1969fbb1046ee0d700ea53`. Tests use mocked HTTP transports.

## Development

```sh
make check  # Ruff lint/format and Pyright static type checking
make test   # pytest, with no live model requests
```

VS Code recommends Python, Pylance, and Ruff. Pylance and the Pyright CLI share
the strict `[tool.pyright]` configuration in `pyproject.toml`; VS Code uses
`.venv/bin/python`. Python source is annotated and the distribution includes
`py.typed`. `make clean` removes generated caches and build output while keeping
the virtual environment and `.env`.

## Evaluate through Harbor

The sibling `evals/` repository supplies the generic installed-agent adapter:

```sh
cd ../evals
make agent AGENT=krisj_echo
make agent AGENT=my-agent MODEL=openai/YOUR_DEPLOYMENT ENV_FILE=../590_agents/.env
```

The second command requires that agent to be registered and implemented and may
make billable model requests. The adapter uploads a source snapshot containing
the package metadata, README, and `src/`, installs its locked runtime dependencies in an isolated sandbox
virtual environment, and invokes `python -m agents590 <agent> -i <instruction>`.
Credentials are read from the explicit environment file on the host and passed
as process environment values; the file itself is not uploaded. New agents need
no new Harbor adapter.
