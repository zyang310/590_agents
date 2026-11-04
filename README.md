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

### Terminal UI

`agents590-ui` runs the same registered agents in a terminal app, so instructions
need no shell quoting. `uv sync` already installs it. Agents that call a model read
their settings from the environment file, as described under
[UNC Azure configuration](#unc-azure-configuration):

```sh
uv run agents590-ui --env-file .env
```

Choose the agent, model, and working directory in the top bar, type an instruction,
and press Enter. Progress and the answer appear in the log. Esc clears the prompt,
Ctrl+L clears the log, and Ctrl+Q quits.

Name an agent to preselect it instead of `morty`, and use `--cwd` to set the starting
directory:

```sh
uv run agents590-ui krisj_echo --cwd ../my-project
```

For use outside this checkout, install the tool with Textual:

```sh
uv tool install --editable /absolute/path/to/590_agents --with textual
agents590-ui --env-file /absolute/path/to/590_agents/.env
```

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

## Skills

Morty supports [Agent Skills](https://agentskills.io/specification): folders of instructions,
and optionally scripts, that it loads only when a task needs them. Each skill is a directory
holding a `SKILL.md`, whose frontmatter names the skill and says when to use it:

```text
.agents/skills/
└── rename-python-variable/
    ├── SKILL.md
    └── scripts/new_name.py
```

```markdown
---
name: rename-python-variable
description: Use when renaming a variable in Python code.
---

From this skill's directory, run `python scripts/new_name.py <current_name>`.
```

Morty looks in these places, and the first skill to claim a name wins:

1. The directory named by `AGENTS590_SKILLS_DIR`. A relative path starts from `--cwd`, and a
   missing directory is an error. The Harbor adapter sets this for tasks with a `skills_dir`.
2. `.agents/skills` in the working directory and each parent up to the git root, nearest first.
   Outside a git repository, only the working directory counts.
3. `~/.agents/skills`.

Only the direct subdirectories of those folders count. The trace lists the skills a run found,
along with any problems. A file with no usable frontmatter or description is skipped, while
smaller problems, such as a name that doesn't match its directory, only warn.

The model sees each skill's name and description in the system prompt, and loads a skill's
instructions with the `skill` tool. They then stay in the conversation for the rest of the run,
up to 20,000 characters. A skill's scripts run through the `bash` tool, under its usual limits.

To load a skill yourself, start the instruction with its name. An unknown name fails before any
model request:

```sh
uv run agents590 morty -i "/skill:rename-python-variable rename total in invoice.py"
```

A skill with `disable-model-invocation: true` in its frontmatter loads only this way, and the
model isn't told about it. The terminal UI finds project and user skills, but doesn't read
`AGENTS590_SKILLS_DIR`.

**Project skills can steer Morty, and it loads them without asking.** A skill can tell the model
to run anything, so treat one like code from its repository. Run Morty on a repository you don't
trust only inside a container, the way `selfhost/` does for this one.

## Let Morty edit itself

Morty stops a run at 200 steps, 10 million total tokens, or 30 minutes. Its shell commands
can't see API keys, tokens or model settings, can't wait on stdin, and are killed along with
everything they started when they time out. These limits prevent accidents, but they don't
contain a misbehaving agent. For that, use `selfhost/` whenever Morty works on this repository.

A frozen copy of Morty, baked into a Docker image, works on a throwaway clone, so a bad edit
can't stop the next run. Its commits come back as a bundle, and only you can merge them. Docker
must be running, and the image is built from commits, so commit your changes first:

```sh
selfhost/build.sh                          # freeze HEAD as morty-stage:latest
selfhost/run.sh -i "Add a --version flag"  # billable; prints a run ID, default model from .env
selfhost/review.sh RUN_ID                  # fetch as refs/morty/RUN_ID and show what to check
```

`run.sh` starts three containers, one after another:

1. **Agent:** edits and tests `/work`, which is the clone, as often as it likes. Anything left
   uncommitted is committed for it when it exits.
2. **Bundle:** packages those commits, with the network off.
3. **Gate:** runs `make check` and the run's tests on the bundle, then runs the image's frozen
   tests against it, with the network off. Offline, a run that changes dependencies fails here.

`review.sh` rejects symlinks, submodules and rewritten history, and lists any change outside
`src/agents590/`, since those files can run on your machine once merged. It prints the commands
to read the diff, run the evals, and accept or discard the run. To accept a run, merge it and
rebuild the image. To roll back, run `selfhost/build.sh OLDER_REF`.

- **Stay out of the clone.** Never run git in `~/morty/runs/*/clone` or open it in an editor.
  The agent can write to its `.git/config`, and git would run what it planted there. `review.sh`
  reads only the bundle.
- **A passing gate isn't approval.** Removing a spending limit could still pass every test, so
  read the diff.
- **The key isn't protected.** It's readable inside the agent's container and the network is
  open, so this protects your machine and this repository, not the key.

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
