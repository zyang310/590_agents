# Project conventions

- This project owns standalone agent implementations, the CLI, and model configuration.
  Keep Harbor and other evaluation frameworks out of its runtime dependencies.
- Put each agent in its own package under `src/agents590/` and register its no-argument
  factory in `[project.entry-points."agents590.agents"]` in `pyproject.toml`.
- Every agent implements the shared async `Agent.run(instruction, context)` protocol.
  The shared CLI owns parsing, environment-file loading, and agent selection.
- Annotate Python code. Pylance and Pyright share the strict `[tool.pyright]` settings.
- Use pytest functions, plain assertions, fixtures, and parametrization. Mock model
  requests in tests; make live requests only when explicitly requested.
- Run `make check` and `make test` after changes. Run `uv sync` after changing entry points.
- Use `-i` / `--instruction` for a one-line instruction; preserve this CLI contract
  because the sibling eval project's Harbor adapter invokes it.
- Morty works on this repository only through `selfhost/`, never on this checkout directly.
