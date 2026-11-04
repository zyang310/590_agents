"""Exercise selfhost's host-side scripts on real git repositories, with a stand-in for docker."""

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

SELFHOST = Path(__file__).parent.parent / "selfhost"
FAKE_DOCKER = """#!/usr/bin/env bash
{
    echo CALL
    for argument in "$@"; do printf '%s\\n' "$argument"; done
} >>"$FAKE_DOCKER_LOG"
while [ $# -gt 0 ]; do
    if [ "$1" = --env-file ]; then cat "$2" >>"$FAKE_DOCKER_ENV"; fi
    [ "$1" = gate ] && exit "${FAKE_GATE_STATUS:-0}"
    shift
done
exit 0
"""


@dataclass(frozen=True)
class Workspace:
    """A stand-in for this repository, plus where its runs and the fake docker's logs go."""

    repo: Path
    runs: Path
    docker_log: Path
    docker_env: Path


def _git(cwd: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _script(workspace: Workspace, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(workspace.repo / "selfhost" / name), *arguments], capture_output=True, text=True
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    made = Workspace(
        repo=tmp_path / "repo",
        runs=tmp_path / "runs",
        docker_log=tmp_path / "docker.log",
        docker_env=tmp_path / "docker.env",
    )
    monkeypatch.setenv("MORTY_RUNS_DIR", str(made.runs))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(made.docker_log))
    monkeypatch.setenv("FAKE_DOCKER_ENV", str(made.docker_env))
    monkeypatch.delenv("FAKE_GATE_STATUS", raising=False)

    (made.repo / "selfhost").mkdir(parents=True)
    (made.repo / "src" / "agents590").mkdir(parents=True)
    for name in ("run.sh", "review.sh"):
        shutil.copy2(SELFHOST / name, made.repo / "selfhost" / name)
    (made.repo / "src" / "agents590" / "a.py").write_text("A = 1\n")
    (made.repo / "Makefile").write_text("all:\n")
    (made.repo / ".gitignore").write_text(".env\n")
    subprocess.run(["git", "init", "-q", str(made.repo)], check=True)
    _git(made.repo, "add", "--all")
    _git(made.repo, "commit", "-qm", "base")
    return made


def _finished_run(
    workspace: Workspace, change: Callable[[Path], None], gate_status: str = "0"
) -> str:
    """Leave behind what run.sh would after an agent ran `change` on its clone."""
    run_id = "test-run"
    run_dir = workspace.runs / run_id
    clone = run_dir / "clone"
    (run_dir / "out").mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(workspace.repo), str(clone)], check=True)
    _git(clone, "checkout", "-qb", "morty/run")
    change(clone)
    _git(clone, "add", "--all")
    if _git(clone, "status", "--porcelain"):
        _git(clone, "commit", "-qm", "work")
    _git(
        clone, "bundle", "create", "-q", str(run_dir / "out" / "run.bundle"), "refs/heads/morty/run"
    )
    (run_dir / "base").write_text(_git(workspace.repo, "rev-parse", "HEAD") + "\n")
    (run_dir / "gate-status").write_text(gate_status + "\n")
    (run_dir / "agent-status").write_text("0\n")
    return run_id


def _has_ref(workspace: Workspace, run_id: str) -> bool:
    return bool(_git(workspace.repo, "for-each-ref", f"refs/morty/{run_id}"))


def _change_src(clone: Path) -> None:
    (clone / "src" / "agents590" / "a.py").write_text("A = 2\n")


def test_review_shows_a_change_without_touching_the_working_tree(workspace: Workspace) -> None:
    run_id = _finished_run(workspace, _change_src)
    head = _git(workspace.repo, "rev-parse", "HEAD")

    done = _script(workspace, "review.sh", run_id)

    assert done.returncode == 0, done.stderr
    assert "src/agents590/a.py" in done.stdout
    assert "Gate: passed" in done.stdout
    assert "Outside src/agents590/" not in done.stdout
    assert _has_ref(workspace, run_id)
    assert _git(workspace.repo, "rev-parse", "HEAD") == head
    assert (workspace.repo / "src" / "agents590" / "a.py").read_text() == "A = 1\n"
    exported = workspace.runs / run_id / "export" / "src" / "agents590" / "a.py"
    assert exported.read_text() == "A = 2\n"


@pytest.mark.parametrize("path", ["Makefile", "tests/conftest.py", ".github/workflows/x.yml"])
def test_review_points_out_changes_outside_the_package(workspace: Workspace, path: str) -> None:
    def change(clone: Path) -> None:
        (clone / path).parent.mkdir(parents=True, exist_ok=True)
        (clone / path).write_text("changed\n")

    done = _script(workspace, "review.sh", _finished_run(workspace, change))

    assert done.returncode == 0, done.stderr
    outside = done.stdout.split("Outside src/agents590/")[1]
    assert f"  {path}\n" in outside


def test_review_rejects_symlinks_before_anything_follows_them(workspace: Workspace) -> None:
    secret = workspace.repo.parent / "secret.txt"
    secret.write_text("secret\n")

    def change(clone: Path) -> None:
        (clone / "src" / "agents590" / "notes.txt").symlink_to(secret)

    run_id = _finished_run(workspace, change)

    done = _script(workspace, "review.sh", run_id)

    assert done.returncode == 2
    assert "REJECTED" in done.stderr
    assert "src/agents590/notes.txt" in done.stderr
    assert not _has_ref(workspace, run_id)
    assert not (workspace.runs / run_id / "export").exists()


def test_review_rejects_rewritten_history(workspace: Workspace) -> None:
    def change(clone: Path) -> None:
        _git(clone, "checkout", "-q", "--orphan", "fresh")
        _git(clone, "commit", "-qm", "new history")
        _git(clone, "update-ref", "refs/heads/morty/run", "HEAD")

    run_id = _finished_run(workspace, change)

    done = _script(workspace, "review.sh", run_id)

    assert done.returncode == 2
    assert "rewrote history" in done.stderr
    assert not _has_ref(workspace, run_id)


def test_review_says_when_a_run_changed_nothing(workspace: Workspace) -> None:
    run_id = _finished_run(workspace, lambda clone: None)

    done = _script(workspace, "review.sh", run_id)

    assert done.returncode == 0, done.stderr
    assert "made no changes" in done.stdout
    assert not _has_ref(workspace, run_id)


def test_review_reports_a_failed_gate(workspace: Workspace) -> None:
    done = _script(workspace, "review.sh", _finished_run(workspace, _change_src, gate_status="1"))

    assert done.returncode == 0, done.stderr
    assert "Gate: FAILED" in done.stdout
    assert "Gate: passed" not in done.stdout


def test_review_never_runs_what_the_clone_planted(workspace: Workspace) -> None:
    marker = workspace.repo.parent / "planted_ran"
    planted = workspace.repo.parent / "planted.sh"
    planted.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
    planted.chmod(0o755)
    run_id = _finished_run(workspace, _change_src)
    clone = workspace.runs / run_id / "clone"
    _git(clone, "config", "core.fsmonitor", str(planted))
    for hook in ("post-checkout", "post-merge", "pre-commit", "reference-transaction"):
        shutil.copy2(planted, clone / ".git" / "hooks" / hook)

    # Planted scripts would run for git commands in the clone, which is why review.sh has none
    subprocess.run(["git", "-C", str(clone), "status"], capture_output=True)
    assert marker.exists()
    marker.unlink()

    done = _script(workspace, "review.sh", run_id)

    assert done.returncode == 0, done.stderr
    assert not marker.exists()
    assert "fsmonitor" not in (workspace.repo / ".git" / "config").read_text()


def test_review_needs_a_real_bundle(workspace: Workspace) -> None:
    assert _script(workspace, "review.sh", "no-such-run").returncode == 64

    run_id = _finished_run(workspace, _change_src)
    bundle = workspace.runs / run_id / "out" / "run.bundle"
    bundle.unlink()
    bundle.symlink_to(workspace.repo / "Makefile")
    assert _script(workspace, "review.sh", run_id).returncode == 64


def _docker_calls(workspace: Workspace) -> list[list[str]]:
    calls = workspace.docker_log.read_text().split("CALL\n")[1:]
    return [call.splitlines() for call in calls]


def _volumes(call: list[str]) -> set[str]:
    return {call[i + 1] for i, argument in enumerate(call) if argument == "--volume"}


def _run_dir(workspace: Workspace) -> Path:
    return next(workspace.runs.iterdir())


def test_run_gives_the_agent_only_its_clone_and_logs(workspace: Workspace) -> None:
    done = _script(workspace, "run.sh", "--model", "some-model", "-i", "do it")

    assert done.returncode == 0, done.stderr
    agent = _docker_calls(workspace)[0]
    run_dir = _run_dir(workspace)
    assert _volumes(agent) == {f"{run_dir}/clone:/work", f"{run_dir}/logs:/logs"}
    assert not any("docker.sock" in argument for argument in agent)
    assert {"--cap-drop=ALL", "--security-opt=no-new-privileges"} <= set(agent)
    assert agent[-5:] == ["run", "morty", "do it", "--model", "some-model"]


def test_run_packages_and_tests_with_no_network(workspace: Workspace) -> None:
    done = _script(workspace, "run.sh", "-i", "do it")

    assert done.returncode == 0, done.stderr
    _, bundle, gate = _docker_calls(workspace)
    run_dir = _run_dir(workspace)
    for call in (bundle, gate):
        assert call[call.index("--network") + 1] == "none"
        assert "--env-file" not in call
    assert bundle[-1] == "bundle"
    assert _volumes(bundle) == {f"{run_dir}/clone:/candidate:ro", f"{run_dir}/out:/out"}
    assert gate[-1] == "gate"
    assert _volumes(gate) == {f"{run_dir}/out:/bundle:ro"}


def test_run_passes_on_only_the_model_settings(workspace: Workspace) -> None:
    (workspace.repo / ".env").write_text(
        "# a comment\n"
        'OPENAI_API_KEY="quoted-key"   # with a note\n'
        "export OPENAI_MODEL=plain-model\n"
        "AZURE_OPENAI_ENDPOINT='https://example.invalid'\n"
        "NOT_MODEL_SECRET=must-not-pass\n"
    )

    done = _script(workspace, "run.sh", "-i", "do it")

    assert done.returncode == 0, done.stderr
    assert workspace.docker_env.read_text().splitlines() == [
        "OPENAI_API_KEY=quoted-key",
        "OPENAI_MODEL=plain-model",
        "AZURE_OPENAI_ENDPOINT=https://example.invalid",
    ]
    arguments = workspace.docker_log.read_text()
    assert "quoted-key" not in arguments
    assert "must-not-pass" not in arguments


def test_run_clones_committed_work_without_a_remote_or_ignored_files(workspace: Workspace) -> None:
    _git(workspace.repo, "remote", "add", "origin", "https://example.invalid/real.git")
    (workspace.repo / ".env").write_text("OPENAI_API_KEY=key\n")
    (workspace.repo / "Makefile").write_text("uncommitted\n")

    done = _script(workspace, "run.sh", "-i", "do it")

    assert done.returncode == 0, done.stderr
    assert "uncommitted changes are not part of the clone" in done.stderr
    clone = _run_dir(workspace) / "clone"
    assert _git(clone, "remote") == ""
    assert _git(clone, "branch", "--show-current") == "morty/run"
    assert not (clone / ".env").exists()
    assert (clone / "Makefile").read_text() == "all:\n"
    assert (workspace.repo / "Makefile").read_text() == "uncommitted\n"
    objects = workspace.repo / ".git" / "objects"
    assert not [path for path in objects.rglob("*") if path.is_file() and path.stat().st_nlink > 1]


def test_run_records_a_failed_gate_and_exits_nonzero(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_GATE_STATUS", "1")

    done = _script(workspace, "run.sh", "-i", "do it")

    assert done.returncode == 1
    assert "Gate: FAILED" in done.stdout
    assert (_run_dir(workspace) / "gate-status").read_text() == "1\n"


@pytest.mark.parametrize(
    "arguments", [[], ["-i"], ["--nope", "x"], ["-i", "do it", "--base", "no-such-ref"]]
)
def test_run_refuses_bad_arguments(workspace: Workspace, arguments: list[str]) -> None:
    done = _script(workspace, "run.sh", *arguments)

    assert done.returncode == 64
    assert not workspace.docker_log.exists()
