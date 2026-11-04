#!/usr/bin/env bash
# Entry point inside the selfhost image. run.sh starts one short-lived container per subcommand.
set -euo pipefail

FROZEN=/opt/morty/source

# Run an agent on /work, then record everything it left as a commit on morty/run.
run_agent() {
    local agent=$1 instruction=$2 status=0
    shift 2
    /opt/morty/venv/bin/agents590 "$agent" --cwd /work -i "$instruction" "$@" || status=$?

    # The agent may have left work uncommitted or moved to another branch. Hooks are skipped
    # so one it planted can't block this commit.
    cd /work
    git add --all
    if ! git diff --cached --quiet; then
        git commit --quiet --no-verify --message "Morty: work left uncommitted by this run"
    fi
    git update-ref refs/heads/morty/run HEAD
    exit "$status"
}

# Package the clone's commits. From here on nothing reads the agent's .git directory again.
make_bundle() {
    git -C /candidate bundle create --quiet /out/run.bundle refs/heads/morty/run
}

check() {
    local name=$1
    shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        printf 'PASSED: %s\n' "$name"
    else
        printf 'FAILED: %s\n' "$name"
        failed=1
    fi
}

# Test exactly what would be merged, using this image's Makefile and tests where it matters: the
# bundle can change its own, so only the frozen ones show whether existing behavior survived.
run_gate() {
    failed=0
    export UV_OFFLINE=1
    git clone --quiet --branch morty/run /bundle/run.bundle /tmp/candidate
    cd /tmp/candidate
    check "lint and types" make -f "$FROZEN/Makefile" check
    check "the run's own tests" make -f "$FROZEN/Makefile" test
    rm -rf tests
    cp -R "$FROZEN/tests" tests
    check "the frozen copy's tests" make -f "$FROZEN/Makefile" test
    exit "$failed"
}

command=${1:-}
[ $# -gt 0 ] && shift
case "$command" in
    run) run_agent "$@" ;;
    bundle) make_bundle ;;
    gate) run_gate ;;
    *)
        echo "usage: container.sh run AGENT INSTRUCTION [cli options] | bundle | gate" >&2
        exit 64
        ;;
esac
