#!/usr/bin/env bash
# Bring one run's commits into this repo under refs/morty/ and show what needs a human look.
# A ref checks nothing out and runs nothing, so everything printed here is safe to read. The run's
# bundle is the only thing taken from the agent's side; its clone is never touched.
set -euo pipefail

[ $# -eq 1 ] || {
    echo "usage: selfhost/review.sh RUN_ID" >&2
    exit 64
}
run_id=$1
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
run_dir=${MORTY_RUNS_DIR:-$HOME/morty/runs}/$run_id
bundle=$run_dir/out/run.bundle
ref=refs/morty/$run_id

if [ ! -f "$bundle" ] || [ -L "$bundle" ]; then
    echo "review.sh: no bundle for run $run_id at $bundle" >&2
    exit 64
fi
base=$(cat "$run_dir/base")

reject() {
    git -C "$repo" update-ref -d "$ref"
    printf 'REJECTED run %s: %s\n' "$run_id" "$1" >&2
    exit 2
}

# fsck refuses malformed trees, such as paths containing .. or .git
git -C "$repo" -c transfer.fsckObjects=true fetch --quiet --force "$bundle" \
    "refs/heads/morty/run:$ref"

git -C "$repo" merge-base --is-ancestor "$base" "$ref" ||
    reject "it rewrote history instead of building on ${base:0:12}"

# Evals and tools on this machine would follow a symlink to files outside the tree. Submodules
# would fetch and run code that no diff here shows.
links=$(git -C "$repo" ls-tree -r "$ref" | awk '$1 == "120000" || $1 == "160000"')
[ -z "$links" ] || reject "it contains symlinks or submodules:
$links"

if [ "$(git -C "$repo" rev-parse "$ref")" = "$base" ]; then
    git -C "$repo" update-ref -d "$ref"
    echo "Run $run_id made no changes."
    exit 0
fi

export_dir=$run_dir/export
rm -rf "$export_dir"
mkdir -p "$export_dir"
git -C "$repo" archive "$ref" | tar -x -C "$export_dir"

gate_status=$(cat "$run_dir/gate-status" 2>/dev/null || echo missing)
echo "Run $run_id"
echo "Agent exit status: $(cat "$run_dir/agent-status" 2>/dev/null || echo missing)"
if [ "$gate_status" = 0 ]; then
    echo "Gate: passed. This shows the tests pass, not that the change is safe."
else
    echo "Gate: FAILED. See $run_dir/gate.log"
fi
echo
git -C "$repo" --no-pager diff --stat "$base" "$ref"

outside=$(git -C "$repo" diff --name-only --no-renames "$base" "$ref" |
    grep -v '^src/agents590/' || true)
if [ -n "$outside" ]; then
    echo
    echo "Outside src/agents590/. Read these first, since they can run on this machine once merged:"
    printf '%s\n' "$outside" | sed 's/^/  /'
fi

cat <<EOF

Read the diff:
  git diff $base $ref
Run the evals in ../agent_evals (billable):
  make agent AGENT=morty AGENTS_PROJECT="$export_dir" ENV_FILE="$repo/.env"
Accept, then freeze the result as the next Morty:
  git merge --no-ff $ref && selfhost/build.sh
Discard:
  git update-ref -d $ref && rm -rf "$run_dir"
EOF
