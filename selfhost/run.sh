#!/usr/bin/env bash
# Run Morty on a throwaway clone inside a container, then package and test what it did. Nothing
# here changes this repo. Everything the agent touched stays in the run's directory, so never run
# git in that clone on this machine or open it in an editor: its .git/config is untrusted.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: selfhost/run.sh -i INSTRUCTION [--model NAME] [--agent NAME] [--base REF]
                       [--env-file PATH] [--image IMAGE]
EOF
    exit 64
}

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
runs_dir=${MORTY_RUNS_DIR:-$HOME/morty/runs}
instruction="" agent=morty base=HEAD env_file=$repo/.env image=morty-stage:latest
cli_options=()

while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || usage
    case "$1" in
        -i | --instruction) instruction=$2 ;;
        -m | --model) cli_options+=(--model "$2") ;;
        --agent) agent=$2 ;;
        --base) base=$2 ;;
        --env-file) env_file=$2 ;;
        --image) image=$2 ;;
        *) usage ;;
    esac
    shift 2
done
[ -n "$instruction" ] || usage

# Pass on only the model settings, as NAME=value lines for docker's --env-file, which keeps any
# quotes literally. This reads the file as text, so nothing in it gets executed.
model_env() {
    [ -f "$env_file" ] || return 0
    sed -nE "s/^[[:space:]]*(export[[:space:]]+)?((OPENAI|AZURE_OPENAI)_[A-Z_]+)[[:space:]]*=[[:space:]]*[\"']?([^\"'#]*[^\"'#[:space:]])?[\"']?[[:space:]]*(#.*)?$/\2=\4/p" "$env_file"
}

base_commit=$(git -C "$repo" rev-parse --verify --quiet "$base^{commit}") || {
    echo "run.sh: no commit named $base" >&2
    exit 64
}

if [ -n "$(git -C "$repo" status --porcelain)" ]; then
    echo "run.sh: uncommitted changes are not part of the clone" >&2
fi

run_id=$(date -u +%Y%m%d-%H%M%S)
run_dir=$runs_dir/$run_id
mkdir -p "$run_dir/out" "$run_dir/logs"
printf '%s\n' "$base_commit" >"$run_dir/base"

# A clone, not a worktree, which would share hooks and config with this repo. Without hardlinks
# the clone can't alter this repo's object files, and with no remote it has nowhere to push.
git clone --quiet --no-hardlinks "$repo" "$run_dir/clone"
git -C "$run_dir/clone" remote remove origin
git -C "$run_dir/clone" checkout --quiet -b morty/run "$base_commit"

# Never add a mount of the Docker socket, $HOME or this repo: each would undo the boundary.
limits=(--rm --init --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=512 --memory=4g)

echo "Run $run_id: $agent on $(git -C "$repo" rev-parse --short "$base_commit") using $image"
agent_status=0
docker run "${limits[@]}" \
    --env-file <(model_env) \
    --env AGENTS590_ATIF_PATH=/logs/trajectory.json \
    --volume "$run_dir/clone:/work" \
    --volume "$run_dir/logs:/logs" \
    --workdir /work \
    "$image" run "$agent" "$instruction" ${cli_options[@]+"${cli_options[@]}"} ||
    agent_status=$?
printf '%s\n' "$agent_status" >"$run_dir/agent-status"

docker run "${limits[@]}" --network none \
    --volume "$run_dir/clone:/candidate:ro" \
    --volume "$run_dir/out:/out" \
    "$image" bundle

gate_status=0
docker run "${limits[@]}" --network none \
    --volume "$run_dir/out:/bundle:ro" \
    "$image" gate 2>&1 | tee "$run_dir/gate.log" || gate_status=$?
printf '%s\n' "$gate_status" >"$run_dir/gate-status"

echo
echo "Agent exit status: $agent_status. Gate: $([ "$gate_status" = 0 ] && echo passed || echo FAILED)."
echo "Next: selfhost/review.sh $run_id"
[ "$agent_status" = 0 ] && [ "$gate_status" = 0 ]
