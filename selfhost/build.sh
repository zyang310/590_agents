#!/usr/bin/env bash
# Build the image holding a frozen Morty from a committed ref, HEAD by default. Building an older
# ref and running it with --image is how you roll back.
set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ref=${1:-HEAD}
commit=$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}") || {
    echo "build.sh: no commit named $ref" >&2
    exit 64
}
tag=morty-stage:$(git -C "$repo" rev-parse --short=12 "$commit")

if [ "$ref" = HEAD ] && [ -n "$(git -C "$repo" status --porcelain)" ]; then
    echo "build.sh: uncommitted changes are not included in the image" >&2
fi

git -C "$repo" archive "$commit" |
    docker build --file selfhost/Dockerfile --build-arg "MORTY_UID=$(id -u)" \
        --tag "$tag" --tag morty-stage:latest -

echo "Built $tag, now also morty-stage:latest"
