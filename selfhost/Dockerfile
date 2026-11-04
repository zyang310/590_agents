# A frozen Morty plus everything the offline gate needs. selfhost/build.sh builds this from a
# committed tree, so nothing uncommitted or ignored (like .env) can reach the image.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates git make nodejs npm ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.28
# A clone's owner on the host doesn't match this user, and containers are the boundary here.
RUN git config --system --add safe.directory '*'

ARG MORTY_UID=1000
RUN useradd --uid "$MORTY_UID" --create-home morty

ENV UV_PYTHON_DOWNLOADS=never
ENV GIT_AUTHOR_NAME=Morty GIT_AUTHOR_EMAIL=morty@selfhost.invalid
ENV GIT_COMMITTER_NAME=Morty GIT_COMMITTER_EMAIL=morty@selfhost.invalid

# Root owns the frozen copy, so nothing a run does can change the harness driving it. Its venv
# stays off PATH, where a bare `pytest` in /work would import this copy instead of the clone's.
COPY . /opt/morty/source
RUN cd /opt/morty/source \
    && UV_PROJECT_ENVIRONMENT=/opt/morty/venv uv sync --locked --no-dev --no-editable --no-cache

# The gate has no network, so fill this user's package cache and dev venv now. Pyright fetches
# its own package through npm the first time it runs.
USER morty
WORKDIR /home/morty
ENV UV_PROJECT_ENVIRONMENT=/home/morty/venv
RUN cp -R /opt/morty/source warm \
    && cd warm \
    && uv sync --locked \
    && uv run --locked pyright --version \
    && cd .. \
    && rm -rf warm

ENTRYPOINT ["/opt/morty/source/selfhost/container.sh"]
