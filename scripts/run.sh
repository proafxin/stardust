#!/usr/bin/env sh

set -e
uv sync --all-extras --all-packages --all-groups
docker compose up -d --build
