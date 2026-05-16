#!/usr/bin/env sh

set -e
uv sync --all-extras --all-packages --all-groups
docker compose up -d --build

echo "waiting for postgres..."
until docker compose exec stardust_db pg_isready -U "${POSTGRES_USER:-stardust}" > /dev/null 2>&1; do
    sleep 1
done
sleep 3

uv run alembic upgrade head
