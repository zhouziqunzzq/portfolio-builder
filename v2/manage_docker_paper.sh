#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

action="${1:-}"
if [[ -z "$action" ]]; then
	echo "Usage: $0 {up|down} [args...]" >&2
	exit 2
fi
shift || true

ENV_FILE="$script_dir/.env.paper_alpaca"
ENV_EXAMPLE_FILE="$script_dir/.env.paper_alpaca.example"

if ! command -v docker >/dev/null 2>&1; then
	echo "Docker not found (missing 'docker' command)." >&2
	exit 1
fi

case "$action" in
	up)
		if [[ ! -f "$ENV_FILE" ]]; then
			echo "Missing $ENV_FILE" >&2
			if [[ -f "$ENV_EXAMPLE_FILE" ]]; then
				echo "Create it by copying the template:" >&2
				echo "  cp '$ENV_EXAMPLE_FILE' '$ENV_FILE'" >&2
			fi
			echo "Then fill in the required Alpaca credentials and settings." >&2
			exit 1
		fi
		exec docker compose \
			-p portfolio_builder_paper \
			-f docker-compose.yml \
			-f docker-compose.paper.yml \
			up --build "$@"
		;;
	down)
		exec docker compose \
			-p portfolio_builder_paper \
			-f docker-compose.yml \
			-f docker-compose.paper.yml \
			down "$@"
		;;
	*)
		echo "Unknown action: $action" >&2
		echo "Usage: $0 {up|down} [args...]" >&2
		exit 2
		;;
esac
