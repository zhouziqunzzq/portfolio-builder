#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

usage() {
	echo "Usage:" >&2
	echo "  $0 {live|paper} {up|down|restart|ps} [args...]" >&2
	echo "" >&2
	echo "Examples:" >&2
	echo "  ./manage_obs.sh up -d" >&2
	echo "  $0 paper up -d --remove-orphans" >&2
	echo "  $0 paper restart" >&2
	echo "  $0 live down" >&2
	echo "  $0 live ps" >&2
}

env_name="${1:-}"
action="${2:-}"
if [[ -z "$env_name" || -z "$action" ]]; then
	usage
	exit 2
fi
shift 2 || true

if ! command -v docker >/dev/null 2>&1; then
	echo "Docker not found (missing 'docker' command)." >&2
	exit 1
fi

case "$env_name" in
	live)
		project="portfolio_builder_live"
		env_file="$script_dir/.env.live_alpaca"
		env_example_file="$script_dir/.env.live_alpaca.example"
		app_compose="docker-compose.live.yml"
		;;
	paper)
		project="portfolio_builder_paper"
		env_file="$script_dir/.env.paper_alpaca"
		env_example_file="$script_dir/.env.paper_alpaca.example"
		app_compose="docker-compose.paper.yml"
		;;
	*)
		echo "Unknown environment: $env_name" >&2
		usage
		exit 2
		;;
esac

# Compose file stack (base + env override)
compose_args=(
	-p "$project"
	-f docker-compose.yml
	-f "$app_compose"
)

case "$action" in
	up)
		if [[ ! -f "$env_file" ]]; then
			echo "Missing $env_file" >&2
			if [[ -f "$env_example_file" ]]; then
				echo "Create it by copying the template:" >&2
				echo "  cp '$env_example_file' '$env_file'" >&2
			fi
			echo "Then fill in the required Alpaca credentials and settings." >&2
			exit 1
		fi
		if ! docker network inspect pb_obs_net >/dev/null 2>&1; then
			echo "Missing Docker network 'pb_obs_net' (global observability network)." >&2
			echo "Start the global observability stack first:" >&2
			echo "  ./manage_obs.sh up -d" >&2
			exit 1
		fi
		exec docker compose "${compose_args[@]}" up --build "$@"
		;;
	down)
		exec docker compose "${compose_args[@]}" down "$@"
		;;
	ps)
		exec docker compose "${compose_args[@]}" ps "$@"
		;;
	restart)
		exec docker compose "${compose_args[@]}" restart "$@"
		;;
	*)
		echo "Unknown action: $action" >&2
		usage
		exit 2
		;;
esac
