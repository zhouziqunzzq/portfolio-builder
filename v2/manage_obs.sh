#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

usage() {
	echo "Usage:" >&2
	echo "  $0 {up|down|restart|ps} [args...]" >&2
	echo "" >&2
	echo "Note:" >&2
	echo "  Observability services are configured with 'restart: unless-stopped' in docker-compose.obs.yml" >&2
	echo "  so they should come back automatically after a host reboot (unless you explicitly stop them)." >&2
	echo "" >&2
	echo "Examples:" >&2
	echo "  $0 up -d" >&2
	echo "  $0 restart otelcol" >&2
	echo "  $0 down" >&2
}

action="${1:-}"
if [[ -z "$action" ]]; then
	usage
	exit 2
fi
shift 1 || true

# Fail fast with actionable errors for required local configs.
if [[ "$action" == "up" ]]; then
	if [[ ! -f "alertmanager.yml" ]]; then
		echo "Missing alertmanager.yml; create it from alertmanager.yml.example" >&2
		exit 1
	fi
fi

if ! command -v docker >/dev/null 2>&1; then
	echo "Docker not found (missing 'docker' command)." >&2
	exit 1
fi

project="portfolio_builder_obs"
compose_args=(
	-p "$project"
	-f docker-compose.obs.yml
)

case "$action" in
	up)
		exec docker compose "${compose_args[@]}" up "$@"
		;;
	down)
		exec docker compose "${compose_args[@]}" down "$@"
		;;
	restart)
		exec docker compose "${compose_args[@]}" restart "$@"
		;;
	ps)
		exec docker compose "${compose_args[@]}" ps "$@"
		;;
	*)
		echo "Unknown action: $action" >&2
		usage
		exit 2
		;;
esac
