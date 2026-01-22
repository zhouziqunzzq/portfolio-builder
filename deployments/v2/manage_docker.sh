#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
deployments_dir="$script_dir"

usage() {
    echo "Usage:" >&2
    echo "  $0 {live_alpaca|live_publicdotcom|paper|all|all_live} {up|down|restart|ps} [args...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  ./manage_obs.sh up -d" >&2
    echo "  $0 live_alpaca up -d --remove-orphans" >&2
    echo "  $0 live_publicdotcom up -d" >&2
    echo "  $0 paper up -d --remove-orphans" >&2
    echo "  $0 paper restart" >&2
    echo "  $0 live_alpaca down" >&2
    echo "  $0 live_publicdotcom ps" >&2
    echo "  $0 all_live up -d   # bring up all live deployments (excludes paper)" >&2
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

# Helper: set environment-specific variables for a given env name
set_env_vars() {
    local env="$1"
    case "$env" in
        live_alpaca|live)
            project="portfolio_builder_live_alpaca"
            env_file="$deployments_dir/.env.live_alpaca"
            env_example_file="$deployments_dir/.env.live_alpaca.example"
            app_compose="$deployments_dir/docker-compose.live_alpaca.yml"
            ;;
        live_publicdotcom)
            project="portfolio_builder_live_publicdotcom"
            env_file="$deployments_dir/.env.live_publicdotcom"
            env_example_file="$deployments_dir/.env.live_publicdotcom.example"
            app_compose="$deployments_dir/docker-compose.live_publicdotcom.yml"
            ;;
        live_publicdotcom_roth)
            project="portfolio_builder_live_publicdotcom_roth"
            env_file="$deployments_dir/.env.live_publicdotcom_roth"
            env_example_file="$deployments_dir/.env.live_publicdotcom_roth.example"
            app_compose="$deployments_dir/docker-compose.live_publicdotcom_roth.yml"
            ;;
        paper)
            project="portfolio_builder_paper"
            env_file="$deployments_dir/.env.paper_alpaca"
            env_example_file="$deployments_dir/.env.paper_alpaca.example"
            app_compose="$deployments_dir/docker-compose.paper.yml"
            ;;
        *)
            return 1
            ;;
    esac
    return 0
}

# Support applying an action to all known environments (two flavors)
if [[ "$env_name" == "all" || "$env_name" == "all_live" ]]; then
    if [[ "$env_name" == "all" ]]; then
        envs=(live_alpaca live_publicdotcom live_publicdotcom_roth paper)
    else
        envs=(live_alpaca live_publicdotcom live_publicdotcom_roth)
    fi
    if [[ "$action" == "up" ]]; then
        if ! docker network inspect pb_obs_net >/dev/null 2>&1; then
            echo "Missing Docker network 'pb_obs_net' (global observability network)." >&2
            echo "Start the global observability stack first:" >&2
            echo "  ./manage_obs.sh up -d" >&2
            exit 1
        fi
    fi

    for e in "${envs[@]}"; do
        if ! set_env_vars "$e"; then
            echo "Unknown environment: $e" >&2
            continue
        fi

        # For 'up' ensure env file exists (skip otherwise)
        if [[ "$action" == "up" && ! -f "$env_file" ]]; then
            echo "Skipping $e: missing $env_file" >&2
            if [[ -f "$env_example_file" ]]; then
                echo "Create it by copying the template:" >&2
                echo "  cp '$env_example_file' '$env_file'" >&2
            fi
            continue
        fi

        echo "== $action: $e =="
        if [[ "$action" == "up" ]]; then
            docker compose -p "$project" -f "$deployments_dir/docker-compose.yml" -f "$app_compose" up --build "$@"
        else
            docker compose -p "$project" -f "$deployments_dir/docker-compose.yml" -f "$app_compose" "$action" "$@"
        fi
    done
    exit 0
fi

if ! set_env_vars "$env_name"; then
    echo "Unknown environment: $env_name" >&2
    usage
    exit 2
fi

# Compose file stack (base + env override)
compose_args=(
    -p "$project"
    -f "$deployments_dir/docker-compose.yml"
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
            echo "Then fill in the required broker credentials and settings." >&2
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
