#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

ENV_FILE="$script_dir/.env_prod_alpaca"
ENV_EXAMPLE_FILE="$script_dir/.env.example"

if [[ ! -f "$ENV_FILE" ]]; then
	echo "Missing $ENV_FILE" >&2
	if [[ -f "$ENV_EXAMPLE_FILE" ]]; then
		echo "Create it by copying the template:" >&2
		echo "  cp '$ENV_EXAMPLE_FILE' '$ENV_FILE'" >&2
	else
		echo "No template found at $ENV_EXAMPLE_FILE" >&2
	fi
	echo "Then fill in the required Alpaca credentials and settings." >&2
	exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_FILE="$script_dir/config/app_prod_alpaca.yml"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "Python not found: $PYTHON_BIN" >&2
	exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
	echo "Missing config file: $CONFIG_FILE" >&2
	exit 1
fi

exec "$PYTHON_BIN" "$script_dir/run_app.py" --config "$CONFIG_FILE" "$@"
