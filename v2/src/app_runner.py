import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from utils.logging import configure_logging
from configs import AppConfig
from app import App
from runtime_manager import RuntimeManagerOptions
from events.event_bus import EventBusOptions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Sleeve V2 Live Runner")

    p.add_argument(
        "--config",
        default="config/app.yml",
        help="Path to application config YAML file",
    )

    return p.parse_args()


def setup_logger(app_cfg: AppConfig) -> logging.Logger:
    # Configure global logging
    configure_logging(
        log_root=Path(app_cfg.runtime.log_root),
        level=app_cfg.runtime.log_level,
        log_to_file=app_cfg.runtime.log_to_file,
    )
    # Module-level logger (will inherit handlers from root_logger)
    return logging.getLogger("v2_live_runner")


def main():
    args = parse_args()

    # Load app config
    try:
        app_cfg = AppConfig.load_from_yaml(Path(args.config))
    except Exception as e:
        print(f"Error loading app config from {args.config}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Configure logging
    log = setup_logger(app_cfg)
    log.info("Starting V2 Live Runner")

    # Create App
    app = App(
        config=app_cfg,
        runtime_manager_options=RuntimeManagerOptions(),  # Use defaults
        event_bus_options=EventBusOptions(
            drop_if_full=False,  # Do not drop events for now
        ),
    )

    # TODO: Implement live runner logic here
    asyncio.run(app.run())

    log.info("V2 Live Runner shutdown complete")


if __name__ == "__main__":
    main()
