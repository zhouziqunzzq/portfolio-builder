from __future__ import annotations

import yaml

from v2.src.configs.config import AppConfig
from v2.src.runtime_manager import RuntimeManager, RuntimeManagerOptions


def test_runtime_manager_builds_singletons(tmp_path):
    # Minimal on-disk artifacts (UniverseManager won't read them during init,
    # but keeping them present makes the test more robust).
    membership_csv = tmp_path / "membership.csv"
    membership_csv.write_text(
        "ticker,date_added,date_removed,sector\nAAA,2020-01-01,,Tech\n",
        encoding="utf-8",
    )
    sectors_yaml = tmp_path / "sectors.yml"
    sectors_yaml.write_text("{}\n", encoding="utf-8")

    app_yml = tmp_path / "app.yml"
    raw = {
        "runtime": {"log_root": str(tmp_path / "logs"), "log_level": "INFO"},
        "universe_manager": {
            "membership_csv": str(membership_csv),
            "sectors_yaml": str(sectors_yaml),
        },
        "market_data_store": {"data_root": str(tmp_path / "prices"), "source": "yfinance"},
        "regime_engine": {},
        "trend_sleeve": {},
        "defensive_sleeve": {},
        "sideways_base_sleeve": {},
        "multi_sleeve_allocator": {
            "enable_friction_control": True,
            "friction_control_config": {"hysteresis_dw_min": 0.01},
        },
    }
    app_yml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    app_cfg = AppConfig.load_from_yaml(app_yml)
    rm = RuntimeManager.from_app_config(app_cfg, options=RuntimeManagerOptions(local_only=True))

    # `get` and `[]` should return the same singleton object.
    assert rm.get("mds") is rm["market_data_store"]
    assert rm.get("signals") is rm["signal_engine"]
    assert rm.get("allocator") is rm["multi_sleeve_allocator"]


def test_runtime_manager_unknown_key_raises(tmp_path):
    app_yml = tmp_path / "app.yml"
    raw = {
        "runtime": {"log_root": str(tmp_path / "logs"), "log_level": "INFO"},
        "universe_manager": {"membership_csv": str(tmp_path / "m.csv"), "sectors_yaml": str(tmp_path / "s.yml")},
        "market_data_store": {"data_root": str(tmp_path / "prices"), "source": "yfinance"},
        "regime_engine": {},
        "trend_sleeve": {},
        "defensive_sleeve": {},
        "sideways_base_sleeve": {},
        "multi_sleeve_allocator": {},
    }
    app_yml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    rm = RuntimeManager.from_app_config(AppConfig.load_from_yaml(app_yml))

    try:
        _ = rm["does_not_exist"]
        assert False, "Expected KeyError"
    except KeyError:
        pass
