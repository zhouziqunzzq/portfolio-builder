import yaml

from v2.src.configs.config import AppConfig
from friction_control.friction_control_config import FrictionControlConfig


def test_app_config_hydrates_friction_control_config(tmp_path):
    cfg_path = tmp_path / "app.yml"
    raw = {
        "runtime": {"log_root": "data/logs", "log_level": "INFO"},
        "universe_manager": {
            "membership_csv": "data/sp500_membership.csv",
            "sectors_yaml": "config/sectors.yml",
        },
        "market_data_store": {"data_root": "data/prices", "source": "yfinance"},
        "regime_engine": {},
        "trend_sleeve": {},
        "defensive_sleeve": {},
        "sideways_base_sleeve": {},
        "multi_sleeve_allocator": {
            "enable_friction_control": True,
            "friction_control_config": {
                "hysteresis_dw_min": 0.01,
                "min_trade_notional_abs": 123.0,
                "min_trade_pct_of_aum": 0.004,
                "min_holding_rebalances": 2,
            },
        },
    }

    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = AppConfig.load_from_yaml(cfg_path)

    assert isinstance(
        cfg.multi_sleeve_allocator.friction_control_config, FrictionControlConfig
    )
    assert cfg.multi_sleeve_allocator.friction_control_config.hysteresis_dw_min == 0.01
    assert (
        cfg.multi_sleeve_allocator.friction_control_config.min_trade_notional_abs
        == 123.0
    )
    assert (
        cfg.multi_sleeve_allocator.friction_control_config.min_trade_pct_of_aum == 0.004
    )
    assert (
        cfg.multi_sleeve_allocator.friction_control_config.min_holding_rebalances == 2
    )
