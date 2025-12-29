import pandas as pd
import pytest

from v2.src.allocator.multi_sleeve_allocator import MultiSleeveAllocator
from v2.src.allocator.multi_sleeve_config import MultiSleeveConfig


class _FakeRegimeEngine:
    def __init__(self):
        # MultiSleeveAllocator.__init__ grabs this, but should_rebalance() never uses it.
        self.signals = None


class _FakeSleeve:
    def __init__(self, *, should_rebalance_value: bool):
        self._should_rebalance_value = bool(should_rebalance_value)

    def should_rebalance(self, now):
        return self._should_rebalance_value


def _mk_allocator(
    *, sleeve_should_rebalance: bool, config: MultiSleeveConfig
) -> MultiSleeveAllocator:
    sleeves = {"stub": _FakeSleeve(should_rebalance_value=sleeve_should_rebalance)}
    return MultiSleeveAllocator(
        regime_engine=_FakeRegimeEngine(), sleeves=sleeves, config=config
    )


def test_should_rebalance_true_if_any_sleeve_requires_rebalance():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=False,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
    )
    alloc = _mk_allocator(sleeve_should_rebalance=True, config=cfg)

    assert alloc.should_rebalance("2025-12-18") is True


def test_should_rebalance_true_if_no_prior_regime_context():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=False,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    # With no cached regime context, allocator should force a rebalance.
    assert alloc.state.last_regime_context is None
    assert alloc.should_rebalance("2025-12-18") is True


def test_should_rebalance_true_when_regime_sampling_frequency_triggers():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=False,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-11-30")

    # Month boundary => should_rebalance=True
    assert alloc.should_rebalance(pd.Timestamp("2025-12-01")) is True


def test_should_rebalance_false_when_sleeves_and_regime_do_not_require():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=False,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-12-01")

    # Same month, later day => no rebalance required.
    assert alloc.should_rebalance("2025-12-15") is False


def test_should_rebalance_true_if_trend_filter_enabled_and_no_prior_trend_status():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=True,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
        trend_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    # Regime is up-to-date
    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-12-01")

    # Trend filter enabled but no cached status => rebalance required.
    assert alloc.state.last_trend_status is None
    assert alloc.should_rebalance("2025-12-15") is True


def test_should_rebalance_true_when_trend_sampling_frequency_triggers():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=True,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
        trend_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    # Regime is up-to-date
    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-12-01")

    # Trend status exists but is stale across month boundary.
    alloc.state.last_trend_status = "risk-on"
    alloc.state.last_trend_sample_ts = pd.Timestamp("2025-11-30")

    # Use a date after the cached regime sample ts to avoid the helper's
    # "current_ts must be after last_rebalance_ts" guard.
    assert alloc.should_rebalance("2025-12-02") is True


def test_should_rebalance_false_when_everything_is_fresh_including_trend():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=True,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
        trend_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-12-01")

    alloc.state.last_trend_status = "risk-on"
    alloc.state.last_trend_sample_ts = pd.Timestamp("2025-12-01")

    assert alloc.should_rebalance("2025-12-15") is False


def test_should_rebalance_propagates_backwards_time_error_from_helper():
    cfg = MultiSleeveConfig(
        trend_filter_enabled=False,
        enable_friction_control=False,
        sleeve_regime_weights={"bull": {"stub": 1.0, "cash": 0.0}},
        regime_sample_freq="M",
    )
    alloc = _mk_allocator(sleeve_should_rebalance=False, config=cfg)

    alloc.state.last_regime_context = ("bull", {"bull": 1.0})
    alloc.state.last_regime_sample_ts = pd.Timestamp("2025-12-10")

    with pytest.raises(ValueError):
        alloc.should_rebalance("2025-12-09")
