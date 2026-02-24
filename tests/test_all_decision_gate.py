from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from algotrading.lib.alpha.base import BaseAlphaOutput, ScalarAlphaOutput
from algotrading.lib.alpha.sma import SMAAlpha
from algotrading.lib.alpha_engine.base import AlphaKey, AlphaView
from algotrading.lib.signal.gate.all import AllNewerThanDecisionGate
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import Timeframe, TimeframeUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REF = InstrumentRef("AAPL")
_TF = Timeframe(1, TimeframeUnit.MINUTE)
_T0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)


def _key(alpha_id: str = "alpha_0") -> AlphaKey:
    return AlphaKey(ref=_REF, tf=_TF, alpha_type=SMAAlpha, alpha_id=alpha_id)


def _view(*pairs: tuple[AlphaKey, BaseAlphaOutput | None]) -> AlphaView:
    return AlphaView(outputs=dict(pairs))


def _scalar(
    updated_ts: datetime, *, is_ready: bool, value: float = 1.0
) -> ScalarAlphaOutput:
    return ScalarAlphaOutput(updated_ts=updated_ts, value=value, is_ready=is_ready)


# A non-scalar output subclass to test the BaseAlphaOutput path (no is_ready check).
@dataclass(frozen=True)
class _RawOutput(BaseAlphaOutput):
    pass


# ---------------------------------------------------------------------------
# Empty gate
# ---------------------------------------------------------------------------


def test_empty_gate_always_passes():
    gate = AllNewerThanDecisionGate(alpha_keys=[])
    assert gate.should_emit(_view(), _T0) is True


# ---------------------------------------------------------------------------
# Missing / None output
# ---------------------------------------------------------------------------


def test_missing_key_in_view_returns_false():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    # Key is not present in the view at all
    assert gate.should_emit(_view(), _T0) is False


def test_none_output_in_view_returns_false():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    assert gate.should_emit(_view((key, None)), _T0) is False


# ---------------------------------------------------------------------------
# ScalarAlphaOutput – readiness checks
# ---------------------------------------------------------------------------


def test_scalar_not_ready_returns_false():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _scalar(_T0, is_ready=False)
    assert gate.should_emit(_view((key, output)), _T0) is False


def test_scalar_ready_and_fresh_returns_true():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _scalar(_T0, is_ready=True)
    assert gate.should_emit(_view((key, output)), _T0) is True


# ---------------------------------------------------------------------------
# Freshness boundary
# ---------------------------------------------------------------------------


def test_output_exactly_at_event_ts_is_fresh():
    """updated_ts == latest_event_ts should pass (gate checks strictly less-than)."""
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _scalar(_T0, is_ready=True)
    assert gate.should_emit(_view((key, output)), _T0) is True


def test_output_newer_than_event_ts_is_fresh():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _scalar(_T0 + timedelta(seconds=1), is_ready=True)
    assert gate.should_emit(_view((key, output)), _T0) is True


def test_stale_output_returns_false():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    stale_ts = _T0 - timedelta(seconds=1)
    output = _scalar(stale_ts, is_ready=True)
    assert gate.should_emit(_view((key, output)), _T0) is False


# ---------------------------------------------------------------------------
# Non-scalar outputs (BaseAlphaOutput subclass – no is_ready attribute)
# ---------------------------------------------------------------------------


def test_non_scalar_fresh_output_returns_true():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _RawOutput(updated_ts=_T0)
    assert gate.should_emit(_view((key, output)), _T0) is True


def test_non_scalar_stale_output_returns_false():
    key = _key()
    gate = AllNewerThanDecisionGate(alpha_keys=[key])
    output = _RawOutput(updated_ts=_T0 - timedelta(seconds=1))
    assert gate.should_emit(_view((key, output)), _T0) is False


# ---------------------------------------------------------------------------
# Multiple keys
# ---------------------------------------------------------------------------


def test_multiple_keys_all_ready_and_fresh_returns_true():
    key_a = _key("alpha_a")
    key_b = _key("alpha_b")
    gate = AllNewerThanDecisionGate(alpha_keys=[key_a, key_b])
    out_a = _scalar(_T0, is_ready=True)
    out_b = _scalar(_T0, is_ready=True)
    assert gate.should_emit(_view((key_a, out_a), (key_b, out_b)), _T0) is True


def test_multiple_keys_one_missing_returns_false():
    key_a = _key("alpha_a")
    key_b = _key("alpha_b")
    gate = AllNewerThanDecisionGate(alpha_keys=[key_a, key_b])
    out_a = _scalar(_T0, is_ready=True)
    # key_b absent from view
    assert gate.should_emit(_view((key_a, out_a)), _T0) is False


def test_multiple_keys_one_not_ready_returns_false():
    key_a = _key("alpha_a")
    key_b = _key("alpha_b")
    gate = AllNewerThanDecisionGate(alpha_keys=[key_a, key_b])
    out_a = _scalar(_T0, is_ready=True)
    out_b = _scalar(_T0, is_ready=False)
    assert gate.should_emit(_view((key_a, out_a), (key_b, out_b)), _T0) is False


def test_multiple_keys_one_stale_returns_false():
    key_a = _key("alpha_a")
    key_b = _key("alpha_b")
    gate = AllNewerThanDecisionGate(alpha_keys=[key_a, key_b])
    out_a = _scalar(_T0, is_ready=True)
    out_b = _scalar(_T0 - timedelta(seconds=1), is_ready=True)
    assert gate.should_emit(_view((key_a, out_a), (key_b, out_b)), _T0) is False
