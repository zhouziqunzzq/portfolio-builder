from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from algotrading.lib.alpha.ema import EMAAlpha, EMAAlphaConfig
from algotrading.lib.alpha.base import MarketDataAlphaInput
from algotrading.lib.eventing.md_events import BarCompleted, BarUpdated
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import BarKey, OHLCVBar, Timeframe, TimeframeUnit


def _make_bar_event(
    ref: InstrumentRef, tf: Timeframe, start_ts: datetime, close: float
) -> BarCompleted:
    key = BarKey(ref=ref, tf=tf, start_ts=start_ts)
    bar = OHLCVBar(
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=tf.seconds),
        o=close,
        h=close,
        l=close,
        c=close,
        v=1.0,
    )
    return BarCompleted(ts=start_ts.timestamp(), key=key, bar=bar)


def _to_alpha_input(event: BarCompleted) -> MarketDataAlphaInput:
    return MarketDataAlphaInput(event=event, ts=event.key.start_ts)


def test_ema_alpha_warmup_and_ready():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = EMAAlpha(EMAAlphaConfig(ref=ref, tf=tf, window=3))

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    out1 = alpha.update(_to_alpha_input(_make_bar_event(ref, tf, t0, 10.0)))
    assert alpha.ready() is False
    assert out1.is_ready is False
    assert math.isnan(out1.value)

    t1 = t0 + timedelta(minutes=1)
    out2 = alpha.update(_to_alpha_input(_make_bar_event(ref, tf, t1, 12.0)))
    assert alpha.ready() is False
    assert out2.is_ready is False
    assert math.isnan(out2.value)

    t2 = t1 + timedelta(minutes=1)
    out3 = alpha.update(_to_alpha_input(_make_bar_event(ref, tf, t2, 14.0)))
    assert alpha.ready() is True
    assert out3.is_ready is True
    assert out3.value == pytest.approx(12.0)

    t3 = t2 + timedelta(minutes=1)
    out4 = alpha.update(_to_alpha_input(_make_bar_event(ref, tf, t3, 16.0)))
    assert out4.is_ready is True
    assert out4.value == pytest.approx(14.0)


def test_ema_alpha_rejects_mismatched_event():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = EMAAlpha(EMAAlphaConfig(ref=ref, tf=tf, window=2))

    other_ref = InstrumentRef("MSFT")
    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    event = _make_bar_event(other_ref, tf, t0, 10.0)

    with pytest.raises(ValueError, match="ref/tf"):
        alpha.update(MarketDataAlphaInput(event=event, ts=event.key.start_ts))


def test_ema_alpha_rejects_unsupported_event_type():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = EMAAlpha(EMAAlphaConfig(ref=ref, tf=tf, window=2))

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    key = BarKey(ref=ref, tf=tf, start_ts=t0)
    bar = OHLCVBar(
        start_ts=t0,
        end_ts=t0 + timedelta(seconds=tf.seconds),
        o=10.0,
        h=10.0,
        l=10.0,
        c=10.0,
        v=1.0,
    )
    event = BarUpdated(ts=t0.timestamp(), key=key, bar=bar, prev=None)

    with pytest.raises(TypeError, match="Unsupported market data event"):
        alpha.update(MarketDataAlphaInput(event=event, ts=event.key.start_ts))
