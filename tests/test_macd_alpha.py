from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from algotrading.lib.alpha.base import MarketDataAlphaInput
from algotrading.lib.alpha.macd import MACDAlpha, MACDAlphaConfig
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


def test_macd_alpha_warmup_and_ready():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = MACDAlpha(
        MACDAlphaConfig(
            ref=ref,
            tf=tf,
            ma_type="ema",
            fast_window=2,
            slow_window=4,
            signal_window=3,
        )
    )

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    out1 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t0, 10.0)))
    assert alpha.ready() is False
    assert out1.is_ready is False
    assert math.isnan(out1.macd)
    assert math.isnan(out1.signal)

    t1 = t0 + timedelta(minutes=1)
    out2 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t1, 11.0)))
    assert alpha.ready() is False
    assert out2.is_ready is False

    t2 = t1 + timedelta(minutes=1)
    out3 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t2, 12.0)))
    assert alpha.ready() is False
    assert out3.is_ready is False

    t3 = t2 + timedelta(minutes=1)
    out4 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t3, 13.0)))
    assert alpha.ready() is False
    assert out4.is_ready is False
    assert not math.isnan(out4.macd)
    assert math.isnan(out4.signal)

    t4 = t3 + timedelta(minutes=1)
    out5 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t4, 14.0)))
    assert alpha.ready() is False
    assert out5.is_ready is False
    assert not math.isnan(out5.macd)
    assert math.isnan(out5.signal)

    t5 = t4 + timedelta(minutes=1)
    out6 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t5, 15.0)))
    assert alpha.ready() is True
    assert out6.is_ready is True
    assert not math.isnan(out6.macd)
    assert not math.isnan(out6.signal)


def test_macd_alpha_uses_sma_signal_when_configured():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = MACDAlpha(
        MACDAlphaConfig(
            ref=ref,
            tf=tf,
            ma_type="sma",
            fast_window=2,
            slow_window=3,
            signal_window=2,
        )
    )

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t0, 10.0)))
    t1 = t0 + timedelta(minutes=1)
    alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t1, 12.0)))
    t2 = t1 + timedelta(minutes=1)
    out = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t2, 14.0)))
    assert out.is_ready is False
    assert math.isnan(out.signal)

    t3 = t2 + timedelta(minutes=1)
    out2 = alpha.update(MarketDataAlphaInput(event=_make_bar_event(ref, tf, t3, 16.0)))
    assert out2.is_ready is True
    assert not math.isnan(out2.signal)


def test_macd_alpha_rejects_mismatched_event():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = MACDAlpha(MACDAlphaConfig(ref=ref, tf=tf, fast_window=2, slow_window=3))

    other_ref = InstrumentRef("MSFT")
    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    event = _make_bar_event(other_ref, tf, t0, 10.0)

    with pytest.raises(ValueError, match="ref/tf"):
        alpha.update(MarketDataAlphaInput(event=event))


def test_macd_alpha_rejects_unsupported_event_type():
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    alpha = MACDAlpha(MACDAlphaConfig(ref=ref, tf=tf, fast_window=2, slow_window=3))

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
        alpha.update(MarketDataAlphaInput(event=event))
