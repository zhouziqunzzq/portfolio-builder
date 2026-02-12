from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from algotrading.lib.alpha.sma import SMAAlpha, SMAAlphaConfig
from algotrading.lib.alpha_engine.base import AlphaKey
from algotrading.lib.alpha_engine.engine import AlphaEngine
from algotrading.lib.eventing.md_events import BarCompleted
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


def test_alpha_engine_subscribe_dedup_and_mismatch():
    engine = AlphaEngine()
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    cfg = SMAAlphaConfig(ref=ref, tf=tf, window=3)

    alpha1 = engine.subscribe(ref=ref, tf=tf, alpha_type=SMAAlpha, config=cfg)
    alpha2 = engine.subscribe(ref=ref, tf=tf, alpha_type=SMAAlpha, config=cfg)
    assert alpha1 is alpha2

    bad_cfg = SMAAlphaConfig(ref=InstrumentRef("MSFT"), tf=tf, window=3)
    with pytest.raises(ValueError, match="ref/tf"):
        engine.subscribe(ref=ref, tf=tf, alpha_type=SMAAlpha, config=bad_cfg)


def test_alpha_engine_routes_by_group():
    engine = AlphaEngine()
    ref_a = InstrumentRef("AAPL")
    ref_b = InstrumentRef("MSFT")
    tf = Timeframe(1, TimeframeUnit.MINUTE)

    engine.subscribe(
        ref=ref_a,
        tf=tf,
        alpha_type=SMAAlpha,
        config=SMAAlphaConfig(ref=ref_a, tf=tf, window=2),
    )
    engine.subscribe(
        ref=ref_b,
        tf=tf,
        alpha_type=SMAAlpha,
        config=SMAAlphaConfig(ref=ref_b, tf=tf, window=2),
    )

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    outputs = engine.update(_make_bar_event(ref_a, tf, t0, 10.0))
    assert len(outputs) == 1
    key = AlphaKey(ref=ref_a, tf=tf, alpha_type=SMAAlpha)
    assert key in outputs


def test_alpha_engine_ready_and_get():
    engine = AlphaEngine()
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    engine.subscribe(
        ref=ref,
        tf=tf,
        alpha_type=SMAAlpha,
        config=SMAAlphaConfig(ref=ref, tf=tf, window=2),
    )

    key = AlphaKey(ref=ref, tf=tf, alpha_type=SMAAlpha)
    assert engine.ready(key) is False
    assert engine.get(key) is None

    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    engine.update(_make_bar_event(ref, tf, t0, 10.0))
    assert engine.ready(key) is False
    assert engine.get(key) is not None
    assert engine.get(key).is_ready is False

    t1 = t0 + timedelta(minutes=1)
    engine.update(_make_bar_event(ref, tf, t1, 12.0))
    assert engine.ready(key) is True
    assert engine.get(key).is_ready is True


def test_alpha_engine_keys():
    engine = AlphaEngine()
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    engine.subscribe(
        ref=ref,
        tf=tf,
        alpha_type=SMAAlpha,
        config=SMAAlphaConfig(ref=ref, tf=tf, window=2),
    )

    keys = list(engine.keys())
    assert keys == [AlphaKey(ref=ref, tf=tf, alpha_type=SMAAlpha)]


def test_alpha_engine_reset_clears_outputs():
    engine = AlphaEngine()
    ref = InstrumentRef("AAPL")
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    engine.subscribe(
        ref=ref,
        tf=tf,
        alpha_type=SMAAlpha,
        config=SMAAlphaConfig(ref=ref, tf=tf, window=2),
    )

    key = AlphaKey(ref=ref, tf=tf, alpha_type=SMAAlpha)
    t0 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    engine.update(_make_bar_event(ref, tf, t0, 10.0))
    assert engine.get(key) is not None

    engine.reset()
    assert engine.get(key) is None
    assert engine.ready(key) is False
